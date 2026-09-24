"""
DriftGuard Traffic Simulator Engine.
Generates live synthetic market trade streams (normal baseline & shifted drift bursts)
for real-time pipeline demonstrations.
"""

import time
import threading
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from backend.config import settings
from backend.db import log_feature_request, log_pipeline_event
from backend.model_manager import model_manager
from backend.drift_detector import drift_detector
from backend.retraining_service import retraining_service
from data.generator import generate_baseline_dataset, generate_drifted_dataset, FEATURE_COLUMNS, TARGET_COLUMN


class TrafficSimulator:
    """Manages background streaming of synthetic trade orders and drift injection."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_streaming = False
        self._streamed_count = 0
        self._lock = threading.Lock()

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def streamed_count(self) -> int:
        return self._streamed_count

    def _process_single_trade(self, row: pd.Series, is_drifted: int = 0) -> Dict[str, Any]:
        """Runs single trade through model and feature store."""
        feature_dict = {col: float(row[col]) for col in FEATURE_COLUMNS}
        ground_truth = int(row[TARGET_COLUMN]) if TARGET_COLUMN in row else None

        pred, prob = model_manager.predict(feature_dict)
        active_version = model_manager.version

        row_id = log_feature_request(
            features=feature_dict,
            prediction=pred,
            probability=prob,
            model_version=active_version,
            ground_truth=ground_truth,
            is_drifted_batch=is_drifted,
        )

        with self._lock:
            self._streamed_count += 1

        return {
            "id": row_id,
            "prediction": pred,
            "prob": prob,
            "is_manipulation": pred == 1,
            "model_version": active_version,
        }

    def _stream_worker(self, rate_per_sec: float = 5.0) -> None:
        """Continuously yields normal market trade vectors."""
        interval = 1.0 / max(0.5, rate_per_sec)
        normal_pool = generate_baseline_dataset(n_samples=1000, seed=int(time.time()) % 10000)
        idx = 0

        while not self._stop_event.is_set():
            row = normal_pool.iloc[idx % len(normal_pool)]
            idx += 1
            self._process_single_trade(row, is_drifted=0)

            # Periodically evaluate drift (every 10 requests)
            if self._streamed_count % 10 == 0:
                drift_res = drift_detector.check_drift()
                if drift_res.get("drift_detected") and settings.AUTO_RETRAIN_ENABLED and not retraining_service.is_retraining:
                    retraining_service.trigger_retrain(
                        reason="AUTO_DRIFT_DETECTED",
                        breached_features=drift_res.get("breached_features", {}),
                    )

            time.sleep(interval)

        self._is_streaming = False

    def start_stream(self, rate_per_sec: float = 5.0) -> Dict[str, Any]:
        """Starts continuous normal trade streaming thread."""
        if self._is_streaming:
            return {"status": "ALREADY_STREAMING", "message": "Stream is already running."}

        self._stop_event.clear()
        self._is_streaming = True
        self._thread = threading.Thread(target=self._stream_worker, args=(rate_per_sec,), daemon=True)
        self._thread.start()

        log_pipeline_event(
            event_type="STREAM_STARTED",
            title="Normal Traffic Stream Started",
            details=f"Streaming synthetic trades at {rate_per_sec} req/sec.",
            metadata={"rate": rate_per_sec}
        )
        return {"status": "STREAM_STARTED", "rate_per_sec": rate_per_sec}

    def stop_stream(self) -> Dict[str, Any]:
        """Stops active background streaming."""
        if not self._is_streaming:
            return {"status": "NOT_STREAMING", "message": "No active stream to stop."}

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._is_streaming = False

        log_pipeline_event(
            event_type="STREAM_STOPPED",
            title="Traffic Stream Stopped",
            details=f"Total streamed requests: {self._streamed_count}",
            metadata={"total_streamed": self._streamed_count}
        )
        return {"status": "STREAM_STOPPED", "total_streamed": self._streamed_count}

    def inject_drift_batch(self, batch_size: int = 50) -> Dict[str, Any]:
        """
        Immediately injects a burst of drifted trade vectors into the pipeline.
        This forces feature distributions to drift and triggers the KS-test alert.
        """
        drift_df = generate_drifted_dataset(n_samples=batch_size, seed=int(time.time()) % 10000)
        processed = []

        for _, row in drift_df.iterrows():
            res = self._process_single_trade(row, is_drifted=1)
            processed.append(res)

        # Immediate drift check
        drift_res = drift_detector.check_drift()

        log_pipeline_event(
            event_type="DRIFT_BATCH_INJECTED",
            title=f"Injected {batch_size} Drifted Trades",
            details=f"Features shifted. KS Drift Detected: {drift_res.get('drift_detected', False)} (p-values breach threshold).",
            metadata={"batch_size": batch_size, "drift_status": drift_res}
        )

        retrain_result = None
        if drift_res.get("drift_detected") and settings.AUTO_RETRAIN_ENABLED and not retraining_service.is_retraining:
            retrain_result = retraining_service.trigger_retrain(
                reason="DRIFT_INJECTION_TRIGGER",
                breached_features=drift_res.get("breached_features", {}),
            )

        return {
            "status": "DRIFT_INJECTED",
            "injected_count": len(processed),
            "drift_evaluation": drift_res,
            "retrain_result": retrain_result,
        }


# Global simulator instance
simulator = TrafficSimulator()
