"""
DriftGuard Automated Retraining & Hot-Swap Service.
Orchestrates data aggregation, challenger model retraining, MLflow logging,
validation gating, and atomic zero-downtime hot-swapping.
"""

import os
import time
import threading
from typing import Dict, Any, Optional
import pandas as pd

from backend.config import settings
from backend.db import (
    get_recent_feature_logs,
    log_pipeline_event,
    register_model_metadata,
)
from backend.model_manager import model_manager
from ml.model import train_model, save_model, FEATURE_COLUMNS, TARGET_COLUMN
from ml.mlflow_utils import log_and_register_model, promote_model_to_production
from ml.validator import validator
from data.generator import generate_drifted_dataset


class RetrainingService:
    """Manages retraining pipeline execution, MLflow registration, and hot-swap."""

    def __init__(self):
        self._retrain_lock = threading.Lock()
        self._is_retraining = False
        self._retrain_count = 0
        self._last_retrain_time = 0.0
        self._latest_comparison: Dict[str, Any] = {
            "status": "NO_RETRAIN_YET",
            "old_accuracy_pct": None,
            "new_accuracy_pct": None,
            "improvement_pct": None,
            "last_promoted_version": "v1",
        }

    @property
    def is_retraining(self) -> bool:
        return self._is_retraining

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    @property
    def last_retrain_time(self) -> float:
        return self._last_retrain_time

    @property
    def latest_comparison(self) -> Dict[str, Any]:
        return dict(self._latest_comparison)

    def trigger_retrain(self, reason: str = "MANUAL_TRIGGER") -> Dict[str, Any]:
        """
        Executes an end-to-end retraining pipeline:
        1. Aggregates baseline + drifted training data.
        2. Trains new challenger GradientBoosting / RandomForest model.
        3. Validates against validation & drifted test sets.
        4. Logs to MLflow & Model Registry.
        5. Hot-swaps the serving model on approval.
        """
        # Acquire retrain lock with 10s timeout to allow in-flight cycle to conclude gracefully
        if not self._retrain_lock.acquire(timeout=10.0):
            return {
                "success": False,
                "message": "Retraining pipeline is currently busy.",
                "status": "IN_PROGRESS",
            }

        try:
            self._is_retraining = True
            start_t = time.time()
            current_version = model_manager.version
            next_version_num = self._retrain_count + 2
            challenger_version = f"v{next_version_num}"

            log_pipeline_event(
                event_type="RETRAIN_TRIGGERED",
                title=f"Retraining Triggered ({reason})",
                details=f"Starting training run for challenger {challenger_version} triggered by {reason}.",
                metadata={"reason": reason, "current_version": current_version}
            )

            # 1. Prepare Augmented Training Dataset
            baseline_df = pd.read_csv(settings.BASELINE_TRAIN_DATA) if os.path.exists(settings.BASELINE_TRAIN_DATA) else pd.DataFrame()
            
            # Incorporate recent drifted data patterns
            recent_logs = get_recent_feature_logs(limit=200)
            if not recent_logs.empty and "ground_truth" in recent_logs.columns:
                valid_recent = recent_logs.dropna(subset=["ground_truth"]).copy()
                valid_recent[TARGET_COLUMN] = valid_recent["ground_truth"].astype(int)
            else:
                valid_recent = pd.DataFrame()

            # Synthesize drifted adaptation batch
            drift_adaptation = generate_drifted_dataset(n_samples=800, seed=int(time.time()) % 10000)

            train_frames = [baseline_df, drift_adaptation]
            if not valid_recent.empty:
                train_frames.append(valid_recent[FEATURE_COLUMNS + [TARGET_COLUMN]])
            
            combined_train_df = pd.concat(train_frames, ignore_index=True).sample(frac=1.0, random_state=42)
            val_df = pd.read_csv(settings.BASELINE_VAL_DATA) if os.path.exists(settings.BASELINE_VAL_DATA) else None

            # 2. Train Challenger Model
            model_type = "gb" if next_version_num % 2 == 0 else "rf"
            challenger_pipeline, meta = train_model(
                train_df=combined_train_df,
                val_df=val_df,
                model_type=model_type,
                random_state=42 + next_version_num,
            )

            # 3. Validation Gate
            current_model = model_manager._model
            is_approved, report = validator.validate_challenger(
                champion_model=current_model,
                challenger_model=challenger_pipeline,
            )

            drift_comp = report["drift_benchmark_comparison"]
            old_acc = drift_comp["old_model_drift_accuracy"]
            new_acc = drift_comp["new_model_drift_accuracy"]

            if is_approved:
                # 4. Log to MLflow and Register
                mlflow_meta = log_and_register_model(
                    model=challenger_pipeline,
                    params={"model_type": model_type, "retrain_trigger": reason, "version": challenger_version},
                    train_metrics=meta["train_metrics"],
                    val_metrics=meta["val_metrics"],
                    run_name=f"retrained_{challenger_version}_{model_type}",
                    tags={"pipeline": "DriftGuard", "stage": "Production", "version": challenger_version, "trigger": reason},
                    register=True,
                    tracking_uri=settings.MLFLOW_TRACKING_URI,
                )

                run_id = mlflow_meta.get("run_id", f"retrain_run_{challenger_version}")
                promote_model_to_production(challenger_version, tracking_uri=settings.MLFLOW_TRACKING_URI)

                # Persist artifact
                save_path = os.path.join(settings.MODEL_DIR, f"model_{challenger_version}.joblib")
                save_model(challenger_pipeline, save_path)
                save_model(challenger_pipeline, settings.ACTIVE_MODEL_PATH)

                # Cache in DB registry
                register_model_metadata(
                    version=challenger_version,
                    run_id=run_id,
                    model_type="GradientBoosting" if model_type == "gb" else "RandomForest",
                    train_accuracy=meta["train_metrics"]["accuracy"],
                    val_accuracy=meta["val_metrics"]["accuracy"],
                    val_f1=meta["val_metrics"]["f1_score"],
                    stage="Production",
                    artifact_path=save_path,
                )

                # 5. Hot-Swap Serving Model
                model_manager.hot_swap(
                    new_model=challenger_pipeline,
                    new_version=challenger_version,
                    new_metrics=meta["val_metrics"],
                    model_type=model_type,
                )

                self._retrain_count += 1
                self._last_retrain_time = time.time()
                self._latest_comparison = {
                    "status": "PROMOTED",
                    "old_version": current_version,
                    "new_version": challenger_version,
                    "old_accuracy_pct": old_acc,
                    "new_accuracy_pct": new_acc,
                    "improvement_pct": round(new_acc - old_acc, 1),
                    "last_promoted_version": challenger_version,
                    "validation_report": report,
                }

                log_pipeline_event(
                    event_type="MODEL_PROMOTED",
                    title=f"Challenger {challenger_version} Promoted to Production",
                    details=f"Drift accuracy: {old_acc}% -> {new_acc}% (+{new_acc - old_acc:.1f}%). Val Accuracy: {meta['val_metrics']['accuracy']:.4f}",
                    metadata=self._latest_comparison,
                )

                duration = round(time.time() - start_t, 2)
                return {
                    "success": True,
                    "status": "PROMOTED",
                    "promoted_version": challenger_version,
                    "previous_version": current_version,
                    "duration_seconds": duration,
                    "comparison": self._latest_comparison,
                }
            else:
                # Validation Rejected
                log_pipeline_event(
                    event_type="VALIDATION_REJECTED",
                    title=f"Challenger {challenger_version} Rejected",
                    details=f"Challenger failed validation gate against champion {current_version}.",
                    metadata=report,
                )
                return {
                    "success": False,
                    "status": "REJECTED",
                    "message": "Challenger model did not meet promotion threshold.",
                    "report": report,
                }

        finally:
            self._is_retraining = False
            self._retrain_lock.release()


# Global retraining service instance
retraining_service = RetrainingService()
