"""
DriftGuard Kolmogorov-Smirnov (KS) Statistical Drift Detection Engine.
Calculates continuous distribution drift per feature using scipy.stats.ks_2samp
comparing the active rolling window against the baseline training distribution.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from backend.config import settings
from backend.db import (
    get_recent_feature_logs,
    record_drift_check,
    log_pipeline_event,
)


class DriftDetector:
    """Evaluates Kolmogorov-Smirnov 2-sample tests against baseline distributions."""

    def __init__(self, baseline_path: str = settings.BASELINE_TRAIN_DATA):
        self.baseline_path = baseline_path
        self._baseline_df: pd.DataFrame = pd.DataFrame()
        self._load_baseline()

    def _load_baseline(self) -> None:
        if os.path.exists(self.baseline_path):
            self._baseline_df = pd.read_csv(self.baseline_path)
            print(f"[DriftDetector] Loaded baseline dataset with {len(self._baseline_df)} rows from {self.baseline_path}")
        else:
            print(f"[DriftDetector] Warning: Baseline dataset not found at {self.baseline_path}")

    def check_drift(
        self,
        window_size: int = settings.WINDOW_SIZE,
        p_threshold: float = settings.P_VALUE_THRESHOLD,
        min_drifted_features: int = settings.MIN_FEATURES_DRIFTED,
    ) -> Dict[str, Any]:
        """
        Retrieves the most recent `window_size` samples from SQLite feature_logs
        and computes the two-sample KS test per feature.
        """
        if self._baseline_df.empty:
            self._load_baseline()
            if self._baseline_df.empty:
                return {
                    "drift_detected": False,
                    "status": "NO_BASELINE",
                    "window_size": 0,
                    "min_required_samples": window_size,
                    "current_samples": 0,
                    "features": {},
                    "drifted_features_count": 0,
                    "p_value_threshold": p_threshold,
                    "min_features_drifted_threshold": min_drifted_features,
                }

        recent_df = get_recent_feature_logs(limit=window_size)
        current_samples = len(recent_df)

        feature_results: Dict[str, Dict[str, Any]] = {}
        drifted_count = 0
        p_values_summary: Dict[str, float] = {}

        for col in settings.FEATURE_COLUMNS:
            if col not in self._baseline_df.columns:
                continue

            baseline_series = self._baseline_df[col].dropna().values

            if current_samples >= 15:  # Minimum statistical sample size for KS test
                current_series = recent_df[col].dropna().values
                stat, p_val = ks_2samp(baseline_series, current_series)
                is_feature_drifted = bool(p_val < p_threshold)
            else:
                stat, p_val = 0.0, 1.0
                is_feature_drifted = False

            if is_feature_drifted:
                drifted_count += 1

            p_val_rounded = float(round(p_val, 6))
            p_values_summary[col] = p_val_rounded

            feature_results[col] = {
                "ks_statistic": float(round(stat, 4)),
                "p_value": p_val_rounded,
                "is_drifted": is_feature_drifted,
                "baseline_mean": float(round(np.mean(baseline_series), 4)),
                "current_mean": float(round(np.mean(recent_df[col].values), 4)) if current_samples > 0 else 0.0,
            }

        # System drift is triggered when >= min_drifted_features have p < threshold
        drift_detected = bool(current_samples >= 20 and drifted_count >= min_drifted_features)

        status_str = "DRIFT_DETECTED" if drift_detected else "HEALTHY"
        if current_samples < 20:
            status_str = "ACCUMULATING_DATA"

        # Build breached features mapping: {feature_name: p_value} for drifted features
        breached_features = {
            col: feature_results[col]["p_value"]
            for col in feature_results
            if feature_results[col]["is_drifted"]
        }

        # Record check in database
        record_drift_check(
            window_size=current_samples,
            drift_detected=drift_detected,
            drifted_features_count=drifted_count,
            p_values=p_values_summary,
            trigger_action="RETRAIN_RECOMMENDED" if drift_detected else "NONE",
        )

        return {
            "status": status_str,
            "drift_detected": drift_detected,
            "window_size": window_size,
            "current_samples": current_samples,
            "drifted_features_count": drifted_count,
            "min_features_drifted_threshold": min_drifted_features,
            "p_value_threshold": p_threshold,
            "features": feature_results,
            "breached_features": breached_features,
        }


# Global detector instance
drift_detector = DriftDetector()
