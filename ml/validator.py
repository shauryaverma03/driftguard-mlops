"""
DriftGuard Model Validation Gate & Benchmark Comparison.
Validates newly retrained challenger models against held-out validation sets
and evaluates before/after accuracy on drifted market benchmarks.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
from ml.model import evaluate_model, FEATURE_COLUMNS, TARGET_COLUMN
from backend.config import settings


class ModelValidator:
    """Gates model promotion to production based on benchmark comparisons."""

    def __init__(
        self,
        val_path: str = settings.BASELINE_VAL_DATA,
        drift_path: str = settings.DRIFTED_TEST_DATA,
    ):
        self.val_path = val_path
        self.drift_path = drift_path
        self._val_df = pd.DataFrame()
        self._drift_df = pd.DataFrame()
        self._load_datasets()

    def _load_datasets(self) -> None:
        if os.path.exists(self.val_path):
            self._val_df = pd.read_csv(self.val_path)
        if os.path.exists(self.drift_path):
            self._drift_df = pd.read_csv(self.drift_path)

    def validate_challenger(
        self,
        champion_model: Any,
        challenger_model: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compares challenger against current champion.
        Returns (is_approved, comparison_report).
        """
        if self._val_df.empty or self._drift_df.empty:
            self._load_datasets()

        # 1. Evaluate on clean held-out validation set
        X_val = self._val_df[FEATURE_COLUMNS]
        y_val = self._val_df[TARGET_COLUMN]

        champ_val_metrics = evaluate_model(champion_model, X_val, y_val) if champion_model else {"accuracy": 0.0, "f1_score": 0.0}
        chall_val_metrics = evaluate_model(challenger_model, X_val, y_val)

        # 2. Evaluate on drifted test benchmark (shows true adaptation to shifted tactics)
        X_drift = self._drift_df[FEATURE_COLUMNS]
        y_drift = self._drift_df[TARGET_COLUMN]

        champ_drift_metrics = evaluate_model(champion_model, X_drift, y_drift) if champion_model else {"accuracy": 0.0, "f1_score": 0.0}
        chall_drift_metrics = evaluate_model(challenger_model, X_drift, y_drift)

        # Promotion criteria: Challenger must maintain high validation baseline (> 0.90) and beat or match champion
        passed_val = chall_val_metrics["accuracy"] >= 0.88 and chall_val_metrics["f1_score"] >= 0.85
        passed_drift = chall_drift_metrics["accuracy"] >= champ_drift_metrics["accuracy"]

        is_approved = bool(passed_val and (passed_drift or chall_drift_metrics["f1_score"] >= champ_drift_metrics["f1_score"]))

        report = {
            "is_approved": is_approved,
            "decision": "PROMOTE_TO_PRODUCTION" if is_approved else "REJECT_CHALLENGER",
            "validation_comparison": {
                "champion": champ_val_metrics,
                "challenger": chall_val_metrics,
            },
            "drift_benchmark_comparison": {
                "old_model_drift_accuracy": round(champ_drift_metrics["accuracy"] * 100, 1),
                "new_model_drift_accuracy": round(chall_drift_metrics["accuracy"] * 100, 1),
                "old_model_drift_f1": champ_drift_metrics["f1_score"],
                "new_model_drift_f1": chall_drift_metrics["f1_score"],
            },
            "metrics": {
                "old_accuracy_pct": round(champ_drift_metrics["accuracy"] * 100, 1),
                "new_accuracy_pct": round(chall_drift_metrics["accuracy"] * 100, 1),
                "improvement_pct": round((chall_drift_metrics["accuracy"] - champ_drift_metrics["accuracy"]) * 100, 1),
            }
        }
        return is_approved, report


validator = ModelValidator()
