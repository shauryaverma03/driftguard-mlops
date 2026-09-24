"""
DriftGuard Thread-Safe Hot-Swappable Model Manager.
Enables serving inference and zero-downtime hot-swapping of production models.
"""

import threading
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from ml.model import load_model, FEATURE_COLUMNS
from backend.db import log_pipeline_event


class ModelManager:
    """Thread-safe hot-swappable model container."""

    def __init__(self):
        self._lock = threading.RLock()
        self._model = None
        self._version = "uninitialized"
        self._model_type = "rf"
        self._metrics: Dict[str, float] = {}
        self._loaded_at: float = 0.0

    @property
    def version(self) -> str:
        with self._lock:
            return self._version

    @property
    def model_type(self) -> str:
        with self._lock:
            return self._model_type

    @property
    def metrics(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._metrics)

    @property
    def loaded_at(self) -> float:
        with self._lock:
            return self._loaded_at

    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def load_from_file(self, model_path: str, version: str = "v1", metrics: Optional[Dict[str, float]] = None) -> None:
        """Loads a model from disk into the active serving slot."""
        model = load_model(model_path)
        with self._lock:
            self._model = model
            self._version = version
            self._loaded_at = time.time()
            if metrics:
                self._metrics = metrics
            print(f"[ModelManager] Loaded active model {version} from {model_path}")

    def hot_swap(
        self,
        new_model: Any,
        new_version: str,
        new_metrics: Dict[str, float],
        model_type: str = "rf",
    ) -> None:
        """
        Atomically hot-swaps the active serving model reference with zero downtime.
        """
        with self._lock:
            old_version = self._version
            self._model = new_model
            self._version = new_version
            self._model_type = model_type
            self._metrics = new_metrics
            self._loaded_at = time.time()

        print(f"[ModelManager] 🔥 Hot-swapped model: {old_version} -> {new_version}")
        log_pipeline_event(
            event_type="HOT_SWAP_COMPLETED",
            title=f"Model Hot-Swapped to {new_version}",
            details=f"Replaced {old_version} with {new_version}. Validation accuracy: {new_metrics.get('val_accuracy', 'N/A')}",
            metadata={
                "old_version": old_version,
                "new_version": new_version,
                "metrics": new_metrics,
            }
        )

    def predict(self, feature_vector: Dict[str, float]) -> Tuple[int, float]:
        """
        Runs inference on a single feature dictionary.
        Returns (prediction, manipulation_probability).
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model is currently loaded in ModelManager.")
            
            # Create a 1-row DataFrame preserving feature order
            df = pd.DataFrame([[feature_vector[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            
            pred = int(self._model.predict(df)[0])
            prob = float(self._model.predict_proba(df)[0][1]) if hasattr(self._model, "predict_proba") else float(pred)
            return pred, prob

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Runs batch inference on a DataFrame."""
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model is currently loaded in ModelManager.")
            X = df[FEATURE_COLUMNS]
            preds = self._model.predict(X)
            probs = self._model.predict_proba(X)[:, 1] if hasattr(self._model, "predict_proba") else preds
            return preds, probs


# Global singleton instance
model_manager = ModelManager()
