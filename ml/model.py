"""
DriftGuard Machine Learning Model Module.
Defines classification model architecture, training, and evaluation routines for market manipulation detection.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

FEATURE_COLUMNS = [
    "cancel_ratio",
    "order_to_cancel_latency_ms",
    "order_size_zscore",
    "order_frequency",
    "time_between_orders_ms",
]

TARGET_COLUMN = "is_manipulation"


def create_model_pipeline(model_type: str = "rf", random_state: int = 42) -> Pipeline:
    """Creates a standard scikit-learn pipeline with preprocessing and classifier."""
    scaler = StandardScaler()
    if model_type == "gb":
        classifier = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        )

    pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", classifier),
    ])
    return pipeline


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Calculates standard classification metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, y_prob)
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(auc, 4)),
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    model_type: str = "rf",
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Trains model on provided training dataframe and computes evaluation metrics."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    pipeline = create_model_pipeline(model_type=model_type, random_state=random_state)
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate_model(pipeline, X_train, y_train)
    val_metrics = {}
    if val_df is not None:
        X_val = val_df[FEATURE_COLUMNS]
        y_val = val_df[TARGET_COLUMN]
        val_metrics = evaluate_model(pipeline, X_val, y_val)

    metadata = {
        "model_type": model_type,
        "n_train_samples": len(train_df),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    return pipeline, metadata


def save_model(model: Any, output_path: str) -> None:
    """Serializes model to disk using joblib."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def load_model(model_path: str) -> Any:
    """Deserializes model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)
