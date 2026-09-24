"""
DriftGuard Configuration Management.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "DriftGuard MLOps"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    BASELINE_TRAIN_DATA: str = os.path.join(DATA_DIR, "baseline_train.csv")
    BASELINE_VAL_DATA: str = os.path.join(DATA_DIR, "baseline_val.csv")
    DRIFTED_TEST_DATA: str = os.path.join(DATA_DIR, "drifted_test.csv")
    ACTIVE_MODEL_PATH: str = os.path.join(MODEL_DIR, "active_model.joblib")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_EXPERIMENT_NAME: str = "DriftGuard-MarketManipulation"
    MLFLOW_MODEL_NAME: str = "DriftGuard-Classifier"

    # SQLite Database for feature store & logs
    DB_PATH: str = os.getenv("DB_PATH", "driftguard.db")

    # Drift Detection Parameters
    WINDOW_SIZE: int = int(os.getenv("WINDOW_SIZE", "100"))  # Rolling window size for KS test
    P_VALUE_THRESHOLD: float = float(os.getenv("P_VALUE_THRESHOLD", "0.05"))  # Statistical significance
    MIN_FEATURES_DRIFTED: int = int(os.getenv("MIN_FEATURES_DRIFTED", "3"))  # Min features to trigger retrain
    AUTO_RETRAIN_ENABLED: bool = os.getenv("AUTO_RETRAIN_ENABLED", "true").lower() == "true"

    # Features
    FEATURE_COLUMNS: list = [
        "cancel_ratio",
        "order_to_cancel_latency_ms",
        "order_size_zscore",
        "order_frequency",
        "time_between_orders_ms",
    ]


settings = Settings()
