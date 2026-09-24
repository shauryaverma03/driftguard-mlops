"""
DriftGuard Baseline Model Training & Initial Setup Script.
Generates synthetic market datasets, trains baseline classifier, logs to MLflow,
registers model v1 as Production, and prepares serving assets.
"""

import os
import pandas as pd
from data.generator import generate_and_save_all, FEATURE_COLUMNS, TARGET_COLUMN
from ml.model import train_model, save_model, evaluate_model
from ml.mlflow_utils import log_and_register_model, promote_model_to_production, init_mlflow
from backend.config import settings
from backend.db import init_db, register_model_metadata, log_pipeline_event


def run_baseline_pipeline(data_dir: str = "data", model_dir: str = "models") -> None:
    """Executes full baseline setup pipeline."""
    print("=" * 65)
    print("  🛡️  DriftGuard: Initializing Baseline Model & Datasets")
    print("=" * 65)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    init_db()

    # Step 1: Generate Datasets
    print("\n[1/4] Generating synthetic market trade datasets...")
    train_path, val_path, drift_path = generate_and_save_all(data_dir=data_dir)
    print(f"      ✓ Baseline Train: {train_path}")
    print(f"      ✓ Baseline Val:   {val_path}")
    print(f"      ✓ Drifted Test:   {drift_path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Step 2: Train Baseline Model
    print("\n[2/4] Training baseline RandomForestClassifier...")
    params = {
        "model_type": "rf",
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 4,
        "random_state": 42,
    }
    pipeline, meta = train_model(df_train, df_val, model_type="rf", random_state=42)
    train_metrics = meta["train_metrics"]
    val_metrics = meta["val_metrics"]

    print(f"      ✓ Train Accuracy: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_score']:.4f}")
    print(f"      ✓ Val Accuracy:   {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f}")

    # Step 3: Log to MLflow & Model Registry
    print("\n[3/4] Logging experiment and registering model in MLflow...")
    mlflow_meta = log_and_register_model(
        model=pipeline,
        params=params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        run_name="baseline_v1_random_forest",
        tags={"pipeline": "DriftGuard", "stage": "Production", "version": "v1"},
        register=True,
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    )
    version = mlflow_meta.get("model_version", "v1")
    run_id = mlflow_meta.get("run_id", "local_run_001")
    promote_model_to_production(version, tracking_uri=settings.MLFLOW_TRACKING_URI)
    print(f"      ✓ Registered in MLflow as {version} (Production) [Run ID: {run_id[:8]}...]")

    # Step 4: Persist Serving Artifact & Local DB Cache
    print("\n[4/4] Saving active serving artifact...")
    save_path = settings.ACTIVE_MODEL_PATH
    save_model(pipeline, save_path)
    print(f"      ✓ Saved active model to: {save_path}")

    # Record in SQLite registry cache
    register_model_metadata(
        version=version,
        run_id=run_id,
        model_type="RandomForest",
        train_accuracy=train_metrics["accuracy"],
        val_accuracy=val_metrics["accuracy"],
        val_f1=val_metrics["f1_score"],
        stage="Production",
        artifact_path=save_path,
    )

    log_pipeline_event(
        event_type="BASELINE_INITIALIZED",
        title=f"Baseline Model {version} Deployed",
        details=f"RandomForest model trained on {len(df_train)} baseline samples. Val Accuracy: {val_metrics['accuracy']:.4f}",
        metadata={"version": version, "metrics": val_metrics, "run_id": run_id}
    )

    print("\n" + "=" * 65)
    print(f"  ✅ Baseline Setup Complete! Model {version} is ready for serving.")
    print("=" * 65)


if __name__ == "__main__":
    run_baseline_pipeline()
