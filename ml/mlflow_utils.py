"""
DriftGuard MLflow Integration Utilities.
Handles experiment tracking, metric logging, model artifact registration, and stage promotion.
"""

import os
import tempfile
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "DriftGuard-MarketManipulation"
MODEL_REGISTRY_NAME = "DriftGuard-Classifier"


def init_mlflow(tracking_uri: str = "sqlite:///mlflow.db") -> MlflowClient:
    """Initializes MLflow tracking URI and ensures target experiment exists."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        try:
            mlflow.create_experiment(EXPERIMENT_NAME)
        except Exception:
            pass
    mlflow.set_experiment(EXPERIMENT_NAME)
    return client


def log_and_register_model(
    model: Any,
    params: Dict[str, Any],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    run_name: str = "baseline_run",
    tags: Optional[Dict[str, str]] = None,
    register: bool = True,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Dict[str, Any]:
    """
    Logs an ML training run with parameters, train/validation metrics, and registers the model in MLflow.
    Returns metadata dict containing run_id, model_uri, and registered model version.
    """
    client = init_mlflow(tracking_uri)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log training metrics
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        # Log validation metrics
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        if tags:
            mlflow.set_tags(tags)

        # Log model artifact using mlflow.sklearn
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME if register else None,
        )

        model_version = "v1"
        if register:
            try:
                # Retrieve latest registered model version
                versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
                if versions:
                    latest_v = max(versions, key=lambda v: int(v.version))
                    model_version = f"v{latest_v.version}"
            except Exception:
                pass

        return {
            "run_id": run_id,
            "model_uri": model_info.model_uri,
            "model_version": model_version,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }


def promote_model_to_production(
    model_version_str: str,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    """Promotes given model version to 'Production' stage and archives previous versions."""
    client = init_mlflow(tracking_uri)
    version_num = model_version_str.replace("v", "")

    try:
        # Transition target version to Production
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=version_num,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        # Fallback if transition API encounters stage restriction
        print(f"[MLflow] Stage transition note: {e}")
