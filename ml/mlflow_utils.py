"""
DriftGuard MLflow Integration Utilities.
Handles experiment tracking, metric logging, model artifact registration, and stage promotion.
"""

import os
import time
import uuid
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "DriftGuard-MarketManipulation"
MODEL_REGISTRY_NAME = "DriftGuard-Classifier"


def init_mlflow(tracking_uri: str = "sqlite:///mlflow.db") -> Optional[MlflowClient]:
    """Initializes MLflow tracking URI and ensures target experiment exists."""
    try:
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
    except Exception as e:
        print(f"[MLflow] Tracking initialization note: {e}")
        return None


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
    fallback_run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    model_version = "v1"
    model_uri = "models/active_model.joblib"

    try:
        client = init_mlflow(tracking_uri)
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log parameters
            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass

            # Log training metrics
            for k, v in train_metrics.items():
                try:
                    mlflow.log_metric(f"train_{k}", v)
                except Exception:
                    pass

            # Log validation metrics
            for k, v in val_metrics.items():
                try:
                    mlflow.log_metric(f"val_{k}", v)
                except Exception:
                    pass

            if tags:
                try:
                    mlflow.set_tags(tags)
                except Exception:
                    pass

            # Log model artifact with cloudpickle format
            try:
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=MODEL_REGISTRY_NAME if register else None,
                    serialization_format="cloudpickle",
                )
                model_uri = model_info.model_uri
            except Exception as e:
                print(f"[MLflow] sklearn.log_model note: {e}")
                try:
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=MODEL_REGISTRY_NAME if register else None,
                    )
                    model_uri = model_info.model_uri
                except Exception as inner_e:
                    print(f"[MLflow] Fallback log_model note: {inner_e}")

            if register and client:
                try:
                    versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
                    if versions:
                        latest_v = max(versions, key=lambda v: int(v.version))
                        model_version = f"v{latest_v.version}"
                except Exception:
                    pass

            return {
                "run_id": run_id,
                "model_uri": model_uri,
                "model_version": model_version,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

    except Exception as e:
        print(f"[MLflow] Run logging note: {e}")
        return {
            "run_id": fallback_run_id,
            "model_uri": model_uri,
            "model_version": model_version,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }


def promote_model_to_production(
    model_version_str: str,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    """Promotes given model version to 'Production' stage and archives previous versions."""
    try:
        client = init_mlflow(tracking_uri)
        if not client:
            return
        version_num = model_version_str.replace("v", "")
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=version_num,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        print(f"[MLflow] Stage transition note: {e}")
