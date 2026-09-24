"""
DriftGuard FastAPI Core Service.
Provides high-throughput prediction serving, real-time feature logging to SQLite,
Kolmogorov-Smirnov statistical drift detection, automated retraining, traffic simulation,
and Prometheus observability.
"""

import os
import time
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from backend.config import settings
from backend.db import (
    init_db,
    log_feature_request,
    get_all_feature_logs_count,
    get_recent_pipeline_events,
    get_all_registered_models,
    clear_feature_logs,
    log_pipeline_event,
)
from backend.model_manager import model_manager
from backend.drift_detector import drift_detector
from backend.retraining_service import retraining_service
from backend.simulator import simulator
from backend.metrics import (
    generate_metrics_response,
    REQUESTS_TOTAL,
    RETRAINS_TOTAL,
    CONTENT_TYPE_LATEST,
)
from ml.model import load_model, FEATURE_COLUMNS


# --- Pydantic Schemas ---

class TradeFeatureVector(BaseModel):
    cancel_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of canceled orders to total orders")
    order_to_cancel_latency_ms: float = Field(..., ge=0.0, description="Latency between placement and cancellation in ms")
    order_size_zscore: float = Field(..., description="Order size normalized z-score")
    order_frequency: float = Field(..., ge=0.0, description="Placed orders per minute")
    time_between_orders_ms: float = Field(..., ge=0.0, description="Inter-arrival time between orders in ms")
    ground_truth: Optional[int] = Field(None, description="Optional ground truth label (0: Normal, 1: Manipulation)")
    is_drifted_batch: Optional[int] = Field(0, description="Flag indicating if trade is part of synthetic drift batch")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cancel_ratio": 0.88,
                "order_to_cancel_latency_ms": 12.5,
                "order_size_zscore": 4.2,
                "order_frequency": 140.0,
                "time_between_orders_ms": 35.0,
                "ground_truth": 1,
            }
        }
    }


class BatchTradeRequest(BaseModel):
    trades: List[TradeFeatureVector]


class PredictionResponse(BaseModel):
    request_id: int
    prediction: int
    is_manipulation: bool
    label: str
    probability: float
    model_version: str
    processed_at: float


class HealthResponse(BaseModel):
    status: str
    active_model_version: str
    model_type: str
    total_requests_processed: int
    total_retrains_triggered: int
    is_streaming: bool
    uptime_seconds: float


START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    print("[DriftGuard] Initializing database & model serving...")
    init_db()

    # Load active model from disk if available
    if os.path.exists(settings.ACTIVE_MODEL_PATH):
        try:
            model_manager.load_from_file(
                model_path=settings.ACTIVE_MODEL_PATH,
                version="v1",
            )
            print(f"[DriftGuard] Loaded active model v1 from {settings.ACTIVE_MODEL_PATH}")
        except Exception as e:
            print(f"[DriftGuard] Error loading active model: {e}")
    else:
        print("[DriftGuard] Warning: Active model artifact not found. Please run baseline training first.")

    yield
    # Stop any background simulators on shutdown
    simulator.stop_stream()
    print("[DriftGuard] Shutting down...")


app = FastAPI(
    title="DriftGuard MLOps API",
    description="Automated MLOps Pipeline for Continuous Model Drift Detection & Retraining in Market Manipulation Detection",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard (any origin, local dev ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 1. Serving & Health Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint exposing system uptime, active model, and counts."""
    return {
        "status": "healthy" if model_manager.is_loaded() else "degraded",
        "active_model_version": model_manager.version,
        "model_type": model_manager.model_type,
        "total_requests_processed": get_all_feature_logs_count(),
        "total_retrains_triggered": retraining_service.retrain_count,
        "is_streaming": simulator.is_streaming,
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_trade(trade: TradeFeatureVector):
    """
    Serves market manipulation inference on a single trade feature vector.
    Logs request features into the SQLite rolling window store.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model serving engine is not loaded.")

    feature_dict = {
        "cancel_ratio": trade.cancel_ratio,
        "order_to_cancel_latency_ms": trade.order_to_cancel_latency_ms,
        "order_size_zscore": trade.order_size_zscore,
        "order_frequency": trade.order_frequency,
        "time_between_orders_ms": trade.time_between_orders_ms,
    }

    pred, prob = model_manager.predict(feature_dict)
    active_version = model_manager.version
    REQUESTS_TOTAL.inc()

    row_id = log_feature_request(
        features=feature_dict,
        prediction=pred,
        probability=prob,
        model_version=active_version,
        ground_truth=trade.ground_truth,
        is_drifted_batch=trade.is_drifted_batch or 0,
    )

    return {
        "request_id": row_id,
        "prediction": pred,
        "is_manipulation": bool(pred == 1),
        "label": "MANIPULATION" if pred == 1 else "NORMAL",
        "probability": round(prob, 4),
        "model_version": active_version,
        "processed_at": time.time(),
    }


@app.post("/predict/batch")
def predict_batch(batch: BatchTradeRequest):
    """Batch inference endpoint."""
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model serving engine is not loaded.")

    results = []
    for trade in batch.trades:
        res = predict_trade(trade)
        results.append(res)
    return {"total": len(results), "predictions": results}


# --- 2. Drift Detection Endpoints ---

@app.get("/drift/status")
def get_drift_status(
    window_size: int = Query(settings.WINDOW_SIZE, ge=10, le=1000),
    p_threshold: float = Query(settings.P_VALUE_THRESHOLD, ge=0.001, le=0.5),
):
    """
    Returns live Kolmogorov-Smirnov drift detection state and per-feature p-values.
    """
    drift_info = drift_detector.check_drift(
        window_size=window_size,
        p_threshold=p_threshold,
    )
    return drift_info


@app.post("/drift/check")
def trigger_manual_drift_check():
    """Forces an immediate evaluation of rolling window drift."""
    return drift_detector.check_drift()


# --- 3. Traffic Simulator Endpoints ---

@app.post("/simulate/stream")
def start_traffic_stream(rate: float = Query(5.0, ge=0.5, le=50.0)):
    """Starts continuous background normal synthetic trade stream."""
    return simulator.start_stream(rate_per_sec=rate)


@app.post("/simulate/drift")
def inject_drift_batch(batch_size: int = Query(50, ge=10, le=500)):
    """Injects a burst of drifted market trades to trigger distribution shift."""
    return simulator.inject_drift_batch(batch_size=batch_size)


@app.post("/simulate/stop")
def stop_traffic_stream():
    """Stops the active background traffic stream."""
    return simulator.stop_stream()


# --- 4. Retraining & Reset Endpoints ---

@app.post("/retrain/trigger")
def trigger_manual_retrain():
    """
    Manually triggers the end-to-end retraining pipeline:
    Aggregates data -> Trains Challenger -> Validates -> Logs to MLflow -> Hot-Swaps on approval.
    """
    RETRAINS_TOTAL.inc()
    result = retraining_service.trigger_retrain(reason="MANUAL_UI_TRIGGER")
    return result


@app.post("/pipeline/reset")
def reset_pipeline():
    """
    Resets feature store rolling window, stops stream, and reloads baseline model v1.
    """
    simulator.stop_stream()
    clear_feature_logs()

    # Reload baseline model
    if os.path.exists(settings.ACTIVE_MODEL_PATH):
        model_manager.load_from_file(settings.ACTIVE_MODEL_PATH, version="v1")

    log_pipeline_event(
        event_type="PIPELINE_RESET",
        title="Pipeline Reset Executed",
        details="Rolling feature store cleared, active model reset to baseline v1.",
    )

    return {
        "status": "RESET_SUCCESSFUL",
        "active_model_version": model_manager.version,
        "total_requests": 0,
        "message": "Pipeline and feature logs have been reset to initial baseline state.",
    }


# --- 5. Model Registry & History Endpoints ---

@app.get("/models/active")
def get_active_model_info():
    """Returns metadata for the currently active production model."""
    return {
        "version": model_manager.version,
        "model_type": model_manager.model_type,
        "metrics": model_manager.metrics,
        "loaded_at": model_manager.loaded_at,
        "is_active": model_manager.is_loaded(),
    }


@app.get("/models")
def get_all_models():
    """Returns model versions in registry cache."""
    return {"models": get_all_registered_models()}


@app.get("/events")
@app.get("/pipeline/log")
def get_pipeline_events(limit: int = Query(30, ge=1, le=100)):
    """Returns recent pipeline audit events (newest first)."""
    return {"events": get_recent_pipeline_events(limit=limit)}


@app.get("/drift/comparison")
def get_before_after_comparison():
    """Returns the latest before vs after accuracy comparison report on drifted test data."""
    return retraining_service.latest_comparison


# --- 6. Prometheus Metrics Endpoint ---

@app.get("/metrics")
def get_prometheus_metrics():
    """Standard Prometheus scraping endpoint."""
    metrics_data = generate_metrics_response()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


# --- 7. Mount Frontend Static Files ---
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

