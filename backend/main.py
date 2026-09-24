"""
DriftGuard FastAPI Core Service.
Provides high-throughput prediction serving, real-time feature logging to SQLite,
and endpoints for drift management, metrics, and pipeline event inspection.
"""

import os
import time
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from backend.config import settings
from backend.db import (
    init_db,
    log_feature_request,
    get_all_feature_logs_count,
    get_recent_pipeline_events,
    get_all_registered_models,
)
from backend.model_manager import model_manager
from ml.model import load_model, FEATURE_COLUMNS


# --- Pydantic Schemas ---

class TradeFeatureVector(BaseModel):
    cancel_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of canceled orders to total orders")
    order_to_cancel_latency_ms: float = Field(..., ge=0.0, description="Latency between placement and cancellation in ms")
    order_size_zscore: float = Field(..., description="Order size normalized z-score")
    order_frequency: float = Field(..., ge=0.0, description="Placed orders per minute")
    time_between_orders_ms: float = Field(..., ge=0.0, description="Inter-arrival time between orders in ms")
    ground_truth: Optional[int] = Field(None, description="Optional ground truth label (for simulation/evaluation)")
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
    print("[DriftGuard] Shutting down...")


app = FastAPI(
    title="DriftGuard MLOps API",
    description="Automated MLOps Pipeline for Continuous Model Drift Detection & Retraining in Market Manipulation Detection",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint exposing system uptime and active model info."""
    return {
        "status": "healthy" if model_manager.is_loaded() else "degraded",
        "active_model_version": model_manager.version,
        "model_type": model_manager.model_type,
        "total_requests_processed": get_all_feature_logs_count(),
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

    # Run inference
    pred, prob = model_manager.predict(feature_dict)
    active_version = model_manager.version

    # Log to feature store
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
def get_pipeline_events(limit: int = Query(30, ge=1, le=100)):
    """Returns recent pipeline audit events."""
    return {"events": get_recent_pipeline_events(limit=limit)}
