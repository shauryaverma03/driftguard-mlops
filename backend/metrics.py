"""
DriftGuard Prometheus Metrics Instrumentation.
Exposes standard Prometheus metrics on /metrics for observability.
"""

from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from backend.drift_detector import drift_detector
from backend.model_manager import model_manager
from backend.retraining_service import retraining_service
from backend.db import get_all_feature_logs_count
from backend.config import settings

# Prometheus Metrics Definitions
REQUESTS_TOTAL = Counter(
    "driftguard_requests_total",
    "Total trade inference requests processed by DriftGuard",
)

RETRAINS_TOTAL = Counter(
    "driftguard_retrains_total",
    "Total automated model retraining pipelines executed",
)

LAST_RETRAIN_TIMESTAMP = Gauge(
    "driftguard_last_retrain_timestamp_seconds",
    "Epoch timestamp of the most recent model retraining run",
)

FEATURE_KS_PVALUE = Gauge(
    "driftguard_feature_ks_pvalue",
    "Current Kolmogorov-Smirnov test p-value per feature against baseline",
    ["feature"],
)

FEATURE_DRIFT_DETECTED = Gauge(
    "driftguard_feature_drift_detected",
    "Flag indicating if individual feature has drifted (1=drift, 0=healthy)",
    ["feature"],
)

DRIFT_SYSTEM_STATUS = Gauge(
    "driftguard_system_drift_detected",
    "Overall system drift status (1=drifted, 0=healthy)",
)

ACTIVE_MODEL_VERSION_NUM = Gauge(
    "driftguard_active_model_version_number",
    "Integer representation of current active production model version",
)


def update_prometheus_metrics():
    """Refreshes gauge values before scraping."""
    # Update drift metrics
    drift_res = drift_detector.check_drift()
    is_drifted = 1 if drift_res.get("drift_detected") else 0
    DRIFT_SYSTEM_STATUS.set(is_drifted)

    features = drift_res.get("features", {})
    for feat, data in features.items():
        FEATURE_KS_PVALUE.labels(feature=feat).set(data.get("p_value", 1.0))
        FEATURE_DRIFT_DETECTED.labels(feature=feat).set(1 if data.get("is_drifted") else 0)

    # Update retraining info
    LAST_RETRAIN_TIMESTAMP.set(retraining_service.last_retrain_time)

    # Active model version
    v_str = model_manager.version.replace("v", "")
    try:
        ACTIVE_MODEL_VERSION_NUM.set(int(v_str))
    except Exception:
        ACTIVE_MODEL_VERSION_NUM.set(1)


def generate_metrics_response() -> bytes:
    """Updates metrics and generates Prometheus text payload."""
    update_prometheus_metrics()
    return generate_latest()
