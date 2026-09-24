"""
DriftGuard SQLite Database Persistence Layer.
Manages rolling feature logs, drift evaluation records, pipeline audit events, and model registry history.
"""

import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from backend.config import settings


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes SQLite schema tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Feature Logs (rolling window data store)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feature_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        cancel_ratio REAL NOT NULL,
        order_to_cancel_latency_ms REAL NOT NULL,
        order_size_zscore REAL NOT NULL,
        order_frequency REAL NOT NULL,
        time_between_orders_ms REAL NOT NULL,
        prediction INTEGER NOT NULL,
        probability REAL NOT NULL,
        ground_truth INTEGER,
        model_version TEXT NOT NULL,
        is_drifted_batch INTEGER DEFAULT 0
    )
    """)

    # Table 2: Drift Check Records
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS drift_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        window_size INTEGER NOT NULL,
        drift_detected INTEGER NOT NULL,
        drifted_features_count INTEGER NOT NULL,
        p_values_json TEXT NOT NULL,
        trigger_action TEXT
    )
    """)

    # Table 3: Pipeline Audit Event Log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        event_type TEXT NOT NULL,
        title TEXT NOT NULL,
        details TEXT,
        metadata_json TEXT
    )
    """)

    # Table 4: Local Model Registry Cache
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT UNIQUE NOT NULL,
        run_id TEXT NOT NULL,
        model_type TEXT NOT NULL,
        train_accuracy REAL NOT NULL,
        val_accuracy REAL NOT NULL,
        val_f1 REAL NOT NULL,
        stage TEXT NOT NULL,
        created_at REAL NOT NULL,
        artifact_path TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def log_feature_request(
    features: Dict[str, float],
    prediction: int,
    probability: float,
    model_version: str,
    ground_truth: Optional[int] = None,
    is_drifted_batch: int = 0,
) -> int:
    """Logs an incoming trade feature vector and prediction into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = time.time()
    cursor.execute("""
    INSERT INTO feature_logs (
        timestamp, cancel_ratio, order_to_cancel_latency_ms,
        order_size_zscore, order_frequency, time_between_orders_ms,
        prediction, probability, ground_truth, model_version, is_drifted_batch
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now,
        features["cancel_ratio"],
        features["order_to_cancel_latency_ms"],
        features["order_size_zscore"],
        features["order_frequency"],
        features["time_between_orders_ms"],
        prediction,
        probability,
        ground_truth,
        model_version,
        is_drifted_batch,
    ))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_recent_feature_logs(limit: int = 100) -> pd.DataFrame:
    """Fetches the most recent `limit` feature log records as a pandas DataFrame."""
    conn = get_db_connection()
    query = """
    SELECT cancel_ratio, order_to_cancel_latency_ms, order_size_zscore,
           order_frequency, time_between_orders_ms, prediction, probability,
           ground_truth, is_drifted_batch, timestamp
    FROM feature_logs
    ORDER BY id DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    # Reverse so it's in chronological order
    return df.iloc[::-1].reset_index(drop=True)


def get_all_feature_logs_count() -> int:
    """Returns the total number of logged requests."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM feature_logs")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def clear_feature_logs() -> None:
    """Clears the rolling feature store (useful for testing or reset)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM feature_logs")
    conn.commit()
    conn.close()


def log_pipeline_event(
    event_type: str,
    title: str,
    details: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Appends an audit event to the pipeline event timeline."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO pipeline_events (timestamp, event_type, title, details, metadata_json)
    VALUES (?, ?, ?, ?, ?)
    """, (
        time.time(),
        event_type,
        title,
        details,
        json.dumps(metadata or {}),
    ))
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_recent_pipeline_events(limit: int = 30) -> List[Dict[str, Any]]:
    """Retrieves recent pipeline audit events."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, timestamp, event_type, title, details, metadata_json
    FROM pipeline_events
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    events = []
    for r in rows:
        events.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "event_type": r["event_type"],
            "title": r["title"],
            "details": r["details"],
            "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
        })
    conn.close()
    return events


def record_drift_check(
    window_size: int,
    drift_detected: bool,
    drifted_features_count: int,
    p_values: Dict[str, float],
    trigger_action: str = "none",
) -> int:
    """Saves a drift check evaluation result."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO drift_records (
        timestamp, window_size, drift_detected, drifted_features_count,
        p_values_json, trigger_action
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        time.time(),
        window_size,
        1 if drift_detected else 0,
        drifted_features_count,
        json.dumps(p_values),
        trigger_action,
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def register_model_metadata(
    version: str,
    run_id: str,
    model_type: str,
    train_accuracy: float,
    val_accuracy: float,
    val_f1: float,
    stage: str,
    artifact_path: str,
) -> None:
    """Inserts or updates model version metadata in the registry cache."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO model_registry (
        version, run_id, model_type, train_accuracy, val_accuracy,
        val_f1, stage, created_at, artifact_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(version) DO UPDATE SET
        stage=excluded.stage,
        val_accuracy=excluded.val_accuracy,
        val_f1=excluded.val_f1
    """, (
        version,
        run_id,
        model_type,
        train_accuracy,
        val_accuracy,
        val_f1,
        stage,
        time.time(),
        artifact_path,
    ))
    conn.commit()
    conn.close()


def get_all_registered_models() -> List[Dict[str, Any]]:
    """Retrieves all registered models from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT version, run_id, model_type, train_accuracy, val_accuracy,
           val_f1, stage, created_at, artifact_path
    FROM model_registry
    ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    models = [dict(r) for r in rows]
    conn.close()
    return models
