"""
DriftGuard Synthetic Data Generator for Market Manipulation Detection.

Generates realistic trade feature distributions for both baseline (normal/standard
market regimes) and drifted (regime shift / evolved manipulation tactics) data.

Features:
- cancel_ratio: Ratio of canceled orders to total orders [0.0 - 1.0]
- order_to_cancel_latency_ms: Milliseconds between order placement and cancellation [1.0 - 5000.0]
- order_size_zscore: Standardized order volume relative to historical average [-3.0 - 10.0]
- order_frequency: Placed orders per minute [1.0 - 250.0]
- time_between_orders_ms: Inter-arrival time between consecutive orders [2.0 - 10000.0]
- is_manipulation: Binary label (0: Normal trade flow, 1: Manipulation pattern)
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "cancel_ratio",
    "order_to_cancel_latency_ms",
    "order_size_zscore",
    "order_frequency",
    "time_between_orders_ms",
]

TARGET_COLUMN = "is_manipulation"


def generate_baseline_dataset(
    n_samples: int = 2500,
    manipulation_rate: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates baseline training dataset with standard market conditions.
    Normal trading exhibits low cancellation, moderate order frequency, and longer latencies.
    Manipulation (spoofing/layering/wash trading) exhibits high cancellation, ultra-short latencies,
    and abnormal order size spikes.
    """
    rng = np.random.default_rng(seed)
    n_manip = int(n_samples * manipulation_rate)
    n_norm = n_samples - n_manip

    # --- Normal Trade Features ---
    # Low cancel ratio, moderate latency, standard order sizes, relaxed frequency
    norm_cancel_ratio = np.clip(rng.beta(a=2, b=8, size=n_norm), 0.01, 0.60)
    norm_latency = np.clip(rng.lognormal(mean=6.5, sigma=0.8, size=n_norm), 50.0, 5000.0)
    norm_size_zscore = rng.normal(loc=0.0, scale=1.0, size=n_norm)
    norm_frequency = np.clip(rng.gamma(shape=3.0, scale=4.0, size=n_norm), 1.0, 50.0)
    norm_time_between = np.clip(rng.exponential(scale=1200.0, size=n_norm), 50.0, 8000.0)
    norm_labels = np.zeros(n_norm, dtype=int)

    # --- Manipulation Trade Features (Baseline Tactics) ---
    # High cancel ratio (spoofing/layering), fast cancel latency, large sizes, high frequency
    manip_cancel_ratio = np.clip(rng.beta(a=8, b=2, size=n_manip), 0.65, 0.99)
    manip_latency = np.clip(rng.lognormal(mean=3.2, sigma=0.5, size=n_manip), 5.0, 150.0)
    manip_size_zscore = np.clip(rng.normal(loc=3.5, scale=1.2, size=n_manip), 0.5, 9.0)
    manip_frequency = np.clip(rng.gamma(shape=8.0, scale=12.0, size=n_manip), 40.0, 200.0)
    manip_time_between = np.clip(rng.exponential(scale=120.0, size=n_manip), 5.0, 600.0)
    manip_labels = np.ones(n_manip, dtype=int)

    # Combine and shuffle
    df_norm = pd.DataFrame({
        "cancel_ratio": norm_cancel_ratio,
        "order_to_cancel_latency_ms": norm_latency,
        "order_size_zscore": norm_size_zscore,
        "order_frequency": norm_frequency,
        "time_between_orders_ms": norm_time_between,
        TARGET_COLUMN: norm_labels,
    })

    df_manip = pd.DataFrame({
        "cancel_ratio": manip_cancel_ratio,
        "order_to_cancel_latency_ms": manip_latency,
        "order_size_zscore": manip_size_zscore,
        "order_frequency": manip_frequency,
        "time_between_orders_ms": manip_time_between,
        TARGET_COLUMN: manip_labels,
    })

    df = pd.concat([df_norm, df_manip], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def generate_drifted_dataset(
    n_samples: int = 500,
    manipulation_rate: float = 0.30,
    seed: int = 100,
) -> pd.DataFrame:
    """
    Generates drifted trade dataset reflecting market regime shifts & stealthier manipulation.
    - Distribution Shift 1: Latency distribution shifts left (sub-millisecond / FPGA routing).
    - Distribution Shift 2: Order size z-scores shift down (smaller sliced orders to evade naive volume thresholds).
    - Distribution Shift 3: Order frequency and cancel ratio shift upwards due to algorithmic liquidity churn.
    - Distribution Shift 4: Inter-arrival time shrinks significantly.
    """
    rng = np.random.default_rng(seed)
    n_manip = int(n_samples * manipulation_rate)
    n_norm = n_samples - n_manip

    # Drifted Normal: Market overall has faster execution & slightly higher frequency
    drift_norm_cancel = np.clip(rng.beta(a=3, b=5, size=n_norm), 0.15, 0.75) # Shifted up
    drift_norm_latency = np.clip(rng.lognormal(mean=5.2, sigma=0.7, size=n_norm), 15.0, 2000.0) # Shifted down
    drift_norm_size = rng.normal(loc=0.5, scale=1.4, size=n_norm) # Slightly wider spread
    drift_norm_frequency = np.clip(rng.gamma(shape=5.0, scale=6.0, size=n_norm), 10.0, 90.0) # Shifted up
    drift_norm_time_between = np.clip(rng.exponential(scale=600.0, size=n_norm), 20.0, 4000.0) # Shifted down
    norm_labels = np.zeros(n_norm, dtype=int)

    # Drifted Manipulation (Stealth / Evolved Tactics):
    # Micro-cancellations, smaller fragmented sizes (zscore 1.5-3.5 instead of 3.5-9.0), ultra-rapid bursts
    drift_manip_cancel = np.clip(rng.beta(a=9, b=1.5, size=n_manip), 0.78, 0.99)
    drift_manip_latency = np.clip(rng.lognormal(mean=2.1, sigma=0.4, size=n_manip), 1.0, 45.0) # Ultra-low latency
    drift_manip_size = np.clip(rng.normal(loc=1.8, scale=0.8, size=n_manip), 0.2, 4.5) # Evades old size detector
    drift_manip_frequency = np.clip(rng.gamma(shape=12.0, scale=14.0, size=n_manip), 70.0, 280.0) # Rapid burst
    drift_manip_time_between = np.clip(rng.exponential(scale=45.0, size=n_manip), 1.0, 180.0)
    manip_labels = np.ones(n_manip, dtype=int)

    df_norm = pd.DataFrame({
        "cancel_ratio": drift_norm_cancel,
        "order_to_cancel_latency_ms": drift_norm_latency,
        "order_size_zscore": drift_norm_size,
        "order_frequency": drift_norm_frequency,
        "time_between_orders_ms": drift_norm_time_between,
        TARGET_COLUMN: norm_labels,
    })

    df_manip = pd.DataFrame({
        "cancel_ratio": drift_manip_cancel,
        "order_to_cancel_latency_ms": drift_manip_latency,
        "order_size_zscore": drift_manip_size,
        "order_frequency": drift_manip_frequency,
        "time_between_orders_ms": drift_manip_time_between,
        TARGET_COLUMN: manip_labels,
    })

    df = pd.concat([df_norm, df_manip], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def generate_and_save_all(data_dir: str = "data") -> Tuple[str, str, str]:
    """Generates and saves baseline training, baseline validation, and drifted datasets."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Baseline train (2500 samples)
    df_train = generate_baseline_dataset(n_samples=2500, seed=42)
    train_path = os.path.join(data_dir, "baseline_train.csv")
    df_train.to_csv(train_path, index=False)

    # Baseline hold-out validation (600 samples)
    df_val = generate_baseline_dataset(n_samples=600, seed=999)
    val_path = os.path.join(data_dir, "baseline_val.csv")
    df_val.to_csv(val_path, index=False)

    # Drifted evaluation batch (600 samples)
    df_drift = generate_drifted_dataset(n_samples=600, seed=2026)
    drift_path = os.path.join(data_dir, "drifted_test.csv")
    df_drift.to_csv(drift_path, index=False)

    return train_path, val_path, drift_path


if __name__ == "__main__":
    t_path, v_path, d_path = generate_and_save_all()
    print(f"Generated datasets:\n - Train: {t_path}\n - Val: {v_path}\n - Drifted: {d_path}")
