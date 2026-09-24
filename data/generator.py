"""
DriftGuard Synthetic Data Generator for Market Manipulation Detection.

Generates realistic trade feature distributions for both baseline (normal/standard
market regimes) and drifted (regime shift / evolved manipulation tactics) data.

Revision (v3): Class distributions are deliberately overlapping with close means on
key features, targeting ~88-95% validation accuracy. The drifted batch shifts means
further to ensure KS-test detection is reliable even though class boundaries are soft.

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
    Generates baseline training dataset with deliberate class overlap.

    Class means are intentionally close on several features, and both classes
    use wide sigma values so their tails cross significantly. This targets
    a realistic validation accuracy of ~88-95% (not 100%).

    The critical design tradeoff:
    - Baseline model accuracy should be sub-96% so the validation gate is meaningful
    - Drift must still be detectable by KS test after inject_drift_batch()
    """
    rng = np.random.default_rng(seed)
    n_manip = int(n_samples * manipulation_rate)
    n_norm = n_samples - n_manip

    # ── Normal Trade Features ──────────────────────────────────────────────────
    # Cancel ratio: Normal peaks at ~0.25 with wide spread up to 0.65
    norm_cancel_ratio = np.clip(rng.beta(a=2.0, b=5.0, size=n_norm), 0.01, 0.90)
    # Latency: log-normal, fairly wide spread (300 - 5000ms typical)
    norm_latency = np.clip(rng.lognormal(mean=7.0, sigma=1.2, size=n_norm), 10.0, 8000.0)
    # Size z-score: centered near 0 but wide sigma so tails reach +3
    norm_size_zscore = rng.normal(loc=0.0, scale=1.5, size=n_norm)
    # Frequency: moderate, up to 50/min but tail to 80
    norm_frequency = np.clip(rng.gamma(shape=2.0, scale=8.0, size=n_norm), 1.0, 80.0)
    # Inter-arrival: exponential, mostly 300-5000ms
    norm_time_between = np.clip(rng.exponential(scale=1200.0, size=n_norm), 30.0, 9000.0)

    # ── Manipulation Trade Features — CLOSER to normal than before ─────────────
    # cancel_ratio: manipulation peaks at ~0.60 (not 0.77), so there is real overlap
    # with normal samples that go up to 0.65+
    manip_cancel_ratio = np.clip(rng.beta(a=4.0, b=2.5, size=n_manip), 0.20, 0.99)
    # Latency: manipulation is faster (10-500ms) but with wide enough spread to overlap
    # normal's lower tail (normal can go as low as 10ms too)
    manip_latency = np.clip(rng.lognormal(mean=4.5, sigma=1.0, size=n_manip), 5.0, 2000.0)
    # Size z-score: manipulation is higher (mean ~2.0) but wide sigma overlaps normal tail
    manip_size_zscore = np.clip(rng.normal(loc=2.0, scale=1.8, size=n_manip), -1.0, 8.0)
    # Frequency: manipulation is higher (mean ~50) but low-manipulation actors at 25+
    # overlap with high-normal actors
    manip_frequency = np.clip(rng.gamma(shape=5.0, scale=10.0, size=n_manip), 8.0, 180.0)
    # Inter-arrival: manipulation faster, mean ~300ms, but tail up to 2000ms
    manip_time_between = np.clip(rng.exponential(scale=300.0, size=n_manip), 5.0, 2000.0)

    norm_labels  = np.zeros(n_norm, dtype=int)
    manip_labels = np.ones(n_manip, dtype=int)

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

    KEY DESIGN: Distributions shift significantly (large KS statistics, p << 0.05) vs
    baseline so drift is reliably detected. The challenger trained on this distribution
    will genuinely outperform the baseline champion on drift-set accuracy.

    Shift summary vs baseline:
    - cancel_ratio: normal pop mean shifts +0.12 (market algo churn), manip stays high
    - order_to_cancel_latency_ms: whole market compresses left by ~50%
    - order_size_zscore: both classes shift upward (larger average orders in new regime)
    - order_frequency: entire market rate increases by ~2x
    - time_between_orders_ms: compresses (faster market tick)
    """
    rng = np.random.default_rng(seed)
    n_manip = int(n_samples * manipulation_rate)
    n_norm = n_samples - n_manip

    # ── Drifted Normal: market-wide execution regime shift ─────────────────────
    # Cancel ratio drifts UP: market-wide algo churn increases
    drift_norm_cancel = np.clip(rng.beta(a=3.5, b=4.0, size=n_norm), 0.08, 0.95)
    # Latency compresses: faster market infra (lognormal shifts left)
    drift_norm_latency = np.clip(rng.lognormal(mean=5.5, sigma=1.1, size=n_norm), 5.0, 3000.0)
    # Size shifts slightly up (larger order sizes in new regime)
    drift_norm_size = rng.normal(loc=0.8, scale=1.5, size=n_norm)
    # Frequency increases (HFT regime)
    drift_norm_frequency = np.clip(rng.gamma(shape=3.5, scale=10.0, size=n_norm), 5.0, 120.0)
    # Inter-arrival compresses
    drift_norm_time_between = np.clip(rng.exponential(scale=550.0, size=n_norm), 15.0, 5000.0)
    norm_labels = np.zeros(n_norm, dtype=int)

    # ── Drifted Manipulation: stealthier micro-cancel tactics ──────────────────
    # Micro-cancellations — cancel ratio stays high but more moderate
    drift_manip_cancel = np.clip(rng.beta(a=5.0, b=1.8, size=n_manip), 0.40, 0.99)
    # Ultra-fast latency (FPGA/co-location)
    drift_manip_latency = np.clip(rng.lognormal(mean=2.5, sigma=0.6, size=n_manip), 2.0, 80.0)
    # Fragmented sizes (smaller slice orders to evade naive volume detectors)
    drift_manip_size = np.clip(rng.normal(loc=1.5, scale=1.2, size=n_manip), -0.5, 6.0)
    # Rapid burst
    drift_manip_frequency = np.clip(rng.gamma(shape=9.0, scale=13.0, size=n_manip), 40.0, 260.0)
    # Very short inter-arrival
    drift_manip_time_between = np.clip(rng.exponential(scale=60.0, size=n_manip), 2.0, 250.0)
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

    # Baseline hold-out validation (600 samples, different seed)
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
