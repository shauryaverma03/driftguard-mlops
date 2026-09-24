"""
Unit tests for Synthetic Market Data Generation.
"""

import os
import pandas as pd
from data.generator import (
    generate_baseline_dataset,
    generate_drifted_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def test_baseline_dataset_properties():
    df = generate_baseline_dataset(n_samples=500, seed=42)
    assert len(df) == 500
    for col in FEATURE_COLUMNS:
        assert col in df.columns
        assert not df[col].isnull().any()
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_drifted_dataset_properties():
    df_drift = generate_drifted_dataset(n_samples=300, seed=100)
    assert len(df_drift) == 300
    for col in FEATURE_COLUMNS:
        assert col in df_drift.columns
        assert not df_drift[col].isnull().any()
    assert TARGET_COLUMN in df_drift.columns
