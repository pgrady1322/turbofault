"""
TurboFault v0.1.0

test_preprocessing.py — Tests for data preprocessing and windowing.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import numpy as np
import pandas as pd
import pytest

from turbofault.data.dataset import OPERATIONAL_SETTINGS, SENSOR_COLUMNS
from turbofault.data.preprocessing import (
    create_sequences,
    drop_low_variance_sensors,
    get_last_cycle_per_engine,
    normalize_sensors,
    temporal_train_val_split,
)


def _make_df(n_engines: int = 5, n_cycles: int = 50) -> pd.DataFrame:
    """Create a simple test DataFrame."""
    np.random.seed(42)
    rows = []
    for eid in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            row = {"engine_id": eid, "cycle": cycle}
            for col in OPERATIONAL_SETTINGS:
                row[col] = np.random.uniform(0, 1)
            for col in SENSOR_COLUMNS:
                row[col] = np.random.uniform(0, 100)
            row["rul"] = max(0, n_cycles - cycle)
            rows.append(row)
    return pd.DataFrame(rows)


class TestNormalization:
    def test_minmax_range(self):
        df = _make_df()
        norm_df, _, scaler = normalize_sensors(df, method="minmax")
        for col in SENSOR_COLUMNS:
            assert norm_df[col].min() >= -0.01  # Allow small float error
            assert norm_df[col].max() <= 1.01

    def test_standard_stats(self):
        df = _make_df(n_engines=10, n_cycles=100)
        norm_df, _, scaler = normalize_sensors(df, method="standard")
        for col in SENSOR_COLUMNS[:3]:
            assert abs(norm_df[col].mean()) < 0.1
            assert abs(norm_df[col].std() - 1.0) < 0.1

    def test_test_transform(self):
        train_df = _make_df(n_engines=3)
        test_df = _make_df(n_engines=2)
        _, test_norm, scaler = normalize_sensors(train_df, test_df, method="minmax")
        assert test_norm is not None
        assert len(test_norm) == len(test_df)

    def test_invalid_method(self):
        df = _make_df()
        with pytest.raises(ValueError, match="Unknown method"):
            normalize_sensors(df, method="invalid")


class TestDropLowVariance:
    def test_drops_sensors(self):
        df = _make_df()
        result = drop_low_variance_sensors(df, sensors=["sensor_1", "sensor_5"])
        assert "sensor_1" not in result.columns
        assert "sensor_5" not in result.columns
        assert "sensor_2" in result.columns


class TestCreateSequences:
    def test_output_shapes(self):
        df = _make_df(n_engines=2, n_cycles=50)
        features = SENSOR_COLUMNS[:3]
        X, y, ids = create_sequences(df, features, window_size=10)
        assert X.ndim == 3
        assert X.shape[1] == 10  # window_size
        assert X.shape[2] == 3  # n_features
        assert len(y) == len(X)
        assert len(ids) == len(X)

    def test_stride(self):
        df = _make_df(n_engines=1, n_cycles=30)
        features = SENSOR_COLUMNS[:2]
        X1, _, _ = create_sequences(df, features, window_size=10, stride=1)
        X5, _, _ = create_sequences(df, features, window_size=10, stride=5)
        assert len(X5) < len(X1)

    def test_short_engine_padded(self):
        df = _make_df(n_engines=1, n_cycles=5)
        features = SENSOR_COLUMNS[:2]
        X, y, ids = create_sequences(df, features, window_size=10)
        # Engine has 5 cycles, window=10, so padded to 10 → 1 window
        assert X.shape[0] >= 1
        assert X.shape[1] == 10


class TestTemporalSplit:
    def test_by_engine(self):
        df = _make_df(n_engines=10)
        train, val = temporal_train_val_split(df, val_fraction=0.2, by_engine=True)
        train_engines = set(train["engine_id"].unique())
        val_engines = set(val["engine_id"].unique())
        assert len(train_engines & val_engines) == 0  # No overlap

    def test_by_cycle(self):
        df = _make_df(n_engines=3, n_cycles=50)
        train, val = temporal_train_val_split(df, val_fraction=0.2, by_engine=False)
        # For each engine, val cycles should come AFTER train cycles
        for eid in df["engine_id"].unique():
            train_max = train[train["engine_id"] == eid]["cycle"].max()
            val_min = val[val["engine_id"] == eid]["cycle"].min()
            assert val_min > train_max

    def test_val_fraction(self):
        df = _make_df(n_engines=10)
        train, val = temporal_train_val_split(df, val_fraction=0.3, by_engine=True)
        # Roughly 30% of engines in validation
        total_engines = df["engine_id"].nunique()
        val_engines = val["engine_id"].nunique()
        assert val_engines == max(1, int(total_engines * 0.3))


class TestGetLastCycle:
    def test_one_row_per_engine(self):
        df = _make_df(n_engines=5, n_cycles=50)
        last = get_last_cycle_per_engine(df)
        assert len(last) == 5
        assert all(last["cycle"] == 50)


# TurboFault v0.1.0
# Any usage is subject to this software's license.
