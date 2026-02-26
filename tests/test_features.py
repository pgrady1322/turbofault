"""
TurboFault v0.1.0

test_features.py — Tests for feature engineering functions.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import numpy as np
import pandas as pd
import pytest

from turbofault.data.dataset import OPERATIONAL_SETTINGS, SENSOR_COLUMNS
from turbofault.data.features import (
    add_cycle_features,
    add_delta_features,
    add_ewma_features,
    add_lag_features,
    add_rolling_features,
    build_feature_set,
    get_feature_columns,
)


def _make_simple_df(n_engines: int = 2, n_cycles: int = 50) -> pd.DataFrame:
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


class TestRollingFeatures:
    def test_adds_columns(self):
        df = _make_simple_df()
        sensors = SENSOR_COLUMNS[:3]
        result = add_rolling_features(df, sensors=sensors, windows=(5,), statistics=("mean",))
        for s in sensors:
            assert f"{s}_roll5_mean" in result.columns

    def test_correct_count(self):
        df = _make_simple_df()
        sensors = SENSOR_COLUMNS[:2]
        result = add_rolling_features(
            df, sensors=sensors, windows=(5, 10), statistics=("mean", "std")
        )
        new_cols = [c for c in result.columns if "_roll" in c]
        # 2 sensors × 2 windows × 2 stats = 8
        assert len(new_cols) == 8

    def test_no_nans_with_min_periods(self):
        df = _make_simple_df(n_engines=1, n_cycles=10)
        result = add_rolling_features(
            df, sensors=SENSOR_COLUMNS[:1], windows=(5,), statistics=("mean",)
        )
        assert not result[f"{SENSOR_COLUMNS[0]}_roll5_mean"].isna().any()


class TestLagFeatures:
    def test_adds_columns(self):
        df = _make_simple_df()
        result = add_lag_features(df, sensors=SENSOR_COLUMNS[:2], lags=(1, 3))
        assert f"{SENSOR_COLUMNS[0]}_lag1" in result.columns
        assert f"{SENSOR_COLUMNS[0]}_lag3" in result.columns

    def test_values_shifted(self):
        df = _make_simple_df(n_engines=1, n_cycles=10)
        result = add_lag_features(df, sensors=SENSOR_COLUMNS[:1], lags=(1,))
        sensor = SENSOR_COLUMNS[0]
        # Value at cycle 5 should equal original value at cycle 4
        orig_val = df.loc[df["cycle"] == 4, sensor].values[0]
        lag_val = result.loc[result["cycle"] == 5, f"{sensor}_lag1"].values[0]
        assert np.isclose(orig_val, lag_val)


class TestDeltaFeatures:
    def test_adds_columns(self):
        df = _make_simple_df()
        result = add_delta_features(df, sensors=SENSOR_COLUMNS[:2], periods=(1,))
        assert f"{SENSOR_COLUMNS[0]}_delta1" in result.columns

    def test_no_nans(self):
        df = _make_simple_df()
        result = add_delta_features(df, sensors=SENSOR_COLUMNS[:1], periods=(1,))
        assert not result[f"{SENSOR_COLUMNS[0]}_delta1"].isna().any()


class TestEWMAFeatures:
    def test_adds_columns(self):
        df = _make_simple_df()
        result = add_ewma_features(df, sensors=SENSOR_COLUMNS[:1], spans=(10,))
        assert f"{SENSOR_COLUMNS[0]}_ewma10" in result.columns


class TestCycleFeatures:
    def test_adds_columns(self):
        df = _make_simple_df()
        result = add_cycle_features(df)
        assert "cycle_norm" in result.columns
        assert "cycle_squared" in result.columns

    def test_normalization_range(self):
        df = _make_simple_df(n_engines=1, n_cycles=50)
        result = add_cycle_features(df)
        assert result["cycle_norm"].min() > 0
        assert np.isclose(result["cycle_norm"].max(), 1.0)


class TestBuildFeatureSet:
    def test_builds_all_features(self):
        df = _make_simple_df(n_engines=1, n_cycles=30)
        result = build_feature_set(
            df,
            sensors=SENSOR_COLUMNS[:2],
            rolling_windows=(5,),
            rolling_stats=("mean",),
            lags=(1,),
            delta_periods=(1,),
            ewma_spans=(10,),
        )
        assert len(result.columns) > len(df.columns)
        assert "cycle_norm" in result.columns

    def test_get_feature_columns(self):
        df = _make_simple_df()
        result = build_feature_set(df, sensors=SENSOR_COLUMNS[:2])
        feature_cols = get_feature_columns(result)
        assert "engine_id" not in feature_cols
        assert "cycle" not in feature_cols
        assert "rul" not in feature_cols

# TurboFault v0.1.0
# Any usage is subject to this software's license.
