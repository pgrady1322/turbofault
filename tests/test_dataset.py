"""
TurboFault v0.1.0

test_dataset.py — Tests for C-MAPSS dataset loading and RUL computation.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import numpy as np
import pandas as pd

from turbofault.data.dataset import (
    ALL_COLUMNS,
    LOW_VARIANCE_SENSORS,
    OPERATIONAL_SETTINGS,
    SENSOR_COLUMNS,
    SUBSET_INFO,
    CMAPSSDataset,
)


def _make_engine_df(engine_id: int, n_cycles: int) -> pd.DataFrame:
    """Create a synthetic engine DataFrame for testing."""
    np.random.seed(engine_id)
    data = {
        "engine_id": [engine_id] * n_cycles,
        "cycle": list(range(1, n_cycles + 1)),
    }
    for col in OPERATIONAL_SETTINGS:
        data[col] = np.random.uniform(0, 1, n_cycles)
    for col in SENSOR_COLUMNS:
        data[col] = np.random.uniform(0, 100, n_cycles)
    return pd.DataFrame(data)


def _make_test_dataset(n_engines: int = 5, min_cycles: int = 20, max_cycles: int = 200):
    """Create a synthetic CMAPSSDataset for testing."""
    train_dfs = []
    test_dfs = []
    rul_values = []

    for eid in range(1, n_engines + 1):
        n_train = np.random.randint(min_cycles, max_cycles)
        n_test = np.random.randint(min_cycles // 2, n_train)
        true_rul = np.random.randint(10, 100)

        train_dfs.append(_make_engine_df(eid, n_train))
        test_dfs.append(_make_engine_df(eid, n_test))
        rul_values.append(true_rul)

    return CMAPSSDataset(
        subset="FD001",
        train_df=pd.concat(train_dfs, ignore_index=True),
        test_df=pd.concat(test_dfs, ignore_index=True),
        rul_df=pd.DataFrame({"rul": rul_values}),
        max_rul=125,
    )


class TestColumnDefinitions:
    """Verify sensor/column constants are consistent."""

    def test_sensor_count(self):
        assert len(SENSOR_COLUMNS) == 21

    def test_operational_count(self):
        assert len(OPERATIONAL_SETTINGS) == 3

    def test_all_columns_count(self):
        # engine_id + cycle + 3 op + 21 sensors = 26
        assert len(ALL_COLUMNS) == 26

    def test_low_variance_subset(self):
        for s in LOW_VARIANCE_SENSORS:
            assert s in SENSOR_COLUMNS

    def test_subset_info(self):
        assert set(SUBSET_INFO.keys()) == {"FD001", "FD002", "FD003", "FD004"}


class TestCMAPSSDataset:
    """Test the CMAPSSDataset dataclass."""

    def test_properties(self):
        ds = _make_test_dataset(n_engines=3)
        assert ds.num_train_engines == 3
        assert ds.num_test_engines == 3
        assert ds.num_sensors == 21
        assert len(ds.sensor_columns) == 21
        assert len(ds.operational_columns) == 3

    def test_add_rul_column(self):
        ds = _make_test_dataset(n_engines=3)
        ds.add_rul_column()
        assert "rul" in ds.train_df.columns
        # RUL should be capped at max_rul
        assert ds.train_df["rul"].max() <= ds.max_rul
        # RUL at last cycle of each engine should be 0
        for eid in ds.train_df["engine_id"].unique():
            engine = ds.train_df[ds.train_df["engine_id"] == eid]
            last_rul = engine.loc[engine["cycle"].idxmax(), "rul"]
            assert last_rul == 0

    def test_add_test_rul(self):
        ds = _make_test_dataset(n_engines=3)
        ds.add_rul_column()
        ds.add_test_rul()
        assert "rul" in ds.test_df.columns
        assert ds.test_df["rul"].max() <= ds.max_rul

    def test_piecewise_linear_rul(self):
        """Early cycles should have RUL capped at max_rul."""
        ds = _make_test_dataset(n_engines=1, min_cycles=200, max_cycles=250)
        ds.max_rul = 125
        ds.add_rul_column()

        engine = ds.train_df[ds.train_df["engine_id"] == 1].sort_values("cycle")
        rul_values = engine["rul"].values

        # First value should be capped at 125 (engine has 200+ cycles)
        assert rul_values[0] == 125
        # Last value should be 0
        assert rul_values[-1] == 0
        # Should be monotonically non-increasing
        assert all(rul_values[i] >= rul_values[i + 1] for i in range(len(rul_values) - 1))

    def test_summary(self):
        ds = _make_test_dataset(n_engines=3)
        ds.add_rul_column()
        summary = ds.summary()
        assert "FD001" in summary
        assert "Train engines: 3" in summary
        assert "Sensors:" in summary


class TestSyntheticDataIntegrity:
    """Verify synthetic data helper works correctly."""

    def test_engine_df_shape(self):
        df = _make_engine_df(1, 50)
        assert len(df) == 50
        assert list(df.columns) == ALL_COLUMNS

    def test_unique_engines(self):
        ds = _make_test_dataset(n_engines=5)
        assert ds.train_df["engine_id"].nunique() == 5
        assert ds.test_df["engine_id"].nunique() == 5


# TurboFault v0.1.0
# Any usage is subject to this software's license.
