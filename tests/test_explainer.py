"""
TurboFault v0.1.0

test_explainer.py — Tests for the feature explainability module.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import numpy as np
import pandas as pd
import pytest

from turbofault.explain.feature_explainer import (
    _compute_error,
    generate_explanation_report,
    permutation_importance,
    sensor_contribution_analysis,
)


class TestComputeError:
    def test_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert _compute_error(y_true, y_pred, "rmse") == 0.0

    def test_rmse_nonzero(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        err = _compute_error(y_true, y_pred, "rmse")
        expected = np.sqrt(np.mean([4, 4, 9]))
        assert abs(err - expected) < 1e-6

    def test_mae(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        err = _compute_error(y_true, y_pred, "mae")
        expected = np.mean([2, 2, 3])
        assert abs(err - expected) < 1e-6

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            _compute_error(np.array([1.0]), np.array([1.0]), "mape")


class TestPermutationImportance:
    def test_basic(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 3.0 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1

        class DummyModel:
            def predict(self, X):
                return X[:, 0] * 3.0 + X[:, 1] * 1.5

        model = DummyModel()
        result = permutation_importance(
            model, X, y,
            feature_names=["a", "b", "c", "d", "e"],
            n_repeats=5,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "feature" in result.columns
        assert "importance_mean" in result.columns
        assert "importance_std" in result.columns

        # Feature "a" should be most important (coefficient 3.0)
        top_feature = result.iloc[0]["feature"]
        assert top_feature == "a", f"Expected 'a', got '{top_feature}'"

    def test_default_feature_names(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] * 2.0

        class DummyModel:
            def predict(self, X):
                return X[:, 0] * 2.0

        result = permutation_importance(DummyModel(), X, y, n_repeats=3)
        assert result.iloc[0]["feature"] == "f0"

    def test_mae_metric(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] * 2.0

        class DummyModel:
            def predict(self, X):
                return X[:, 0] * 2.0

        result = permutation_importance(DummyModel(), X, y, metric="mae", n_repeats=3)
        assert len(result) == 3
        assert result.iloc[0]["importance_mean"] > 0


class TestSensorContributionAnalysis:
    def _make_importance_df(self):
        return pd.DataFrame({
            "feature": [
                "sensor_2", "sensor_2_roll_mean_5", "sensor_2_lag_1",
                "sensor_7", "sensor_7_delta_1",
                "sensor_11_ewma_10", "sensor_11",
                "op_setting_1", "cycle_norm",
            ],
            "importance_mean": [0.5, 0.3, 0.2, 0.4, 0.15, 0.35, 0.25, 0.1, 0.05],
        })

    def test_sensor_ranking(self):
        df = self._make_importance_df()
        result = sensor_contribution_analysis(df, top_n=5)

        assert "sensor_ranking" in result
        assert "feature_type_ranking" in result
        assert "top_features" in result

        sensor_rank = result["sensor_ranking"]
        assert "sensor_2" in sensor_rank.index
        # sensor_2 has 3 features summing to 1.0
        assert sensor_rank.loc["sensor_2", "total_importance"] == pytest.approx(1.0)
        assert sensor_rank.loc["sensor_2", "n_features"] == 3

    def test_feature_type_ranking(self):
        df = self._make_importance_df()
        result = sensor_contribution_analysis(df)

        type_rank = result["feature_type_ranking"]
        assert "raw" in type_rank.index
        assert "rolling" in type_rank.index
        assert "lag" in type_rank.index
        assert "delta" in type_rank.index
        assert "ewma" in type_rank.index

    def test_top_features(self):
        df = self._make_importance_df()
        result = sensor_contribution_analysis(df, top_n=3)
        assert len(result["top_features"]) == 3

    def test_shap_column(self):
        df = pd.DataFrame({
            "feature": ["sensor_2", "sensor_7"],
            "mean_abs_shap": [0.5, 0.3],
        })
        result = sensor_contribution_analysis(df)
        assert "sensor_ranking" in result


class TestGenerateReport:
    def test_report_structure(self):
        importance_df = pd.DataFrame({
            "feature": ["sensor_2", "sensor_7", "sensor_11"],
            "importance_mean": [0.5, 0.3, 0.2],
        })
        sensor_analysis = {
            "sensor_ranking": pd.DataFrame({
                "total_importance": [0.5, 0.3, 0.2],
                "avg_importance": [0.5, 0.3, 0.2],
                "n_features": [1, 1, 1],
            }, index=["sensor_2", "sensor_7", "sensor_11"]),
            "feature_type_ranking": pd.DataFrame({
                "total_importance": [1.0],
                "avg_importance": [0.33],
                "n_features": [3],
            }, index=["raw"]),
            "top_features": importance_df.copy(),
        }

        report = generate_explanation_report(
            "XGBoost", importance_df, sensor_analysis, top_n=3
        )

        assert "XGBoost" in report
        assert "sensor_2" in report
        assert "Top Sensors" in report
        assert "Feature Type Breakdown" in report

# TurboFault v0.1.0
# Any usage is subject to this software's license.
