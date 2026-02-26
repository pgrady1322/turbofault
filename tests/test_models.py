"""
TurboFault v0.1.0

test_models.py — Tests for model instantiation and forward passes.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import numpy as np
import pytest

from turbofault.models.xgboost_baseline import RandomForestRUL, RidgeRUL, XGBoostRUL


# ── Tabular model tests ────────────────────────────────────────────
class TestXGBoostRUL:
    def test_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.rand(100) * 125

        model = XGBoostRUL(n_estimators=10, max_depth=3)
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape == (100,)
        assert model.feature_importance_ is not None
        assert len(model.feature_importance_) == 10

    def test_with_validation(self):
        np.random.seed(42)
        X_train = np.random.randn(80, 5)
        y_train = np.random.rand(80) * 125
        X_val = np.random.randn(20, 5)
        y_val = np.random.rand(20) * 125

        model = XGBoostRUL(n_estimators=20, max_depth=3)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        assert preds.shape == (20,)

    def test_feature_importance(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.rand(50) * 125
        model = XGBoostRUL(n_estimators=10)
        model.fit(X, y)

        top = model.get_feature_importance(
            feature_names=["a", "b", "c", "d", "e"], top_n=3
        )
        assert len(top) == 3
        assert all(isinstance(t, tuple) for t in top)


class TestRandomForestRUL:
    def test_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.rand(50) * 125

        model = RandomForestRUL(n_estimators=10, max_depth=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)


class TestRidgeRUL:
    def test_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.rand(50) * 125

        model = RidgeRUL(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)


# ── Evaluation metrics tests ──────────────────────────────────────
class TestEvaluation:
    def test_nasa_score_perfect(self):
        from turbofault.training.evaluation import nasa_score

        y = np.array([100.0, 50.0, 25.0])
        assert nasa_score(y, y) == 0.0

    def test_nasa_score_asymmetric(self):
        from turbofault.training.evaluation import nasa_score

        y_true = np.array([50.0])
        # d = pred - true
        # d > 0: overestimates RUL (dangerous "late" failure) → /10 denominator
        # d < 0: underestimates RUL (conservative "early" alert) → /13 denominator
        overestimate_score = nasa_score(y_true, np.array([60.0]))  # d = +10
        underestimate_score = nasa_score(y_true, np.array([40.0]))  # d = -10
        # Overestimating RUL (dangerous) penalized more than underestimating
        assert overestimate_score > underestimate_score

    def test_evaluate_rul(self):
        from turbofault.training.evaluation import evaluate_rul

        y_true = np.array([100.0, 50.0, 25.0, 10.0])
        y_pred = np.array([95.0, 55.0, 20.0, 12.0])
        results = evaluate_rul(y_true, y_pred)

        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results
        assert "nasa_score" in results
        assert results["rmse"] > 0
        assert results["r2"] > 0  # Predictions are decent

    def test_comparison_table(self):
        from turbofault.training.evaluation import print_comparison_table

        results = {
            "XGBoost": {"rmse": 15.2, "mae": 10.1, "r2": 0.85, "nasa_score": 320.5},
            "Ridge": {"rmse": 22.1, "mae": 16.3, "r2": 0.72, "nasa_score": 580.2},
        }
        table = print_comparison_table(results)
        assert "XGBoost" in table
        assert "Ridge" in table


# ── Deep model tests (skipped if torch not installed) ──────────────
torch = pytest.importorskip("torch")


class TestLSTMModel:
    def test_forward_shape(self):
        from turbofault.models.lstm import LSTMModel

        model = LSTMModel(input_dim=14, hidden_dim=32, n_layers=1)
        x = torch.randn(8, 30, 14)  # (batch, seq, features)
        out = model(x)
        assert out.shape == (8, 1)

    def test_bidirectional(self):
        from turbofault.models.lstm import LSTMModel

        model = LSTMModel(input_dim=14, hidden_dim=32, bidirectional=True)
        x = torch.randn(4, 20, 14)
        out = model(x)
        assert out.shape == (4, 1)

    def test_parameter_count(self):
        from turbofault.models.lstm import LSTMModel

        model = LSTMModel(input_dim=14, hidden_dim=32)
        assert model.num_parameters > 0


class TestGRUModel:
    def test_forward_shape(self):
        from turbofault.models.lstm import GRUModel

        model = GRUModel(input_dim=14, hidden_dim=32, n_layers=1)
        x = torch.randn(4, 20, 14)
        out = model(x)
        assert out.shape == (4, 1)


class TestTransformerModel:
    def test_forward_shape(self):
        from turbofault.models.transformer import TransformerRUL

        model = TransformerRUL(
            input_dim=14, d_model=32, n_heads=4, n_layers=2, d_ff=64
        )
        x = torch.randn(4, 30, 14)
        out = model(x)
        assert out.shape == (4, 1)

    def test_with_padding_mask(self):
        from turbofault.models.transformer import TransformerRUL

        model = TransformerRUL(input_dim=14, d_model=32, n_heads=4, n_layers=2)
        x = torch.randn(4, 30, 14)
        mask = torch.zeros(4, 30, dtype=torch.bool)
        mask[:, -5:] = True  # Last 5 positions padded
        out = model(x, src_key_padding_mask=mask)
        assert out.shape == (4, 1)


class TestCNN1DModel:
    def test_forward_shape(self):
        from turbofault.models.cnn1d import CNN1DModel

        model = CNN1DModel(
            input_dim=14, channels=(32, 64), kernel_sizes=(5, 3), pool_sizes=(2, 2)
        )
        x = torch.randn(4, 30, 14)
        out = model(x)
        assert out.shape == (4, 1)

    def test_parameter_count(self):
        from turbofault.models.cnn1d import CNN1DModel

        model = CNN1DModel(input_dim=14)
        assert model.num_parameters > 0

# TurboFault v0.1.0
# Any usage is subject to this software's license.
