"""
TurboFault v0.1.0

xgboost_baseline.py — XGBoost baseline for tabular RUL prediction.

Gradient-boosted trees on engineered features provide a strong baseline
that typically outperforms many deep-learning approaches on small datasets.
This mirrors the tabular-vs-graph baseline pattern from GraphFraud.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

logger = logging.getLogger("turbofault")


class XGBoostRUL:
    """XGBoost regressor for Remaining Useful Life prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            **kwargs,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_importance_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostRUL":
        """Fit the XGBoost model with optional early stopping."""
        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            self.model.set_params(
                early_stopping_rounds=self.early_stopping_rounds
            )
            fit_params["verbose"] = 50

        self.model.fit(X_train, y_train, **fit_params)
        self.feature_importance_ = self.model.feature_importances_
        logger.info(f"✓ XGBoost fitted — best iteration: "
                    f"{self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL values."""
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Save model to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"✓ Model saved to {path}")

    def load(self, path: Path) -> "XGBoostRUL":
        """Load model from JSON file."""
        self.model.load_model(str(path))
        self.feature_importance_ = self.model.feature_importances_
        logger.info(f"✓ Model loaded from {path}")
        return self

    def get_feature_importance(
        self,
        feature_names: Optional[list[str]] = None,
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Return top-N features by importance."""
        if self.feature_importance_ is None:
            raise RuntimeError("Model not yet fitted.")
        names = feature_names or [f"f{i}" for i in range(len(self.feature_importance_))]
        pairs = list(zip(names, self.feature_importance_))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]


class RandomForestRUL:
    """Random Forest baseline for RUL prediction."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestRUL":
        """Fit the Random Forest model."""
        self.model.fit(X_train, y_train)
        logger.info(f"✓ Random Forest fitted — {self.model.n_estimators} trees")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RidgeRUL:
    """Regularized linear regression baseline."""

    def __init__(self, alpha: float = 1.0, **kwargs: Any):
        self.model = Ridge(alpha=alpha, **kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RidgeRUL":
        # Ridge cannot handle NaN (unlike tree-based models); fill with 0
        X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X_clean, y_train)
        logger.info("✓ Ridge regression fitted")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X_clean)

# TurboFault v0.1.0
# Any usage is subject to this software's license.
