"""
TurboFault v0.1.0

hp_search.py — Hyperparameter optimization with Optuna.

Supports tuning for all model types:
  - XGBoost / Random Forest (tabular)
  - LSTM / GRU (recurrent)
  - Transformer (attention-based)
  - CNN1D (convolutional)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger("turbofault")


def _objective_xgboost(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for XGBoost hyperparameters."""
    from turbofault.models.xgboost_baseline import XGBoostRUL
    from turbofault.training.evaluation import evaluate_rul

    model = XGBoostRUL(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    metrics = evaluate_rul(y_val, preds)
    return metrics["rmse"]


def _objective_deep(
    trial: optuna.Trial,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for deep learning models."""

    from turbofault.training.evaluation import evaluate_rul
    from turbofault.training.trainer import predict_deep, train_deep

    input_dim = X_train.shape[2]

    if model_type == "lstm":
        from turbofault.models.lstm import LSTMModel

        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            n_layers=trial.suggest_int("n_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            bidirectional=trial.suggest_categorical("bidirectional", [True, False]),
        )
    elif model_type == "gru":
        from turbofault.models.lstm import GRUModel

        model = GRUModel(
            input_dim=input_dim,
            hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            n_layers=trial.suggest_int("n_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
        )
    elif model_type == "transformer":
        from turbofault.models.transformer import TransformerRUL

        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
        model = TransformerRUL(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=trial.suggest_int("n_layers", 2, 6),
            d_ff=trial.suggest_categorical("d_ff", [128, 256, 512]),
            dropout=trial.suggest_float("dropout", 0.1, 0.4),
        )
    elif model_type == "cnn1d":
        from turbofault.models.cnn1d import CNN1DModel

        n_blocks = trial.suggest_int("n_blocks", 2, 4)
        base_ch = trial.suggest_categorical("base_channels", [32, 64, 128])
        channels = tuple(base_ch * (2**i) for i in range(n_blocks))
        kernel_sizes = tuple(
            trial.suggest_categorical(f"kernel_{i}", [3, 5, 7]) for i in range(n_blocks)
        )
        model = CNN1DModel(
            input_dim=input_dim,
            channels=channels,
            kernel_sizes=kernel_sizes,
            pool_sizes=tuple(2 for _ in range(n_blocks)),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    result = train_deep(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        batch_size=batch_size,
        learning_rate=lr,
        patience=10,
    )

    preds = predict_deep(result["model"], X_val)
    metrics = evaluate_rul(y_val, preds)
    return metrics["rmse"]


def run_hp_search(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    study_name: str | None = None,
    direction: str = "minimize",
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter search.

    Args:
        model_type: One of 'xgboost', 'random_forest', 'lstm', 'gru',
                    'transformer', 'cnn1d'.
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        n_trials: Number of Optuna trials.
        study_name: Optional study name.
        direction: 'minimize' for RMSE.

    Returns:
        Dict with 'best_params', 'best_value', 'study'.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for HP search. " "Install with: pip install turbofault[optuna]"
        ) from None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = study_name or f"turbofault-{model_type}"
    study = optuna.create_study(study_name=study_name, direction=direction)

    if model_type == "xgboost":
        study.optimize(
            lambda trial: _objective_xgboost(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
        )
    elif model_type in ("lstm", "gru", "transformer", "cnn1d"):
        study.optimize(
            lambda trial: _objective_deep(trial, model_type, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
        )
    else:
        raise ValueError(f"HP search not implemented for: {model_type}")

    logger.info(f"✓ HP search complete — best RMSE: {study.best_value:.2f}")
    logger.info(f"  Best params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


# TurboFault v0.1.0
# Any usage is subject to this software's license.
