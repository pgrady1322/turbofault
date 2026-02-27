"""
TurboFault v0.1.0

cli.py — Command-line interface for TurboFault.

Commands:
    turbofault download   — Download NASA C-MAPSS dataset
    turbofault train      — Train a model (XGBoost, LSTM, Transformer, CNN1D)
    turbofault evaluate   — Evaluate a trained model on test data
    turbofault tune       — Hyperparameter optimization with Optuna
    turbofault explain    — Feature importance & sensor attribution analysis

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import json
import logging
import sys
from pathlib import Path

import click
import yaml

from turbofault import __version__

logger = logging.getLogger("turbofault")

TABULAR_MODELS = {"xgboost", "random_forest", "ridge"}
DEEP_MODELS = {"lstm", "gru", "transformer", "cnn1d"}
ALL_MODELS = TABULAR_MODELS | DEEP_MODELS


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise click.BadParameter(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


@click.group()
@click.version_option(__version__, prog_name="turbofault")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """TurboFault — Turbofan Engine RUL Prediction & Anomaly Detection."""
    _setup_logging(verbose)


@main.command()
@click.option(
    "-o",
    "--output-dir",
    default="data",
    show_default=True,
    help="Directory to save the C-MAPSS dataset.",
)
def download(output_dir: str) -> None:
    """Download the NASA C-MAPSS turbofan dataset."""
    from turbofault.data.dataset import download_cmapss

    output = Path(output_dir)
    download_cmapss(output)
    click.echo(f"✓ Dataset downloaded to {output}")


@main.command()
@click.option(
    "-m",
    "--model",
    "model_type",
    required=True,
    type=click.Choice(sorted(ALL_MODELS)),
    help="Model type.",
)
@click.option(
    "-s",
    "--subset",
    default="FD001",
    show_default=True,
    type=click.Choice(["FD001", "FD002", "FD003", "FD004"]),
    help="C-MAPSS subset.",
)
@click.option(
    "-d",
    "--data-dir",
    default="data/CMAPSSData",
    show_default=True,
    help="Path to CMAPSSData/ directory.",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    default=None,
    help="YAML config file for model hyperparameters.",
)
@click.option("--max-rul", default=125, show_default=True, help="Piecewise-linear RUL cap.")
@click.option(
    "--window-size", default=30, show_default=True, help="Sliding window size (deep models only)."
)
@click.option(
    "--epochs", default=100, show_default=True, help="Training epochs (deep models only)."
)
@click.option("--batch-size", default=256, show_default=True, help="Batch size (deep models only).")
@click.option(
    "--save-dir",
    default="trained_models",
    show_default=True,
    help="Directory to save trained model.",
)
def train(
    model_type: str,
    subset: str,
    data_dir: str,
    config_path: str | None,
    max_rul: int,
    window_size: int,
    epochs: int,
    batch_size: int,
    save_dir: str,
) -> None:
    """Train a model on C-MAPSS data."""
    from turbofault.data.dataset import load_cmapss
    from turbofault.data.features import build_feature_set, get_feature_columns
    from turbofault.data.preprocessing import prepare_sequence_data, prepare_tabular_data
    from turbofault.training.evaluation import evaluate_rul

    config = _load_config(config_path)

    # Load data
    dataset = load_cmapss(Path(data_dir), subset=subset, max_rul=max_rul)
    click.echo(dataset.summary())

    # Feature engineering
    dataset.train_df = build_feature_set(dataset.train_df)
    dataset.test_df = build_feature_set(dataset.test_df)
    feature_cols = get_feature_columns(dataset.train_df)

    save_path = Path(save_dir) / f"{model_type}_{subset}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type in TABULAR_MODELS:
        data = prepare_tabular_data(dataset, feature_cols)
        model_obj = _build_tabular_model(model_type, config)

        from turbofault.training.trainer import train_tabular

        result = train_tabular(
            model_obj, data["X_train"], data["y_train"], data["X_val"], data["y_val"]
        )

        preds = result["model"].predict(data["X_test"])
        metrics = evaluate_rul(data["y_test"], preds, prefix="test_")

    elif model_type in DEEP_MODELS:
        data = prepare_sequence_data(dataset, feature_cols, window_size=window_size)
        model_obj = _build_deep_model(model_type, data["X_train"].shape[2], config)

        from turbofault.training.trainer import predict_deep, train_deep

        result = train_deep(
            model_obj,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_path,
        )

        preds = predict_deep(result["model"], data["X_test"])
        metrics = evaluate_rul(data["y_test"], preds, prefix="test_")

    # Save results
    results_path = Path("results") / f"{model_type}_{subset}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    click.echo(f"\n✓ Results saved to {results_path}")
    click.echo(
        f"  RMSE: {metrics['test_rmse']:.2f}  |  " f"NASA Score: {metrics['test_nasa_score']:.2f}"
    )


@main.command()
@click.option(
    "-m",
    "--model",
    "model_type",
    required=True,
    type=click.Choice(sorted(ALL_MODELS)),
    help="Model type.",
)
@click.option(
    "-s",
    "--subset",
    default="FD001",
    show_default=True,
    type=click.Choice(["FD001", "FD002", "FD003", "FD004"]),
)
@click.option("-d", "--data-dir", default="data/CMAPSSData", show_default=True)
@click.option("-n", "--n-trials", default=50, show_default=True, help="Number of Optuna trials.")
def tune(
    model_type: str,
    subset: str,
    data_dir: str,
    n_trials: int,
) -> None:
    """Run hyperparameter search with Optuna."""
    from turbofault.data.dataset import load_cmapss
    from turbofault.data.features import build_feature_set, get_feature_columns
    from turbofault.data.preprocessing import prepare_sequence_data, prepare_tabular_data
    from turbofault.training.hp_search import run_hp_search

    dataset = load_cmapss(Path(data_dir), subset=subset)
    dataset.train_df = build_feature_set(dataset.train_df)
    dataset.test_df = build_feature_set(dataset.test_df)
    feature_cols = get_feature_columns(dataset.train_df)

    if model_type in TABULAR_MODELS:
        data = prepare_tabular_data(dataset, feature_cols)
    else:
        data = prepare_sequence_data(dataset, feature_cols)

    result = run_hp_search(
        model_type=model_type,
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        n_trials=n_trials,
    )

    click.echo(f"\n✓ Best RMSE: {result['best_value']:.2f}")
    click.echo(f"  Best params: {json.dumps(result['best_params'], indent=2)}")


@main.command()
@click.option(
    "-m",
    "--model",
    "model_type",
    required=True,
    type=click.Choice(sorted(ALL_MODELS)),
    help="Model type.",
)
@click.option(
    "-s",
    "--subset",
    default="FD001",
    show_default=True,
    type=click.Choice(["FD001", "FD002", "FD003", "FD004"]),
    help="C-MAPSS subset.",
)
@click.option(
    "-d",
    "--data-dir",
    default="data/CMAPSSData",
    show_default=True,
    help="Path to CMAPSSData/ directory.",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    default=None,
    help="YAML config file for model hyperparameters.",
)
@click.option("--max-rul", default=125, show_default=True, help="Piecewise-linear RUL cap.")
@click.option(
    "--window-size", default=30, show_default=True, help="Sliding window size (deep models only)."
)
@click.option("--model-path", default=None, help="Path to saved model checkpoint (deep models).")
@click.option("-o", "--output", "output_path", default=None, help="Save results JSON to this path.")
def evaluate(
    model_type: str,
    subset: str,
    data_dir: str,
    config_path: str | None,
    max_rul: int,
    window_size: int,
    model_path: str | None,
    output_path: str | None,
) -> None:
    """Evaluate a model on the C-MAPSS test set."""
    from turbofault.data.dataset import load_cmapss
    from turbofault.data.features import build_feature_set, get_feature_columns
    from turbofault.data.preprocessing import prepare_sequence_data, prepare_tabular_data
    from turbofault.training.evaluation import evaluate_rul

    config = _load_config(config_path)

    # Load data
    dataset = load_cmapss(Path(data_dir), subset=subset, max_rul=max_rul)
    dataset.train_df = build_feature_set(dataset.train_df)
    dataset.test_df = build_feature_set(dataset.test_df)
    feature_cols = get_feature_columns(dataset.train_df)

    if model_type in TABULAR_MODELS:
        data = prepare_tabular_data(dataset, feature_cols)
        model_obj = _build_tabular_model(model_type, config)
        model_obj.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        preds = model_obj.predict(data["X_test"])
        metrics = evaluate_rul(data["y_test"], preds, prefix="test_")
    elif model_type in DEEP_MODELS:
        data = prepare_sequence_data(dataset, feature_cols, window_size=window_size)
        model_obj = _build_deep_model(model_type, data["X_train"].shape[2], config)

        if model_path:
            import torch

            model_obj.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            click.echo(f"✓ Loaded model from {model_path}")
        else:
            from turbofault.training.trainer import train_deep

            result = train_deep(
                model_obj,
                data["X_train"],
                data["y_train"],
                data["X_val"],
                data["y_val"],
                epochs=100,
            )
            model_obj = result["model"]

        from turbofault.training.trainer import predict_deep

        preds = predict_deep(model_obj, data["X_test"])
        metrics = evaluate_rul(data["y_test"], preds, prefix="test_")
    else:
        raise click.BadParameter(f"Unknown model: {model_type}")

    # Display results
    click.echo(f"\n{'─' * 50}")
    click.echo(f"  Model:      {model_type}")
    click.echo(f"  Subset:     {subset}")
    click.echo(f"  RMSE:       {metrics['test_rmse']:.2f}")
    click.echo(f"  MAE:        {metrics['test_mae']:.2f}")
    click.echo(f"  R²:         {metrics['test_r2']:.4f}")
    click.echo(f"  NASA Score: {metrics['test_nasa_score']:.2f}")
    click.echo(f"{'─' * 50}")

    # Save results
    if output_path is None:
        output_path = f"results/{model_type}_{subset}_eval.json"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"✓ Results saved to {out}")


@main.command()
@click.option(
    "-m",
    "--model",
    "model_type",
    required=True,
    type=click.Choice(sorted(ALL_MODELS)),
    help="Model type.",
)
@click.option(
    "-s",
    "--subset",
    default="FD001",
    show_default=True,
    type=click.Choice(["FD001", "FD002", "FD003", "FD004"]),
    help="C-MAPSS subset.",
)
@click.option(
    "-d",
    "--data-dir",
    default="data/CMAPSSData",
    show_default=True,
    help="Path to CMAPSSData/ directory.",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    default=None,
    help="YAML config file for model hyperparameters.",
)
@click.option("--max-rul", default=125, show_default=True, help="Piecewise-linear RUL cap.")
@click.option(
    "--method",
    default="permutation",
    show_default=True,
    type=click.Choice(["permutation", "shap"]),
    help="Explanation method.",
)
@click.option("--top-n", default=20, show_default=True, help="Number of top features to show.")
@click.option("-o", "--output", "output_path", default=None, help="Save report to this path.")
def explain(
    model_type: str,
    subset: str,
    data_dir: str,
    config_path: str | None,
    max_rul: int,
    method: str,
    top_n: int,
    output_path: str | None,
) -> None:
    """Explain model predictions with feature importance & sensor attribution."""
    from turbofault.data.dataset import load_cmapss
    from turbofault.data.features import build_feature_set, get_feature_columns
    from turbofault.data.preprocessing import prepare_tabular_data
    from turbofault.explain.feature_explainer import (
        generate_explanation_report,
        permutation_importance,
        sensor_contribution_analysis,
        shap_feature_importance,
    )

    if model_type not in TABULAR_MODELS:
        click.echo(
            f"⚠ Explanation currently supports tabular models only "
            f"({', '.join(sorted(TABULAR_MODELS))})"
        )
        sys.exit(1)

    config = _load_config(config_path)

    # Load + prepare data
    dataset = load_cmapss(Path(data_dir), subset=subset, max_rul=max_rul)
    dataset.train_df = build_feature_set(dataset.train_df)
    dataset.test_df = build_feature_set(dataset.test_df)
    feature_cols = get_feature_columns(dataset.train_df)
    data = prepare_tabular_data(dataset, feature_cols)

    # Train model
    model_obj = _build_tabular_model(model_type, config)
    model_obj.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

    # Explain
    if method == "permutation":
        importance_df = permutation_importance(
            model_obj,
            data["X_test"],
            data["y_test"],
            feature_names=data["feature_columns"],
        )
    elif method == "shap":
        shap_result = shap_feature_importance(
            model_obj,
            data["X_test"],
            feature_names=data["feature_columns"],
        )
        importance_df = shap_result["feature_importance"]

    sensor_analysis = sensor_contribution_analysis(importance_df, top_n=top_n)
    report = generate_explanation_report(model_type, importance_df, sensor_analysis, top_n=top_n)
    click.echo(report)

    # Save report
    if output_path is None:
        output_path = f"results/{model_type}_{subset}_explanation.txt"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(report)
    click.echo(f"\n✓ Report saved to {out}")


def _build_tabular_model(model_type: str, config: dict):
    """Instantiate a tabular model from type + config."""
    if model_type == "xgboost":
        from turbofault.models.xgboost_baseline import XGBoostRUL

        return XGBoostRUL(**config.get("model", {}))
    elif model_type == "random_forest":
        from turbofault.models.xgboost_baseline import RandomForestRUL

        return RandomForestRUL(**config.get("model", {}))
    elif model_type == "ridge":
        from turbofault.models.xgboost_baseline import RidgeRUL

        return RidgeRUL(**config.get("model", {}))
    raise ValueError(f"Unknown tabular model: {model_type}")


def _build_deep_model(model_type: str, input_dim: int, config: dict):
    """Instantiate a deep-learning model from type + config."""
    model_config = config.get("model", {})
    model_config["input_dim"] = input_dim

    if model_type == "lstm":
        from turbofault.models.lstm import LSTMModel

        return LSTMModel(**model_config)
    elif model_type == "gru":
        from turbofault.models.lstm import GRUModel

        return GRUModel(**model_config)
    elif model_type == "transformer":
        from turbofault.models.transformer import TransformerRUL

        return TransformerRUL(**model_config)
    elif model_type == "cnn1d":
        from turbofault.models.cnn1d import CNN1DModel

        return CNN1DModel(**model_config)
    raise ValueError(f"Unknown deep model: {model_type}")


# TurboFault v0.1.0
# Any usage is subject to this software's license.
