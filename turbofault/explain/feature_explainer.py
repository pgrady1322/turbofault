"""
TurboFault v0.1.0

feature_explainer.py — Feature-based explainability for RUL predictions.

Answers: "Which sensors and features drive the RUL prediction?"
  - SHAP values for XGBoost / tree-based models
  - Permutation importance for any model
  - Sensor contribution ranking
  - Temporal attention weights for Transformer models

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("turbofault")


# ── Permutation Importance (model-agnostic) ─────────────────────────

def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list[str]] = None,
    n_repeats: int = 10,
    metric: str = "rmse",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance for any model with a predict() method.

    Shuffles each feature column independently and measures the increase
    in prediction error. Features that cause the largest error increase
    when shuffled are the most important.

    Args:
        model: Fitted model with predict(X) method.
        X: Feature matrix (n_samples, n_features).
        y: True target values.
        feature_names: Optional feature names.
        n_repeats: Number of shuffles per feature.
        metric: Error metric ('rmse' or 'mae').
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std,
        sorted by importance descending.
    """
    rng = np.random.RandomState(random_state)
    n_features = X.shape[1]
    feature_names = feature_names or [f"f{i}" for i in range(n_features)]

    # Baseline score
    baseline_preds = model.predict(X)
    baseline_error = _compute_error(y, baseline_preds, metric)

    importances = np.zeros((n_features, n_repeats))

    for feat_idx in range(n_features):
        for rep in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
            perm_preds = model.predict(X_perm)
            perm_error = _compute_error(y, perm_preds, metric)
            importances[feat_idx, rep] = perm_error - baseline_error

    results = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importances.mean(axis=1),
        "importance_std": importances.std(axis=1),
    })
    results = results.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    logger.info(f"✓ Permutation importance computed ({n_repeats} repeats, {metric})")
    return results


def _compute_error(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute error metric."""
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    elif metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    raise ValueError(f"Unknown metric: {metric}")


# ── SHAP Explainability (tree models) ───────────────────────────────

def shap_feature_importance(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[list[str]] = None,
    max_samples: int = 500,
) -> dict[str, Any]:
    """
    Compute SHAP values for a tree-based model (XGBoost, Random Forest).

    SHAP (SHapley Additive exPlanations) provides theoretically grounded
    feature attributions based on cooperative game theory.

    Args:
        model: Fitted XGBoostRUL or RandomForestRUL model.
        X: Feature matrix for explanation.
        feature_names: Optional feature names.
        max_samples: Maximum background samples for SHAP explainer.

    Returns:
        Dict with 'shap_values', 'feature_importance', 'expected_value'.
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP is required for tree-based explanations. "
            "Install with: pip install shap"
        )

    feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

    # Subsample for efficiency
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Get the underlying sklearn/xgb model
    inner_model = getattr(model, "model", model)
    explainer = shap.TreeExplainer(inner_model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info(f"✓ SHAP values computed for {X_sample.shape[0]} samples")

    return {
        "shap_values": shap_values,
        "feature_importance": importance_df,
        "expected_value": float(explainer.expected_value),
        "X_sample": X_sample,
        "feature_names": feature_names,
    }


# ── Sensor Contribution Analysis ────────────────────────────────────

def sensor_contribution_analysis(
    importance_df: pd.DataFrame,
    top_n: int = 20,
) -> dict[str, Any]:
    """
    Group feature importances by sensor to identify which physical
    sensors are most important for RUL prediction.

    Aggregates importance across all derived features (rolling, lag,
    delta, EWMA) for each base sensor.

    Args:
        importance_df: DataFrame with 'feature' and an importance column
                       (either 'importance_mean' or 'mean_abs_shap').
        top_n: Number of top individual features to return.

    Returns:
        Dict with 'sensor_ranking', 'feature_type_ranking',
        'top_features'.
    """
    # Determine which importance column to use
    imp_col = "mean_abs_shap" if "mean_abs_shap" in importance_df.columns else "importance_mean"

    # Extract base sensor from feature name
    def _extract_sensor(name: str) -> str:
        for prefix in ["sensor_", "op_setting_"]:
            if name.startswith(prefix):
                # Handle derived features: sensor_2_roll_mean_5 → sensor_2
                parts = name.split("_")
                if prefix == "sensor_":
                    return f"sensor_{parts[1]}"
                return f"op_setting_{parts[2]}"
        if name.startswith("cycle"):
            return "cycle"
        return name

    # Extract feature type
    def _extract_type(name: str) -> str:
        if "_roll_" in name:
            return "rolling"
        elif "_lag_" in name:
            return "lag"
        elif "_delta_" in name:
            return "delta"
        elif "_ewma_" in name:
            return "ewma"
        elif "cycle" in name:
            return "cycle"
        elif "op_setting" in name:
            return "operational"
        return "raw"

    df = importance_df.copy()
    df["sensor"] = df["feature"].apply(_extract_sensor)
    df["feature_type"] = df["feature"].apply(_extract_type)

    # Aggregate by sensor
    sensor_ranking = (
        df.groupby("sensor")[imp_col]
        .agg(["sum", "mean", "count"])
        .sort_values("sum", ascending=False)
        .rename(columns={"sum": "total_importance", "mean": "avg_importance",
                         "count": "n_features"})
    )

    # Aggregate by feature type
    type_ranking = (
        df.groupby("feature_type")[imp_col]
        .agg(["sum", "mean", "count"])
        .sort_values("sum", ascending=False)
        .rename(columns={"sum": "total_importance", "mean": "avg_importance",
                         "count": "n_features"})
    )

    # Top individual features
    top_features = df.nlargest(top_n, imp_col)[["feature", "sensor", "feature_type", imp_col]]

    logger.info(f"✓ Sensor contribution analysis — "
                f"top sensor: {sensor_ranking.index[0]}")

    return {
        "sensor_ranking": sensor_ranking,
        "feature_type_ranking": type_ranking,
        "top_features": top_features.reset_index(drop=True),
    }


# ── Attention Weight Extraction (Transformer) ──────────────────────

def extract_attention_weights(
    model: "torch.nn.Module",
    X: np.ndarray,
    sample_idx: int = 0,
) -> dict[str, Any]:
    """
    Extract attention weights from a TransformerRUL model.

    Hooks into the multi-head attention layers to capture which
    time steps the model attends to for RUL prediction.

    Args:
        model: Trained TransformerRUL model.
        X: Input data (n_samples, seq_len, n_features).
        sample_idx: Which sample to extract attention for.

    Returns:
        Dict with 'attention_weights' (list of layer attention matrices),
        'prediction', and 'input_sequence'.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    # Register hooks to capture attention weights
    attention_weights = []

    def _hook_fn(module, input, output):
        # TransformerEncoderLayer attention output
        if isinstance(output, tuple) and len(output) >= 2:
            attention_weights.append(output[1].detach().cpu().numpy())

    # Hook into each encoder layer's self-attention
    hooks = []
    for layer in model.transformer_encoder.layers:
        hook = layer.self_attn.register_forward_hook(_hook_fn)
        hooks.append(hook)

    # Forward pass with attention capture
    x_tensor = torch.tensor(X[sample_idx:sample_idx + 1], dtype=torch.float32).to(device)

    # Enable attention output
    for layer in model.transformer_encoder.layers:
        layer.self_attn.need_weights = True

    with torch.no_grad():
        prediction = model(x_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Reset attention output flag
    for layer in model.transformer_encoder.layers:
        layer.self_attn.need_weights = False

    result = {
        "attention_weights": attention_weights,
        "prediction": float(prediction.cpu().item()),
        "input_sequence": X[sample_idx],
        "n_layers": len(attention_weights),
    }

    if attention_weights:
        logger.info(f"✓ Extracted attention from {len(attention_weights)} layers "
                    f"(shape per layer: {attention_weights[0].shape})")
    else:
        logger.warning("No attention weights captured — model may not expose them")

    return result


# ── Report Generation ───────────────────────────────────────────────

def generate_explanation_report(
    model_type: str,
    importance_df: pd.DataFrame,
    sensor_analysis: dict[str, Any],
    top_n: int = 15,
) -> str:
    """
    Generate a human-readable text report of feature explanations.

    Args:
        model_type: Name of the model (e.g., 'XGBoost').
        importance_df: Feature importance DataFrame.
        sensor_analysis: Output from sensor_contribution_analysis().
        top_n: Number of top items to show.

    Returns:
        Formatted report string.
    """
    lines = [
        "═" * 60,
        f"FEATURE EXPLANATION REPORT — {model_type}",
        "═" * 60,
        "",
        "Top Sensors (by total importance):",
        "─" * 40,
    ]

    sensor_rank = sensor_analysis["sensor_ranking"]
    for i, (sensor, row) in enumerate(sensor_rank.head(top_n).iterrows()):
        bar = "█" * int(row["total_importance"] / sensor_rank["total_importance"].max() * 20)
        lines.append(f"  {i + 1:2d}. {sensor:15s}  {row['total_importance']:.4f}  {bar}")

    lines.extend([
        "",
        "Feature Type Breakdown:",
        "─" * 40,
    ])

    type_rank = sensor_analysis["feature_type_ranking"]
    for ftype, row in type_rank.iterrows():
        pct = row["total_importance"] / type_rank["total_importance"].sum() * 100
        lines.append(f"  {ftype:12s}  {pct:5.1f}%  ({int(row['n_features'])} features)")

    lines.extend([
        "",
        f"Top {top_n} Individual Features:",
        "─" * 40,
    ])

    imp_col = "mean_abs_shap" if "mean_abs_shap" in importance_df.columns else "importance_mean"
    for i, row in importance_df.head(top_n).iterrows():
        lines.append(f"  {i + 1:2d}. {row['feature']:35s}  {row[imp_col]:.6f}")

    lines.extend(["", "═" * 60])

    report = "\n".join(lines)
    logger.info(f"\n{report}")
    return report

# TurboFault v0.1.0
# Any usage is subject to this software's license.
