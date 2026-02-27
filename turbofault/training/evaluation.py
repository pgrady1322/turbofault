"""
TurboFault v0.1.0

evaluation.py — Evaluation metrics and reporting for RUL prediction.

Implements prognostics-specific metrics:
  - RMSE, MAE, R^2 (standard regression metrics)
  - NASA Scoring Function (asymmetric: penalizes late predictions more)
  - Per-engine evaluation

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("turbofault")


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA Prognostics Scoring Function (asymmetric penalty).

    Overestimating RUL (predicting the engine has more life than it does)
    is more dangerous than underestimating, and is penalized more heavily.

        d = y_pred - y_true

        For d < 0 (conservative — underestimates RUL):  s = exp(-d / 13) - 1
        For d >= 0 (dangerous — overestimates RUL):      s = exp(d / 10)  - 1

    Lower is better. Zero means perfect prediction.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Total NASA score (sum, not mean).
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def evaluate_rul(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute standard + NASA metrics for RUL predictions.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        prefix: Optional prefix for metric keys (e.g., 'test_').

    Returns:
        Dict with RMSE, MAE, R2, and NASA score.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    score = nasa_score(y_true, y_pred)

    results = {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}nasa_score": score,
    }

    summary = (
        f"{'—' * 40}\n"
        f"RMSE:        {rmse:.2f}\n"
        f"MAE:         {mae:.2f}\n"
        f"R²:          {r2:.4f}\n"
        f"NASA Score:  {score:.2f}\n"
        f"{'—' * 40}"
    )
    print(summary)
    logger.info(summary)

    return results


def evaluate_per_engine(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: np.ndarray,
) -> dict[int, dict[str, float]]:
    """
    Compute metrics per engine (for test-set analysis).

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        engine_ids: Engine ID for each prediction.

    Returns:
        Dict mapping engine_id → metrics dict.
    """
    results = {}
    for engine_id in np.unique(engine_ids):
        mask = engine_ids == engine_id
        if mask.sum() < 2:
            continue
        results[int(engine_id)] = evaluate_rul(y_true[mask], y_pred[mask], prefix="")
    return results


def print_comparison_table(
    results: dict[str, dict[str, float]],
    metrics: tuple[str, ...] = ("rmse", "mae", "r2", "nasa_score"),
) -> str:
    """
    Format a comparison table of multiple models.

    Args:
        results: Dict mapping model_name → metrics dict.
        metrics: Which metrics to show.

    Returns:
        Formatted table string.
    """
    # Header
    header = f"{'Model':<20}"
    for m in metrics:
        header += f" {m.upper():>12}"
    lines = [header, "─" * len(header)]

    # Rows
    for model_name, model_metrics in results.items():
        row = f"{model_name:<20}"
        for m in metrics:
            val = model_metrics.get(m, model_metrics.get(f"test_{m}", float("nan")))
            if m == "r2":
                row += f" {val:>12.4f}"
            else:
                row += f" {val:>12.2f}"
        lines.append(row)

    table = "\n".join(lines)
    print(table)
    logger.info(f"\n{table}")
    return table


# TurboFault v0.1.0
# Any usage is subject to this software's license.
