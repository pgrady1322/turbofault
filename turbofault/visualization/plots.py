"""
TurboFault v0.1.0

plots.py — Visualization utilities for RUL prediction and sensor analysis.

Provides publication-ready matplotlib figures for:
  - Sensor degradation traces
  - RUL prediction vs. ground truth
  - Feature importance rankings
  - Training loss curves
  - Model comparison charts

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("turbofault")

# ── Style defaults ──────────────────────────────────────────────────
STYLE = {
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def plot_sensor_traces(
    df: pd.DataFrame,
    engine_ids: list[int],
    sensors: list[str],
    figsize: tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot sensor readings over cycle for selected engines.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        engine_ids: Which engines to plot.
        sensors: Which sensors to plot.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    _apply_style()
    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=figsize, sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for idx, sensor in enumerate(sensors):
        ax = axes[idx]
        for eid in engine_ids:
            engine = df[df["engine_id"] == eid].sort_values("cycle")
            ax.plot(engine["cycle"], engine[sensor], label=f"Engine {eid}", alpha=0.7)
        ax.set_ylabel(sensor)
        if idx == 0:
            ax.legend(fontsize=8, ncol=min(len(engine_ids), 5))

    axes[-1].set_xlabel("Cycle")
    fig.suptitle("Sensor Degradation Traces", fontsize=16)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Saved sensor traces → {save_path}")

    return fig


def plot_rul_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: Optional[np.ndarray] = None,
    title: str = "RUL Prediction vs. Ground Truth",
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot of predicted vs. true RUL with diagonal reference line.

    Also includes a residual histogram subplot.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        engine_ids: Optional engine IDs for color coding.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save.

    Returns:
        Matplotlib Figure.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Scatter: Predicted vs True
    ax1.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolors="none")
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("True RUL")
    ax1.set_ylabel("Predicted RUL")
    ax1.set_title(title)
    ax1.legend()

    # Residual histogram
    residuals = y_pred - y_true
    ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Prediction Error (Pred − True)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")

    # Annotate with early/late
    n_early = (residuals > 0).sum()
    n_late = (residuals < 0).sum()
    ax2.text(0.02, 0.95, f"Early: {n_early}  Late: {n_late}",
             transform=ax2.transAxes, fontsize=10, verticalalignment="top")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Saved RUL predictions plot → {save_path}")

    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance (XGBoost)",
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of top-N feature importances.

    Args:
        feature_names: List of feature names.
        importances: Array of importance values.
        top_n: Number of top features to show.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save.

    Returns:
        Matplotlib Figure.
    """
    _apply_style()
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(idx)), importances[idx], align="center")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Saved feature importance → {save_path}")

    return fig


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    figsize: tuple[int, int] = (10, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        history: Dict with 'train_loss' and optionally 'val_loss' lists.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save.

    Returns:
        Matplotlib Figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="Val Loss")
        best_epoch = np.argmin(history["val_loss"]) + 1
        best_val = min(history["val_loss"])
        ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7)
        ax.annotate(f"Best: {best_val:.4f}", xy=(best_epoch, best_val),
                    fontsize=9, ha="left")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Saved training history → {save_path}")

    return fig


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metrics: tuple[str, ...] = ("rmse", "mae", "nasa_score"),
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing multiple models across metrics.

    Args:
        results: Dict mapping model_name → metrics dict.
        metrics: Which metrics to plot.
        figsize: Figure size.
        save_path: Optional path to save.

    Returns:
        Matplotlib Figure.
    """
    _apply_style()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    models = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for ax, metric in zip(axes, metrics):
        values = [
            results[m].get(metric, results[m].get(f"test_{metric}", 0))
            for m in models
        ]
        bars = ax.bar(models, values, color=colors, edgecolor="black", alpha=0.8)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Saved model comparison → {save_path}")

    return fig

# TurboFault v0.1.0
# Any usage is subject to this software's license.
