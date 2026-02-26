"""
TurboFault v0.1.0

trainer.py — Training orchestration for tabular and deep-learning models.

Supports:
  - XGBoost / Random Forest / Ridge (tabular, sklearn-style fit/predict)
  - LSTM / GRU / Transformer / CNN1D (PyTorch training loop)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("turbofault")


def train_tabular(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """
    Train a tabular model (XGBoost, RF, Ridge).

    Args:
        model: Model with fit(X, y) and predict(X) methods.
        X_train: Training features.
        y_train: Training target (RUL).
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        Dict with 'model' and 'train_time'.
    """
    t0 = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0

    logger.info(f"✓ Tabular model trained in {train_time:.1f}s")

    return {
        "model": model,
        "train_time": train_time,
    }


def train_deep(
    model: "torch.nn.Module",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str = "auto",
    save_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    PyTorch training loop for sequence models.

    Args:
        model: PyTorch model (LSTM, Transformer, CNN1D, etc.).
        X_train: Training features (n_samples, seq_len, n_features).
        y_train: Training targets (n_samples,).
        X_val: Validation features.
        y_val: Validation targets.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience.
        device: 'cpu', 'cuda', 'mps', or 'auto'.
        save_path: Optional path to save best model checkpoint.

    Returns:
        Dict with 'model', 'history', 'best_epoch', 'train_time'.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_obj = torch.device(device)
    model = model.to(device_obj)
    logger.info(f"Training on {device_obj} — {model.num_parameters:,} parameters")

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Optimizer + loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3
    )
    criterion = torch.nn.MSELoss()

    # Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device_obj)
            y_batch = y_batch.to(device_obj)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # ── Validate ──
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device_obj)
                    y_batch = y_batch.to(device_obj)
                    preds = model(X_batch)
                    val_losses.append(criterion(preds, y_batch).item())
            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(avg_val_loss)

            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                no_improve = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch} "
                                f"(best={best_epoch}, val_loss={best_val_loss:.4f})")
                    break

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            msg = f"Epoch {epoch:3d} — train_loss: {avg_train_loss:.4f}"
            if avg_val_loss is not None:
                msg += f"  val_loss: {avg_val_loss:.4f}"
            logger.info(msg)

    train_time = time.time() - t0

    # Reload best weights
    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device_obj, weights_only=True))
        logger.info(f"✓ Loaded best model from epoch {best_epoch}")

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_time": train_time,
    }


def predict_deep(
    model: "torch.nn.Module",
    X: np.ndarray,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    """
    Batch prediction with a PyTorch model.

    Args:
        model: Trained PyTorch model.
        X: Input features (n_samples, seq_len, n_features).
        batch_size: Batch size.
        device: Device to use.

    Returns:
        Predictions array of shape (n_samples,).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device_obj)
            out = model(X_batch)
            preds.append(out.cpu().numpy())

    return np.concatenate(preds).flatten()

# TurboFault v0.1.0
# Any usage is subject to this software's license.
