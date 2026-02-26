"""
TurboFault v0.1.0

preprocessing.py — Data preprocessing for turbofan prognostics.

Handles sensor normalization, sliding window construction for sequence
models, and train/validation split strategies that respect temporal order.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from turbofault.data.dataset import (
    LOW_VARIANCE_SENSORS,
    OPERATIONAL_SETTINGS,
    SENSOR_COLUMNS,
    CMAPSSDataset,
)

logger = logging.getLogger("turbofault")


def normalize_sensors(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    columns: Optional[list[str]] = None,
    method: str = "minmax",
) -> tuple[pd.DataFrame, pd.DataFrame | None, MinMaxScaler | StandardScaler]:
    """
    Normalize sensor columns, fitting scaler on training data only.

    Args:
        train_df: Training DataFrame.
        test_df: Optional test DataFrame (transformed with train statistics).
        columns: Columns to normalize (default: sensors + op settings).
        method: 'minmax' (0-1) or 'standard' (zero mean, unit variance).

    Returns:
        Tuple of (normalized_train, normalized_test_or_None, fitted_scaler).
    """
    columns = columns or (SENSOR_COLUMNS + OPERATIONAL_SETTINGS)
    existing_cols = [c for c in columns if c in train_df.columns]

    if method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'minmax' or 'standard'.")

    train_out = train_df.copy()
    train_out[existing_cols] = scaler.fit_transform(train_df[existing_cols])

    test_out = None
    if test_df is not None:
        test_out = test_df.copy()
        test_out[existing_cols] = scaler.transform(test_df[existing_cols])

    logger.info(f"✓ Normalized {len(existing_cols)} columns using {method} scaling")
    return train_out, test_out, scaler


def drop_low_variance_sensors(
    df: pd.DataFrame,
    sensors: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Drop sensors that have near-zero variance (uninformative).

    The default list comes from analysis of FD001; adjust for other subsets.

    Args:
        df: DataFrame with sensor columns.
        sensors: Sensors to drop (default: LOW_VARIANCE_SENSORS).

    Returns:
        DataFrame with specified sensors removed.
    """
    sensors = sensors or LOW_VARIANCE_SENSORS
    existing = [s for s in sensors if s in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"✓ Dropped {len(existing)} low-variance sensors: {existing}")
    return df


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rul",
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM/Transformer/CNN models.

    For each engine, slides a window of `window_size` cycles and extracts:
      - X: sensor feature windows  → (n_samples, window_size, n_features)
      - y: RUL at the last cycle in the window → (n_samples,)
      - engine_ids: engine ID for each window → (n_samples,)

    Engines shorter than window_size are padded with zeros at the start.

    Args:
        df: Feature-engineered DataFrame with engine_id, cycle, features, RUL.
        feature_columns: Columns to include in the feature window.
        target_column: Target column name (usually 'rul').
        window_size: Number of cycles per window.
        stride: Step size between windows.

    Returns:
        (X, y, engine_ids) arrays.
    """
    X_list, y_list, id_list = [], [], []

    for engine_id, engine_df in df.groupby("engine_id"):
        features = engine_df[feature_columns].values
        targets = engine_df[target_column].values

        # Zero-pad if engine has fewer cycles than window_size
        if len(features) < window_size:
            pad_len = window_size - len(features)
            features = np.vstack([
                np.zeros((pad_len, features.shape[1])),
                features,
            ])
            targets = np.concatenate([
                np.full(pad_len, targets[0]),
                targets,
            ])

        # Slide window
        for i in range(0, len(features) - window_size + 1, stride):
            X_list.append(features[i: i + window_size])
            y_list.append(targets[i + window_size - 1])
            id_list.append(engine_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    engine_ids = np.array(id_list)

    logger.info(f"✓ Created {len(X):,} sequences: "
                f"shape=({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")
    return X, y, engine_ids


def temporal_train_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    by_engine: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data respecting temporal order (no future leakage).

    Two strategies:
      - by_engine=True:  Hold out last N engines as validation.
      - by_engine=False: For each engine, hold out last N% of cycles.

    Args:
        df: DataFrame with engine_id and cycle columns.
        val_fraction: Fraction for validation (0.2 = 20%).
        by_engine: Split by engine ID or by cycle within engine.

    Returns:
        (train_df, val_df)
    """
    if by_engine:
        engine_ids = sorted(df["engine_id"].unique())
        n_val = max(1, int(len(engine_ids) * val_fraction))
        val_engines = engine_ids[-n_val:]
        train_engines = engine_ids[:-n_val]

        train_df = df[df["engine_id"].isin(train_engines)].copy()
        val_df = df[df["engine_id"].isin(val_engines)].copy()

        logger.info(f"✓ Temporal split (by engine): "
                    f"{len(train_engines)} train, {len(val_engines)} val engines")
    else:
        train_parts, val_parts = [], []
        for engine_id, engine_df in df.groupby("engine_id"):
            engine_df = engine_df.sort_values("cycle")
            n_val = max(1, int(len(engine_df) * val_fraction))
            train_parts.append(engine_df.iloc[:-n_val])
            val_parts.append(engine_df.iloc[-n_val:])

        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)

        logger.info(f"✓ Temporal split (by cycle): "
                    f"{len(train_df):,} train, {len(val_df):,} val samples")

    return train_df, val_df


def get_last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the last cycle for each engine (used for test-set evaluation).

    In C-MAPSS, the test set RUL is defined at the last available cycle.

    Args:
        df: DataFrame with engine_id and cycle columns.

    Returns:
        DataFrame with one row per engine (the final cycle).
    """
    idx = df.groupby("engine_id")["cycle"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def prepare_tabular_data(
    dataset: CMAPSSDataset,
    feature_columns: list[str],
    normalize: bool = True,
    norm_method: str = "minmax",
    drop_low_var: bool = True,
    val_fraction: float = 0.2,
) -> dict:
    """
    End-to-end preparation for tabular models (XGBoost, RF, etc.).

    Args:
        dataset: Loaded CMAPSSDataset.
        feature_columns: Columns to use as features.
        normalize: Whether to normalize features.
        norm_method: 'minmax' or 'standard'.
        drop_low_var: Drop known low-variance sensors.
        val_fraction: Validation split fraction.

    Returns:
        Dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
        feature_columns, scaler.
    """
    train_df = dataset.train_df.copy()
    test_df = dataset.test_df.copy()

    if drop_low_var:
        train_df = drop_low_variance_sensors(train_df)
        test_df = drop_low_variance_sensors(test_df)
        feature_columns = [c for c in feature_columns if c in train_df.columns]

    scaler = None
    if normalize:
        train_df, test_df, scaler = normalize_sensors(
            train_df, test_df, columns=feature_columns, method=norm_method
        )

    # Split training data
    train_split, val_split = temporal_train_val_split(
        train_df, val_fraction=val_fraction
    )

    # Test: use last cycle per engine
    test_last = get_last_cycle_per_engine(test_df)

    existing_features = [c for c in feature_columns if c in train_split.columns]

    return {
        "X_train": train_split[existing_features].values,
        "y_train": train_split["rul"].values,
        "X_val": val_split[existing_features].values,
        "y_val": val_split["rul"].values,
        "X_test": test_last[existing_features].values,
        "y_test": test_last["rul"].values,
        "feature_columns": existing_features,
        "scaler": scaler,
    }


def prepare_sequence_data(
    dataset: CMAPSSDataset,
    feature_columns: list[str],
    window_size: int = 30,
    stride: int = 1,
    normalize: bool = True,
    norm_method: str = "minmax",
    drop_low_var: bool = True,
    val_fraction: float = 0.2,
) -> dict:
    """
    End-to-end preparation for sequence models (LSTM, Transformer, CNN).

    Args:
        dataset: Loaded CMAPSSDataset.
        feature_columns: Columns to use as features.
        window_size: Sliding window size.
        stride: Window stride.
        normalize: Whether to normalize.
        norm_method: Normalization method.
        drop_low_var: Drop low-variance sensors.
        val_fraction: Validation fraction.

    Returns:
        Dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
        engine_ids_test, feature_columns, scaler, window_size.
    """
    train_df = dataset.train_df.copy()
    test_df = dataset.test_df.copy()

    if drop_low_var:
        train_df = drop_low_variance_sensors(train_df)
        test_df = drop_low_variance_sensors(test_df)
        feature_columns = [c for c in feature_columns if c in train_df.columns]

    scaler = None
    if normalize:
        train_df, test_df, scaler = normalize_sensors(
            train_df, test_df, columns=feature_columns, method=norm_method
        )

    # Split before windowing
    train_split, val_split = temporal_train_val_split(
        train_df, val_fraction=val_fraction
    )

    existing_features = [c for c in feature_columns if c in train_split.columns]

    X_train, y_train, _ = create_sequences(
        train_split, existing_features, window_size=window_size, stride=stride
    )
    X_val, y_val, _ = create_sequences(
        val_split, existing_features, window_size=window_size, stride=stride
    )
    X_test, y_test, engine_ids_test = create_sequences(
        test_df, existing_features, window_size=window_size, stride=stride
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "engine_ids_test": engine_ids_test,
        "feature_columns": existing_features,
        "scaler": scaler,
        "window_size": window_size,
    }

# TurboFault v0.1.0
# Any usage is subject to this software's license.
