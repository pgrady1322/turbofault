"""
TurboFault v0.1.0

features.py — Feature engineering for turbofan sensor time-series.

Transforms raw sensor readings into predictive features:
  - Rolling statistics (mean, std, min, max, skew) over configurable windows
  - Lag features and rate-of-change (delta) features
  - Sensor cross-correlations and interaction terms
  - Operational regime clustering features

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging

import pandas as pd

from turbofault.data.dataset import OPERATIONAL_SETTINGS, SENSOR_COLUMNS

logger = logging.getLogger("turbofault")


def add_rolling_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    windows: tuple[int, ...] = (5, 10, 20),
    statistics: tuple[str, ...] = ("mean", "std"),
) -> pd.DataFrame:
    """
    Add rolling window statistics per engine for selected sensors.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        sensors: Sensor columns to use (default: all SENSOR_COLUMNS).
        windows: Window sizes in cycles.
        statistics: Aggregation functions ('mean', 'std', 'min', 'max', 'skew').

    Returns:
        DataFrame with new rolling feature columns appended.
    """
    sensors = sensors or SENSOR_COLUMNS
    df = df.copy()

    for window in windows:
        for stat in statistics:
            cols = {}
            for sensor in sensors:
                col_name = f"{sensor}_roll{window}_{stat}"
                grouped = df.groupby("engine_id")[sensor]
                rolled = grouped.transform(
                    lambda x, w=window, s=stat: x.rolling(window=w, min_periods=1).agg(s)
                )
                cols[col_name] = rolled
            new_cols = pd.DataFrame(cols, index=df.index)
            df = pd.concat([df, new_cols], axis=1)

    n_features = sum(len(sensors) for _ in windows for _ in statistics)
    logger.info(
        f"✓ Added {n_features} rolling features " f"(windows={windows}, stats={statistics})"
    )
    return df


def add_lag_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    lags: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """
    Add lagged sensor values per engine.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        sensors: Sensor columns to lag (default: all SENSOR_COLUMNS).
        lags: Number of cycles to lag.

    Returns:
        DataFrame with lagged feature columns.
    """
    sensors = sensors or SENSOR_COLUMNS
    df = df.copy()

    for lag in lags:
        for sensor in sensors:
            col_name = f"{sensor}_lag{lag}"
            df[col_name] = df.groupby("engine_id")[sensor].shift(lag)

    # Fill NaN at start of each engine's time-series with first valid value
    lag_cols = [c for c in df.columns if "_lag" in c]
    df[lag_cols] = df.groupby("engine_id")[lag_cols].transform(lambda x: x.bfill())

    logger.info(f"✓ Added {len(sensors) * len(lags)} lag features (lags={lags})")
    return df


def add_delta_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    periods: tuple[int, ...] = (1, 5),
) -> pd.DataFrame:
    """
    Add rate-of-change (delta) features per engine.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        sensors: Sensor columns (default: all SENSOR_COLUMNS).
        periods: Number of cycles for difference computation.

    Returns:
        DataFrame with delta feature columns.
    """
    sensors = sensors or SENSOR_COLUMNS
    df = df.copy()

    for period in periods:
        for sensor in sensors:
            col_name = f"{sensor}_delta{period}"
            df[col_name] = df.groupby("engine_id")[sensor].diff(periods=period)

    delta_cols = [c for c in df.columns if "_delta" in c]
    df[delta_cols] = df[delta_cols].fillna(0.0)

    logger.info(f"✓ Added {len(sensors) * len(periods)} delta features " f"(periods={periods})")
    return df


def add_ewma_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    spans: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    """
    Add exponentially weighted moving average features per engine.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        sensors: Sensor columns (default: all SENSOR_COLUMNS).
        spans: EWMA span parameters.

    Returns:
        DataFrame with EWMA feature columns.
    """
    sensors = sensors or SENSOR_COLUMNS
    df = df.copy()

    for span in spans:
        for sensor in sensors:
            col_name = f"{sensor}_ewma{span}"
            df[col_name] = df.groupby("engine_id")[sensor].transform(
                lambda x, s=span: x.ewm(span=s, min_periods=1).mean()
            )

    logger.info(f"✓ Added {len(sensors) * len(spans)} EWMA features (spans={spans})")
    return df


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized cycle position features per engine.

    Creates:
        - cycle_norm: cycle / max_cycle for each engine (0 → 1)
        - cycle_squared: cycle_norm^2 (captures non-linear degradation)

    Args:
        df: DataFrame with engine_id and cycle columns.

    Returns:
        DataFrame with cycle position features.
    """
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycles
    df["cycle_squared"] = df["cycle_norm"] ** 2
    logger.info("✓ Added cycle normalization features")
    return df


def build_feature_set(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    rolling_windows: tuple[int, ...] = (5, 10, 20),
    rolling_stats: tuple[str, ...] = ("mean", "std"),
    lags: tuple[int, ...] = (1, 3, 5),
    delta_periods: tuple[int, ...] = (1, 5),
    ewma_spans: tuple[int, ...] = (10, 20),
    include_cycle: bool = True,
    include_operational: bool = True,
) -> pd.DataFrame:
    """
    Build the full engineered feature set.

    Applies all feature transformations in sequence:
      1. Rolling statistics
      2. Lag features
      3. Delta (rate-of-change) features
      4. EWMA features
      5. Cycle position features

    Args:
        df: Raw C-MAPSS DataFrame with engine_id, cycle, sensors.
        sensors: Which sensors to engineer (default: all).
        rolling_windows: Windows for rolling stats.
        rolling_stats: Statistics for rolling windows.
        lags: Lag periods.
        delta_periods: Periods for diff/delta.
        ewma_spans: Spans for EWMA.
        include_cycle: Include normalized cycle features.
        include_operational: Include operational settings.

    Returns:
        Feature-engineered DataFrame ready for modeling.
    """
    sensors = sensors or SENSOR_COLUMNS
    logger.info(f"Building feature set for {len(sensors)} sensors...")

    df = add_rolling_features(df, sensors, rolling_windows, rolling_stats)
    df = add_lag_features(df, sensors, lags)
    df = add_delta_features(df, sensors, delta_periods)
    df = add_ewma_features(df, sensors, ewma_spans)

    if include_cycle:
        df = add_cycle_features(df)

    # Count total engineered features
    base_cols = {"engine_id", "cycle", "rul"} | set(OPERATIONAL_SETTINGS) | set(SENSOR_COLUMNS)
    engineered_cols = [c for c in df.columns if c not in base_cols]
    logger.info(
        f"✓ Feature set complete: {len(engineered_cols)} engineered features "
        f"+ {len(sensors)} raw sensors"
    )

    return df


def get_feature_columns(
    df: pd.DataFrame,
    include_operational: bool = True,
    include_raw_sensors: bool = True,
) -> list[str]:
    """
    Return the list of feature columns (excluding engine_id, cycle, rul).

    Args:
        df: Feature-engineered DataFrame.
        include_operational: Include operational setting columns.
        include_raw_sensors: Include raw sensor columns.

    Returns:
        List of column names to use as model features.
    """
    exclude = {"engine_id", "cycle", "rul"}
    if not include_operational:
        exclude.update(OPERATIONAL_SETTINGS)
    if not include_raw_sensors:
        exclude.update(SENSOR_COLUMNS)

    return [c for c in df.columns if c not in exclude]


# TurboFault v0.1.0
# Any usage is subject to this software's license.
