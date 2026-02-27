"""
TurboFault v0.1.0

dataset.py — NASA C-MAPSS Turbofan Engine Degradation dataset loader.

The C-MAPSS dataset contains run-to-failure simulations for turbofan engines
under different operating conditions and fault modes:

| Subset | Engines (train) | Engines (test) | Op. Conditions | Fault Modes |
|--------|-----------------|----------------|----------------|-------------|
| FD001  | 100             | 100            | 1 (sea level)  | 1 (HPC)     |
| FD002  | 260             | 259            | 6              | 1 (HPC)     |
| FD003  | 100             | 100            | 1 (sea level)  | 2 (HPC+Fan) |
| FD004  | 249             | 248            | 6              | 2 (HPC+Fan) |

Each engine has 3 operational settings + 21 sensor channels recorded per cycle.

Reference:
    Saxena et al. "Damage Propagation Modeling for Aircraft Engine
    Run-to-Failure Simulation" (PHM 2008)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger("turbofault")

# ── Column definitions ──────────────────────────────────────────────
OPERATIONAL_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

SENSOR_COLUMNS = [
    "sensor_1",  # Total temperature at fan inlet (°R)
    "sensor_2",  # Total temperature at LPC outlet (°R)
    "sensor_3",  # Total temperature at HPC outlet (°R)
    "sensor_4",  # Total temperature at LPT outlet (°R)
    "sensor_5",  # Pressure at fan inlet (psia)
    "sensor_6",  # Total pressure in bypass-duct (psia)
    "sensor_7",  # Total pressure at HPC outlet (psia)
    "sensor_8",  # Physical fan speed (rpm)
    "sensor_9",  # Physical core speed (rpm)
    "sensor_10",  # Engine pressure ratio (P50/P2)
    "sensor_11",  # Static pressure at HPC outlet (psia)
    "sensor_12",  # Ratio of fuel flow to Ps30 (pps/psi)
    "sensor_13",  # Corrected fan speed (rpm)
    "sensor_14",  # Corrected core speed (rpm)
    "sensor_15",  # Bypass ratio
    "sensor_16",  # Burner fuel-air ratio
    "sensor_17",  # Bleed enthalpy
    "sensor_18",  # Demanded fan speed (rpm)
    "sensor_19",  # Demanded corrected fan speed (rpm)
    "sensor_20",  # HPT coolant bleed (lbm/s)
    "sensor_21",  # LPT coolant bleed (lbm/s)
]

ALL_COLUMNS = ["engine_id", "cycle"] + OPERATIONAL_SETTINGS + SENSOR_COLUMNS

# Sensors with near-zero variance in FD001 (often dropped)
LOW_VARIANCE_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# C-MAPSS download URLs (primary: PHM Society S3 mirror; fallback: NASA)
CMAPSS_URLS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    "https://ti.arc.nasa.gov/c/6/",
]
CMAPSS_URL = CMAPSS_URLS[0]  # kept for backwards-compat

# Subset metadata
SUBSET_INFO = {
    "FD001": {"train_engines": 100, "test_engines": 100, "op_conditions": 1, "fault_modes": 1},
    "FD002": {"train_engines": 260, "test_engines": 259, "op_conditions": 6, "fault_modes": 1},
    "FD003": {"train_engines": 100, "test_engines": 100, "op_conditions": 1, "fault_modes": 2},
    "FD004": {"train_engines": 249, "test_engines": 248, "op_conditions": 6, "fault_modes": 2},
}


@dataclass
class CMAPSSDataset:
    """Container for a single C-MAPSS subset (FD001–FD004)."""

    subset: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    rul_df: pd.DataFrame
    max_rul: int = 125  # Cap RUL at piecewise-linear threshold

    @property
    def num_train_engines(self) -> int:
        return self.train_df["engine_id"].nunique()

    @property
    def num_test_engines(self) -> int:
        return self.test_df["engine_id"].nunique()

    @property
    def num_sensors(self) -> int:
        return len(SENSOR_COLUMNS)

    @property
    def sensor_columns(self) -> list[str]:
        return SENSOR_COLUMNS

    @property
    def operational_columns(self) -> list[str]:
        return OPERATIONAL_SETTINGS

    def add_rul_column(self) -> None:
        """
        Add Remaining Useful Life (RUL) to training data.

        Uses piecewise-linear RUL: constant at max_rul for early cycles,
        then linear decay to 0 at failure. This prevents the model from
        trying to predict exact RUL at engine start (which is noisy).
        """
        rul_list = []
        for engine_id in self.train_df["engine_id"].unique():
            engine = self.train_df[self.train_df["engine_id"] == engine_id]
            max_cycle = engine["cycle"].max()
            rul = max_cycle - engine["cycle"]
            rul = rul.clip(upper=self.max_rul)
            rul_list.append(rul)
        self.train_df["rul"] = pd.concat(rul_list).values

    def add_test_rul(self) -> None:
        """Add RUL to test data using ground truth RUL file."""
        rul_list = []
        for engine_id in self.test_df["engine_id"].unique():
            engine = self.test_df[self.test_df["engine_id"] == engine_id]
            max_cycle = engine["cycle"].max()
            true_rul = self.rul_df.iloc[engine_id - 1, 0]
            rul = true_rul + max_cycle - engine["cycle"]
            rul = rul.clip(upper=self.max_rul)
            rul_list.append(rul)
        self.test_df["rul"] = pd.concat(rul_list).values

    def summary(self) -> str:
        """Return human-readable dataset summary."""
        lines = [
            f"C-MAPSS {self.subset}",
            f"  Train engines: {self.num_train_engines}",
            f"  Test engines:  {self.num_test_engines}",
            f"  Train samples: {len(self.train_df):,}",
            f"  Test samples:  {len(self.test_df):,}",
            f"  Sensors:       {self.num_sensors}",
            f"  Op settings:   {len(OPERATIONAL_SETTINGS)}",
            f"  Max RUL cap:   {self.max_rul}",
        ]
        if "rul" in self.train_df.columns:
            lines.append(
                f"  Train RUL range: [{self.train_df['rul'].min()}, "
                f"{self.train_df['rul'].max()}]"
            )
        return "\n".join(lines)


def _read_cmapss_file(filepath: Path) -> pd.DataFrame:
    """Read a whitespace-separated C-MAPSS data file."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
    # Drop trailing NaN columns if present
    df = df.dropna(axis=1, how="all")
    if len(df.columns) == len(ALL_COLUMNS):
        df.columns = ALL_COLUMNS
    elif len(df.columns) == 1:
        # RUL file — single column
        df.columns = ["rul"]
    else:
        logger.warning(f"Unexpected column count ({len(df.columns)}) in {filepath}")
        cols = ALL_COLUMNS[: len(df.columns)]
        df.columns = cols
    return df


def download_cmapss(output_dir: Path) -> Path:
    """
    Download the C-MAPSS dataset from NASA.

    Args:
        output_dir: Directory to save extracted files.

    Returns:
        Path to the extracted data directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "CMAPSSData.zip"
    extract_dir = output_dir / "CMAPSSData"

    if extract_dir.exists():
        logger.info(f"✓ C-MAPSS data already exists: {extract_dir}")
        return extract_dir

    logger.info("Downloading C-MAPSS dataset...")

    def _download(url: str, dest: Path) -> None:
        """Download *url* to *dest*, falling back to unverified SSL if needed."""
        try:
            urllib.request.urlretrieve(url, dest)
        except (ssl.SSLCertVerificationError, urllib.error.URLError):
            logger.warning("SSL verification failed — retrying without verification")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            with opener.open(url) as resp, open(dest, "wb") as f:
                f.write(resp.read())

    # Try each mirror until we get a valid zip
    last_err: Exception | None = None
    for url in CMAPSS_URLS:
        logger.info(f"  Trying {url} ...")
        try:
            _download(url, zip_path)
            # Validate that we actually got a zip (not an HTML landing page)
            if not zipfile.is_zipfile(zip_path):
                zip_path.unlink(missing_ok=True)
                raise ValueError("Downloaded file is not a valid zip (URL may have changed)")
            break  # success
        except Exception as e:
            last_err = e
            logger.warning(f"  Mirror failed: {e}")
            zip_path.unlink(missing_ok=True)
    else:
        logger.error(f"All download mirrors failed. Last error: {last_err}")
        logger.info("Manual download: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        logger.info("Extract into <output_dir>/CMAPSSData/ and re-run.")
        raise RuntimeError("C-MAPSS download failed from all mirrors") from last_err

    logger.info("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Handle nested zip: the PHM S3 mirror wraps CMAPSSData.zip inside an
    # outer zip under a long-named folder.  The inner zip has the 14 data
    # files at the root (no CMAPSSData/ prefix), so we extract them into
    # the expected extract_dir.
    if not extract_dir.exists():
        inner_zips = [p for p in output_dir.rglob("CMAPSSData.zip") if p != zip_path]
        if inner_zips:
            inner = inner_zips[0]
            logger.info(f"  Found nested zip: {inner.relative_to(output_dir)}")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(inner, "r") as zf2:
                zf2.extractall(extract_dir)
            # Clean up the outer extraction folder
            inner.unlink()
            inner_parent = inner.parent
            if inner_parent != output_dir:
                import shutil

                shutil.rmtree(inner_parent, ignore_errors=True)

    zip_path.unlink(missing_ok=True)
    logger.info(f"✓ Extracted to {extract_dir}")
    return extract_dir


def load_cmapss(
    data_dir: Path,
    subset: str = "FD001",
    max_rul: int = 125,
    add_rul: bool = True,
) -> CMAPSSDataset:
    """
    Load a C-MAPSS subset with optional RUL computation.

    Args:
        data_dir: Path to the CMAPSSData/ directory.
        subset: One of FD001, FD002, FD003, FD004.
        max_rul: Piecewise-linear RUL cap (default 125 cycles).
        add_rul: Whether to add RUL column to train/test data.

    Returns:
        CMAPSSDataset with train_df, test_df, rul_df loaded.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    subset = subset.upper()
    if subset not in SUBSET_INFO:
        raise ValueError(f"Invalid subset '{subset}'. Choose from: {list(SUBSET_INFO.keys())}")

    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    for p in [train_path, test_path, rul_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    logger.info(f"Loading C-MAPSS {subset}...")
    train_df = _read_cmapss_file(train_path)
    test_df = _read_cmapss_file(test_path)
    rul_df = _read_cmapss_file(rul_path)

    dataset = CMAPSSDataset(
        subset=subset,
        train_df=train_df,
        test_df=test_df,
        rul_df=rul_df,
        max_rul=max_rul,
    )

    if add_rul:
        dataset.add_rul_column()
        dataset.add_test_rul()
        logger.info(f"✓ RUL column added (capped at {max_rul})")

    logger.info(f"✓ Loaded {subset}: {len(train_df):,} train, {len(test_df):,} test samples")
    return dataset


# TurboFault v0.1.0
# Any usage is subject to this software's license.
