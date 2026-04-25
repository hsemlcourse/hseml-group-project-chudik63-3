"""Data download and parsing helpers."""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd

from src.config import DATASET_HANDLE, RAW_CSV_FILENAME, TARGET


def download_dataset(dataset_handle: str = DATASET_HANDLE) -> Path:
    """Download the Kaggle dataset via kagglehub and return local path."""
    path = kagglehub.dataset_download(dataset_handle)
    return Path(path)


def find_weather_csv(dataset_path: str | Path) -> Path:
    """Find `weatherAUS.csv` in the downloaded Kaggle dataset folder."""
    dataset_path = Path(dataset_path)
    exact_match = dataset_path / RAW_CSV_FILENAME
    if exact_match.exists():
        return exact_match

    candidates = list(dataset_path.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    for candidate in candidates:
        if candidate.name == RAW_CSV_FILENAME:
            return candidate
    return candidates[0]


def load_weather_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load weather data, parse dates and validate the target column."""
    if csv_path is None:
        csv_path = find_weather_csv(download_dataset())

    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if TARGET not in df.columns:
        raise ValueError(f"Target column `{TARGET}` is absent. Available columns: {list(df.columns)}")

    return df
