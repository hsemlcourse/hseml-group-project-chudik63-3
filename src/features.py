"""Feature engineering transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import LEAKAGE_COLUMNS


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop service, target and leakage columns if they are present."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        return X.drop(columns=[col for col in self.columns if col in X.columns], errors="ignore")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create interpretable weather features without using future information."""

    def __init__(self, drop_date: bool = True) -> None:
        self.drop_date = drop_date

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(columns=[col for col in LEAKAGE_COLUMNS if col in X.columns], errors="ignore")

        if "Date" in X.columns:
            date = pd.to_datetime(X["Date"], errors="coerce")
            X["Year"] = date.dt.year
            X["Month"] = date.dt.month
            X["DayOfYear"] = date.dt.dayofyear
            X["WeekOfYear"] = date.dt.isocalendar().week.astype("float")
            X["Season"] = X["Month"].map(self._month_to_season)
            if self.drop_date:
                X = X.drop(columns=["Date"])

        self._safe_diff(X, "MaxTemp", "MinTemp", "TempRange")
        self._safe_mean(X, ["MaxTemp", "MinTemp"], "TempMean")
        self._safe_diff(X, "Humidity3pm", "Humidity9am", "HumidityChange")
        self._safe_diff(X, "Pressure3pm", "Pressure9am", "PressureChange")
        self._safe_mean(X, ["WindSpeed9am", "WindSpeed3pm"], "WindSpeedMean")
        self._safe_ratio(X, "WindGustSpeed", "WindSpeedMean", "WindGustToMeanSpeed")

        if "Rainfall" in X.columns:
            X["RainfallLog1p"] = np.log1p(pd.to_numeric(X["Rainfall"], errors="coerce").clip(lower=0))
            X["HadRainTodayByMm"] = (pd.to_numeric(X["Rainfall"], errors="coerce") > 0).astype("float")

        if "Sunshine" in X.columns:
            X["LowSunshine"] = (pd.to_numeric(X["Sunshine"], errors="coerce") < 5).astype("float")

        return X

    @staticmethod
    def _month_to_season(month: float | int | None) -> str | float:
        if pd.isna(month):
            return np.nan
        month = int(month)
        if month in [12, 1, 2]:
            return "summer"
        if month in [3, 4, 5]:
            return "autumn"
        if month in [6, 7, 8]:
            return "winter"
        return "spring"

    @staticmethod
    def _safe_diff(X: pd.DataFrame, left: str, right: str, new_col: str) -> None:
        if left in X.columns and right in X.columns:
            X[new_col] = pd.to_numeric(X[left], errors="coerce") - pd.to_numeric(
                X[right], errors="coerce"
            )

    @staticmethod
    def _safe_mean(X: pd.DataFrame, cols: list[str], new_col: str) -> None:
        existing_cols = [col for col in cols if col in X.columns]
        if existing_cols:
            X[new_col] = X[existing_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    @staticmethod
    def _safe_ratio(X: pd.DataFrame, numerator: str, denominator: str, new_col: str) -> None:
        if numerator in X.columns and denominator in X.columns:
            den = pd.to_numeric(X[denominator], errors="coerce").replace(0, np.nan)
            X[new_col] = pd.to_numeric(X[numerator], errors="coerce") / den
