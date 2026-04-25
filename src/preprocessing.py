"""Preprocessing, cleaning and split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import LEAKAGE_COLUMNS, RANDOM_STATE, TARGET
from src.features import ColumnDropper, FeatureEngineer


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numerical columns by train quantiles to reduce influence of extreme values."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OutlierClipper":
        X = X.copy()
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.lower_bounds_ = X[self.numeric_columns_].quantile(self.lower_quantile)
        self.upper_bounds_ = X[self.numeric_columns_].quantile(self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        existing = [col for col in getattr(self, "numeric_columns_", []) if col in X.columns]
        if existing:
            X[existing] = X[existing].clip(
                lower=self.lower_bounds_[existing], upper=self.upper_bounds_[existing], axis=1
            )
        return X


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, parse dates, remove rows without target and normalize binary target values."""
    df = df.copy()
    df = df.drop_duplicates()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if TARGET in df.columns:
        df = df[df[TARGET].notna()].copy()
        df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1, 0: 0, 1: 1}).astype(int)

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def chronological_train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split to imitate future prediction and avoid time leakage."""
    if "Date" in df.columns:
        df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n_rows = len(df)
    train_end = int(n_rows * train_size)
    val_end = int(n_rows * (train_size + val_size))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def stratified_train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback stratified split for experiments where time split is not needed."""
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[TARGET],
        random_state=random_state,
    )
    relative_val_size = val_size / (1 - train_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df[TARGET],
        random_state=random_state,
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def build_column_preprocessor() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric and categorical weather features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_pipeline(model, use_feature_engineering: bool = True) -> Pipeline:
    """Build a full train-only fitted pipeline."""
    service_columns = LEAKAGE_COLUMNS + [TARGET]
    if not use_feature_engineering:
        service_columns = service_columns + ["Date"]

    steps = [("drop_service", ColumnDropper(service_columns))]
    if use_feature_engineering:
        steps.append(("features", FeatureEngineer(drop_date=True)))
    steps.extend(
        [
            ("outliers", OutlierClipper()),
            ("preprocess", build_column_preprocessor()),
            ("model", model),
        ]
    )
    return Pipeline(steps)
