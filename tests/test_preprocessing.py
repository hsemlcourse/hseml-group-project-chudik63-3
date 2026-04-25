import pandas as pd

from src.features import FeatureEngineer
from src.preprocessing import basic_clean, chronological_train_val_test_split


def test_basic_clean_removes_duplicates_and_converts_target():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "RainTomorrow": ["Yes", "Yes", "No"],
            "MinTemp": [1.0, 1.0, 2.0],
        }
    )

    cleaned = basic_clean(df)

    assert len(cleaned) == 2
    assert cleaned["RainTomorrow"].tolist() == [1, 0]
    assert str(cleaned["Date"].dtype).startswith("datetime64")


def test_feature_engineer_adds_weather_features_and_drops_leakage():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "MinTemp": [10.0],
            "MaxTemp": [25.0],
            "Humidity9am": [80.0],
            "Humidity3pm": [50.0],
            "Pressure9am": [1012.0],
            "Pressure3pm": [1009.0],
            "WindSpeed9am": [10.0],
            "WindSpeed3pm": [20.0],
            "WindGustSpeed": [40.0],
            "Rainfall": [2.0],
            "RISK_MM": [99.0],
        }
    )

    transformed = FeatureEngineer().fit_transform(df)

    assert "RISK_MM" not in transformed.columns
    assert "Date" not in transformed.columns
    assert transformed.loc[0, "TempRange"] == 15.0
    assert transformed.loc[0, "HumidityChange"] == -30.0
    assert transformed.loc[0, "PressureChange"] == -3.0
    assert "Season" in transformed.columns


def test_chronological_split_keeps_time_order():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=10),
            "RainTomorrow": [0, 1] * 5,
        }
    )

    train, val, test = chronological_train_val_test_split(df, train_size=0.6, val_size=0.2)

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()
