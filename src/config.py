"""Project configuration."""

from pathlib import Path

DATASET_HANDLE = "jsphyg/weather-dataset-rattle-package"
RAW_CSV_FILENAME = "weatherAUS.csv"
TARGET = "RainTomorrow"
LEAKAGE_COLUMNS = ["RISK_MM"]
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"
