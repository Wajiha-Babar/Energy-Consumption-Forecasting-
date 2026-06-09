from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TXT_DATASET_PATH = DATASET_DIR / "household_power_consumption.txt"
ZIP_DATASET_PATH = DATASET_DIR / "individual+household+electric+power+consumption.zip"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"

TARGET_COLUMN = "Global_active_power"
RANDOM_STATE = 42

ARIMA_ORDER = (2, 1, 2)
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
}

