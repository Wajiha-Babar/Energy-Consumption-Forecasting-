import pandas as pd

from config import TARGET_COLUMN


LAG_FEATURES = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS = [3, 6, 12, 24]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    index = featured.index
    featured["hour"] = index.hour
    featured["day"] = index.day
    featured["day_of_week"] = index.dayofweek
    featured["month"] = index.month
    featured["year"] = index.year
    featured["is_weekend"] = (index.dayofweek >= 5).astype(int)
    featured["quarter"] = index.quarter
    featured["week_of_year"] = index.isocalendar().week.astype(int)
    return featured


def add_lag_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    featured = df.copy()
    for lag in lags or LAG_FEATURES:
        featured[f"lag_{lag}"] = featured[target_column].shift(lag)
    return featured


def add_rolling_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    featured = df.copy()
    for window in ROLLING_WINDOWS:
        featured[f"rolling_mean_{window}"] = featured[target_column].shift(1).rolling(window).mean()
    featured["rolling_std_24"] = featured[target_column].shift(1).rolling(24).std()
    return featured


def create_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Create time, lag, and rolling features for supervised forecasting."""
    featured = add_time_features(df)
    featured = add_lag_features(featured, target_column=target_column)
    featured = add_rolling_features(featured, target_column=target_column)
    return featured.dropna()


def get_feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    return [column for column in df.columns if column != target_column]

