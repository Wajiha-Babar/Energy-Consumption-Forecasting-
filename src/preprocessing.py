import pandas as pd

from config import TARGET_COLUMN


def normalize_frequency(frequency: str) -> str:
    frequency_map = {
        "Hourly": "h",
        "hourly": "h",
        "H": "h",
        "h": "h",
        "Daily": "D",
        "daily": "D",
        "D": "D",
        "d": "D",
    }

    if frequency not in frequency_map:
        raise ValueError(f"Invalid frequency '{frequency}'. Use Hourly, Daily, h, or D.")

    return frequency_map[frequency]


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing time-series values with interpolation, forward fill, then backward fill."""
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    cleaned[numeric_columns] = cleaned[numeric_columns].interpolate(method="time")
    cleaned[numeric_columns] = cleaned[numeric_columns].ffill().bfill()
    return cleaned


def resample_power_data(
    df: pd.DataFrame,
    frequency: str = "h",
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Resample target consumption to hourly or daily mean values."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    frequency = normalize_frequency(frequency)
    resampled = df[[target_column]].resample(frequency).mean()
    return clean_missing_values(resampled)


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series chronologically."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    split_index = int(len(df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Not enough records for the requested train/test split.")
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def limit_recent_data(df: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
    """Return the latest N days of data, or the original frame when days is None."""
    if days is None or df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df.loc[df.index >= cutoff].copy()
