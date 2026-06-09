from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from config import TXT_DATASET_PATH, ZIP_DATASET_PATH


NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def find_dataset(
    txt_path: Path = TXT_DATASET_PATH,
    zip_path: Path = ZIP_DATASET_PATH,
) -> tuple[str, Path]:
    """Return the available dataset type and path."""
    if txt_path.exists():
        return "txt", txt_path
    if zip_path.exists():
        return "zip", zip_path
    raise FileNotFoundError(
        "Dataset not found. Place household_power_consumption.txt or "
        "individual+household+electric+power+consumption.zip inside the dataset folder."
    )


def _read_from_zip(zip_path: Path) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        txt_members = [
            name for name in archive.namelist() if name.endswith("household_power_consumption.txt")
        ]
        if not txt_members:
            raise FileNotFoundError("The zip file does not contain household_power_consumption.txt.")
        with archive.open(txt_members[0]) as file:
            return pd.read_csv(file, sep=";", na_values="?", low_memory=False)


def load_raw_data() -> pd.DataFrame:
    """Load the power consumption dataset from TXT or ZIP."""
    dataset_type, dataset_path = find_dataset()
    if dataset_type == "txt":
        return pd.read_csv(dataset_path, sep=";", na_values="?", low_memory=False)
    return _read_from_zip(dataset_path)


def load_power_consumption_data() -> pd.DataFrame:
    """Load, parse datetime, type numeric columns, and index the dataset."""
    df = load_raw_data()
    required_columns = {"Date", "Time", *NUMERIC_COLUMNS}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.drop(columns=["Date", "Time"])

