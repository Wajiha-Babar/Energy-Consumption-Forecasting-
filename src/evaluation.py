import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def build_comparison_table(results: dict[str, pd.Series], actual: pd.Series) -> pd.DataFrame:
    rows = []
    for model_name, predictions in results.items():
        aligned_actual = actual.loc[predictions.index]
        metrics = calculate_metrics(aligned_actual, predictions)
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
