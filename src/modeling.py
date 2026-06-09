import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from config import ARIMA_ORDER, TARGET_COLUMN, XGBOOST_PARAMS
from src.feature_engineering import get_feature_columns


def train_arima_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    order: tuple[int, int, int] = ARIMA_ORDER,
    max_train_points: int = 24 * 90,
) -> pd.Series:
    """Fit ARIMA on a recent sample and forecast the test horizon."""
    train_series = train[target_column].dropna().iloc[-max_train_points:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train_series, order=order)
        fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    return pd.Series(forecast.to_numpy(), index=test.index, name="ARIMA")


def train_prophet_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.Series | None, str | None]:
    """Fit Prophet when available; return a clean skip message on failure."""
    try:
        from prophet import Prophet
    except Exception as exc:
        return None, f"Prophet model skipped because package is not installed or failed to run. Details: {exc}"

    try:
        prophet_train = train[[target_column]].reset_index()
        prophet_train.columns = ["ds", "y"]
        yearly = len(train) >= 24 * 365
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=yearly,
        )
        model.fit(prophet_train)
        future = pd.DataFrame({"ds": test.index})
        forecast = model.predict(future)
        predictions = pd.Series(forecast["yhat"].to_numpy(), index=test.index, name="Prophet")
        return predictions, None
    except Exception as exc:
        return None, f"Prophet model skipped because package is not installed or failed to run. Details: {exc}"


def train_xgboost_forecast(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.Series:
    """Fit XGBoost on engineered features and predict the test period."""
    feature_columns = get_feature_columns(train_features, target_column=target_column)
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(train_features[feature_columns], train_features[target_column])
    predictions = model.predict(test_features[feature_columns])
    return pd.Series(predictions, index=test_features.index, name="XGBoost")


def run_all_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    include_prophet: bool = True,
) -> tuple[dict[str, pd.Series], list[str]]:
    """Run available forecasting models and collect warnings for skipped models."""
    forecasts: dict[str, pd.Series] = {}
    messages: list[str] = []

    try:
        forecasts["ARIMA"] = train_arima_forecast(train, test)
    except Exception as exc:
        messages.append(f"ARIMA skipped because it failed to run. Details: {exc}")

    if include_prophet:
        prophet_forecast, message = train_prophet_forecast(train, test)
        if prophet_forecast is not None:
            forecasts["Prophet"] = prophet_forecast
        elif message:
            messages.append(message)

    try:
        forecasts["XGBoost"] = train_xgboost_forecast(train_features, test_features)
    except Exception as exc:
        messages.append(f"XGBoost skipped because it failed to run. Details: {exc}")

    return forecasts, messages

