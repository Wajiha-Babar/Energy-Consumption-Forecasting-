from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import FORECASTS_DIR, METRICS_DIR, TARGET_COLUMN
from src.data_loader import find_dataset, load_power_consumption_data
from src.evaluation import build_comparison_table
from src.feature_engineering import create_features
from src.modeling import run_all_models
from src.preprocessing import chronological_train_test_split, clean_missing_values, limit_recent_data, resample_power_data


st.set_page_config(
    page_title="Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
)


THEME_CSS = """
<style>
    .stApp {
        background: #071A13;
        color: #ECFDF5;
    }
    [data-testid="stSidebar"] {
        background: #0A2118;
        border-right: 1px solid rgba(52, 211, 153, 0.16);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #ECFDF5;
        letter-spacing: 0;
    }
    .subtitle {
        color: #A7F3D0;
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(145deg, #0F2A1F, #0B2018);
        border: 1px solid rgba(52, 211, 153, 0.18);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
        min-height: 118px;
    }
    .kpi-label {
        color: #A7F3D0;
        font-size: 0.82rem;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .kpi-value {
        color: #ECFDF5;
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    div[data-testid="stMetric"] {
        background: #0F2A1F;
        border-radius: 10px;
        padding: 0.85rem;
        border: 1px solid rgba(52, 211, 153, 0.18);
    }
    .stButton button {
        background: #10B981;
        color: #071A13;
        border: 0;
        font-weight: 700;
    }
    .stButton button:hover {
        background: #34D399;
        color: #071A13;
    }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


SAMPLE_OPTIONS = {
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 180 days": 180,
    "Full data if possible": None,
}
HORIZON_OPTIONS = {
    "24 hours": 24,
    "48 hours": 48,
    "7 days": 24 * 7,
    "14 days": 24 * 14,
}
FREQUENCY_MAP = {
    "Hourly": "h",
    "Daily": "D",
}


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return clean_missing_values(load_power_consumption_data())


@st.cache_data(show_spinner=False)
def prepare_data(frequency: str, sample_days: int | None) -> pd.DataFrame:
    raw = cached_dataset()
    resampled = resample_power_data(raw, frequency=frequency)
    if frequency == "D" and sample_days is not None:
        sample_days = max(sample_days, 365)
    return limit_recent_data(resampled, sample_days)


@st.cache_data(show_spinner=False)
def train_dashboard_models(data: pd.DataFrame, model_choice: str, horizon: int):
    if len(data) < 220:
        raise ValueError("Not enough records after filtering. Choose a larger data sample.")

    test_rows = min(horizon, max(24, int(len(data) * 0.2)))
    train = data.iloc[:-test_rows]
    test = data.iloc[-test_rows:]

    featured = create_features(data)
    train_features = featured.loc[featured.index.isin(train.index)]
    test_features = featured.loc[featured.index.isin(test.index)]

    include_prophet = model_choice in {"Prophet", "Compare All"}
    forecasts, messages = run_all_models(
        train=train,
        test=test,
        train_features=train_features,
        test_features=test_features,
        include_prophet=include_prophet,
    )

    if model_choice in {"ARIMA", "Prophet", "XGBoost"}:
        forecasts = {name: series for name, series in forecasts.items() if name == model_choice}

    if not forecasts:
        raise RuntimeError("No models produced forecasts. Review package installation and sample size.")

    actual = test[TARGET_COLUMN]
    metrics = build_comparison_table(forecasts, actual)
    forecast_table = pd.DataFrame({"Actual": actual})
    for name, forecast in forecasts.items():
        forecast_table[name] = forecast
    return metrics, forecast_table, messages


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#071A13",
        plot_bgcolor="#0F2A1F",
        font_color="#ECFDF5",
        margin=dict(l=25, r=25, t=55, b=25),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(167, 243, 208, 0.12)")
    fig.update_yaxes(gridcolor="rgba(167, 243, 208, 0.12)")
    return fig


st.title("Energy Consumption Forecasting Dashboard")
st.markdown(
    '<div class="subtitle">Short-term household electricity usage prediction using ARIMA, Prophet, and XGBoost</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    try:
        dataset_type, dataset_path = find_dataset()
        st.success(f"Dataset loaded from {dataset_type.upper()}: {dataset_path.name}")
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    frequency_label = st.selectbox("Resampling frequency", ["Hourly", "Daily"])
    model_choice = st.selectbox("Model", ["Compare All", "ARIMA", "Prophet", "XGBoost"])
    horizon_label = st.selectbox("Forecast horizon", list(HORIZON_OPTIONS.keys()))
    sample_label = st.selectbox("Data sample", list(SAMPLE_OPTIONS.keys()), index=1)

frequency = FREQUENCY_MAP[frequency_label]
horizon = HORIZON_OPTIONS[horizon_label]
sample_days = SAMPLE_OPTIONS[sample_label]
if frequency == "D":
    horizon = max(7, horizon // 24)

try:
    with st.spinner("Loading and preparing data..."):
        data = prepare_data(frequency, sample_days)
except Exception as exc:
    if "Invalid frequency" in str(exc):
        st.error("Invalid resampling frequency. Please choose Hourly or Daily from the sidebar.")
        st.stop()
    st.error(f"Unable to load dataset. {exc}")
    st.stop()

try:
    with st.spinner("Training selected model configuration..."):
        metrics_df, forecast_df, model_messages = train_dashboard_models(data, model_choice, horizon)
except Exception as exc:
    st.error(f"Modeling failed: {exc}")
    st.stop()

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
metrics_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
forecast_df.to_csv(FORECASTS_DIR / "forecast_results.csv", index=True)

for message in model_messages:
    st.warning(message)

best_model = metrics_df.iloc[0]["Model"] if not metrics_df.empty else "N/A"
missing_values = int(cached_dataset().isna().sum().sum())

cols = st.columns(6)
with cols[0]:
    kpi_card("Total Records", f"{len(data):,}")
with cols[1]:
    date_range = f"{data.index.min().date()} to {data.index.max().date()}"
    kpi_card("Date Range", date_range)
with cols[2]:
    kpi_card("Average Power Usage", f"{data[TARGET_COLUMN].mean():.3f} kW")
with cols[3]:
    kpi_card("Maximum Power Usage", f"{data[TARGET_COLUMN].max():.3f} kW")
with cols[4]:
    kpi_card("Missing Values", f"{missing_values:,}")
with cols[5]:
    kpi_card("Best Model Based on RMSE", str(best_model))

st.subheader("Energy Consumption Over Time")
trend_fig = px.line(data, y=TARGET_COLUMN, title="Resampled Global Active Power")
trend_fig.update_traces(line_color="#34D399", line_width=1.5)
st.plotly_chart(apply_plot_theme(trend_fig), use_container_width=True)

left, right = st.columns(2)
with left:
    hourly = data.copy()
    hourly["hour"] = hourly.index.hour
    hour_avg = hourly.groupby("hour", as_index=False)[TARGET_COLUMN].mean()
    fig = px.bar(hour_avg, x="hour", y=TARGET_COLUMN, title="Average Usage by Hour of Day")
    fig.update_traces(marker_color="#10B981")
    st.plotly_chart(apply_plot_theme(fig), use_container_width=True)

with right:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = data.copy()
    by_day["day_of_week"] = by_day.index.dayofweek
    day_avg = by_day.groupby("day_of_week", as_index=False)[TARGET_COLUMN].mean()
    day_avg["day"] = day_avg["day_of_week"].map(dict(enumerate(day_names)))
    fig = px.bar(day_avg, x="day", y=TARGET_COLUMN, title="Average Usage by Day of Week")
    fig.update_traces(marker_color="#047857")
    st.plotly_chart(apply_plot_theme(fig), use_container_width=True)

st.subheader("Actual vs Forecasted Consumption")
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Actual"], name="Actual", line=dict(color="#ECFDF5", width=2)))
for column in forecast_df.columns:
    if column != "Actual":
        forecast_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[column], name=column, line=dict(width=2)))
forecast_fig.update_layout(title="Actual vs Forecasted Energy Usage", yaxis_title="Global Active Power")
st.plotly_chart(apply_plot_theme(forecast_fig), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Model Performance Comparison")
    melted = metrics_df.melt(id_vars="Model", value_vars=["MAE", "RMSE", "MAPE"], var_name="Metric", value_name="Value")
    fig = px.bar(melted, x="Model", y="Value", color="Metric", barmode="group", title="MAE, RMSE, and MAPE")
    st.plotly_chart(apply_plot_theme(fig), use_container_width=True)
    st.dataframe(metrics_df, use_container_width=True)

with right:
    st.subheader("Residual Distribution")
    selected_for_residuals = best_model if best_model in forecast_df.columns else next((c for c in forecast_df.columns if c != "Actual"), None)
    if selected_for_residuals:
        residuals = forecast_df["Actual"] - forecast_df[selected_for_residuals]
        fig = px.histogram(residuals, nbins=35, title=f"Residuals for {selected_for_residuals}")
        fig.update_traces(marker_color="#34D399")
        st.plotly_chart(apply_plot_theme(fig), use_container_width=True)
    else:
        st.info("No forecast available for residual analysis.")

st.caption(f"Metrics saved to {Path('outputs/metrics/model_comparison.csv')} and forecasts saved to {Path('outputs/forecasts/forecast_results.csv')}.")
