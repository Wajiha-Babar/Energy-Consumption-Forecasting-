# Energy Consumption Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Forecasting-10B981?style=for-the-badge)
![Time Series](https://img.shields.io/badge/Time%20Series-Forecasting-047857?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Completed-34D399?style=for-the-badge)

## Project Overview

This project forecasts short-term household electricity consumption using historical time-series data from the **Individual Household Electric Power Consumption Dataset**.

It includes a complete end-to-end data science workflow:

- Dataset loading and preprocessing
- Datetime parsing and resampling
- Exploratory data analysis
- Time-based feature engineering
- ARIMA, Prophet, and XGBoost forecasting
- Model evaluation using MAE, RMSE, and MAPE
- Interactive Streamlit dashboard with a premium emerald-green theme

The project is designed for academic submission, GitHub portfolio presentation, and practical time-series forecasting demonstration.

---

## Objective

The objective of this project is to forecast short-term household energy usage using historical electricity consumption patterns.

The project compares three different forecasting approaches:

| Model | Forecasting Approach |
|---|---|
| ARIMA | Traditional statistical time-series forecasting |
| Prophet | Trend and seasonality-based forecasting |
| XGBoost | Machine learning forecasting using engineered features |

---

## Dataset

### Dataset Name

**Individual Household Electric Power Consumption Dataset**

### Dataset Description

The dataset contains minute-level household electricity consumption measurements. It includes power usage, voltage, intensity, and sub-metering readings recorded over time.

### Main Columns

| Column | Description |
|---|---|
| `Date` | Date of measurement |
| `Time` | Time of measurement |
| `Global_active_power` | Total active power consumed by the household |
| `Global_reactive_power` | Total reactive power consumed |
| `Voltage` | Average voltage |
| `Global_intensity` | Current intensity |
| `Sub_metering_1` | Kitchen-related energy sub-metering |
| `Sub_metering_2` | Laundry-related energy sub-metering |
| `Sub_metering_3` | Water heater and air-conditioner energy sub-metering |

### Target Variable

```text
Global_active_power
```

This column is used as the main forecasting target.

---

## Project Workflow

```text
Raw Dataset
   ↓
Data Loading
   ↓
Datetime Parsing
   ↓
Missing Value Handling
   ↓
Hourly / Daily Resampling
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Chronological Train-Test Split
   ↓
Model Training
   ↓
Forecasting
   ↓
Model Evaluation
   ↓
Streamlit Dashboard
```

---

## Key Features

### Data Processing

- Automatic dataset detection from `.txt` or `.zip`
- Safe datetime parsing using `Date` and `Time`
- Missing value handling
- Numeric column conversion
- Hourly and daily resampling
- Chronological train/test splitting

### Feature Engineering

The project creates strong forecasting features such as:

| Feature Type | Examples |
|---|---|
| Time features | Hour, day, month, year, day of week, weekend flag |
| Lag features | Lag 1, lag 2, lag 3, lag 24, lag 48, lag 168 |
| Rolling features | Rolling mean and rolling standard deviation |
| Calendar features | Week of year and quarter |

### Model Evaluation

Each model is evaluated using:

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |

The best model is selected based on the lowest RMSE.

---

## Models Used

### 1. ARIMA

ARIMA is used as a statistical baseline model.

Default order:

```text
ARIMA(2, 1, 2)
```

ARIMA helps compare machine learning models against a classical time-series forecasting method.

### 2. Prophet

Prophet is used for trend and seasonality forecasting.

It is useful for data with:

- Daily seasonality
- Weekly seasonality
- Long-term trend behavior

If Prophet is not installed or fails to run, the project skips it gracefully and continues with ARIMA and XGBoost.

### 3. XGBoost

XGBoost is used as the main machine learning model.

It uses:

- Time-based features
- Lag features
- Rolling mean features
- Rolling standard deviation features

XGBoost usually performs strongly because it can learn non-linear relationships from engineered time-series features.

---

## Streamlit Dashboard

The project includes a professional interactive dashboard built with Streamlit.

### Dashboard Title

```text
Energy Consumption Forecasting Dashboard
```

### Dashboard Features

- Dataset loading status
- Resampling selector: Hourly or Daily
- Model selector: ARIMA, Prophet, XGBoost, or Compare All
- Forecast horizon selector
- Data sample selector
- KPI cards
- Energy consumption trend chart
- Average usage by hour of day
- Average usage by day of week
- Actual vs forecasted chart
- Model performance comparison
- Residual distribution chart

### Dashboard Theme

The dashboard uses a premium dark emerald visual style.

| UI Element | Color |
|---|---|
| Primary Emerald | `#10B981` |
| Dark Emerald | `#047857` |
| Deep Background | `#071A13` |
| Card Background | `#0F2A1F` |
| Light Text | `#ECFDF5` |
| Muted Text | `#A7F3D0` |
| Accent | `#34D399` |

---

## Repository Structure

```text
Energy Consumption Time Series Forecasting/
│
├── dataset/
│   ├── individual+household+electric+power+consumption.zip
│   └── household_power_consumption.txt
│
├── notebooks/
│   └── energy_consumption_forecasting.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
│
├── models/
│   └── README.md
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── forecasts/
│
├── app.py
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation and Setup

### 1. Create Virtual Environment

```powershell
python -m venv .venv
```

### 2. Activate Virtual Environment

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Dataset Setup

Place the dataset inside the `dataset/` folder.

Accepted formats:

```text
dataset/household_power_consumption.txt
```

or:

```text
dataset/individual+household+electric+power+consumption.zip
```

The project automatically detects and loads the available file.

---

## How to Run the Notebook

Start Jupyter Notebook:

```powershell
jupyter notebook
```

Then open:

```text
notebooks/energy_consumption_forecasting.ipynb
```

The notebook includes:

- Problem statement and objective
- Dataset description
- Data loading
- Data cleaning
- Exploratory data analysis
- Feature engineering
- Model building
- Model evaluation
- Visualizations
- Final conclusion

---

## How to Run the Dashboard

Run the dashboard from the project root folder:

```powershell
streamlit run app.py
```

The dashboard will open at:

```text
http://localhost:8501
```

---

## Requirements

Main libraries used:

```text
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
statsmodels
prophet
xgboost
streamlit
jupyter
notebook
```

---

## Prophet Installation Note

Prophet may sometimes fail to install on Windows depending on the Python version and environment.

If Prophet fails, try:

```powershell
pip install prophet
```

If it still fails, the project can continue without Prophet. The application handles Prophet failure gracefully and still runs ARIMA and XGBoost.

---

## Model Evaluation Results

After running the notebook or dashboard, the model comparison file is generated at:

```text
outputs/metrics/model_comparison.csv
```

Expected format:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| ARIMA | To be calculated | To be calculated | To be calculated |
| Prophet | To be calculated | To be calculated | To be calculated |
| XGBoost | To be calculated | To be calculated | To be calculated |

---

## Output Files

```text
outputs/
│
├── metrics/
│   └── model_comparison.csv
│
├── forecasts/
│   └── forecast_results.csv
│
└── figures/
```

### Output Description

| File | Description |
|---|---|
| `model_comparison.csv` | Stores MAE, RMSE, and MAPE for each model |
| `forecast_results.csv` | Stores actual and predicted values |
| `figures/` | Stores exported visualizations |

---

## Visualizations

The project includes the following visualizations:

- Energy consumption over time
- Hourly consumption pattern
- Daily average consumption trend
- Monthly consumption trend
- Average consumption by hour of day
- Average consumption by day of week
- Weekend vs weekday consumption
- Actual vs forecasted energy usage
- Model performance comparison
- Residual distribution

---

## Key Insights

The analysis helps identify important electricity consumption patterns:

- Household energy usage follows strong hourly and daily patterns.
- Weekend and weekday behavior can affect electricity demand.
- Lag features improve machine learning forecasting performance.
- Rolling averages capture recent consumption trends.
- XGBoost often performs strongly because it uses engineered features.
- ARIMA is useful as a statistical baseline.
- Prophet is useful for identifying trend and seasonality.

---

## Troubleshooting

### Invalid Frequency Error

If pandas gives this error:

```text
Invalid frequency: H. Did you mean h?
```

Use lowercase hourly frequency:

```python
"h"
```

instead of:

```python
"H"
```

### Dataset Not Found

Make sure the dataset is inside:

```text
dataset/
```

Accepted filenames:

```text
household_power_consumption.txt
individual+household+electric+power+consumption.zip
```

### Dashboard is Slow

Use a smaller sample from the sidebar:

- Last 30 days
- Last 90 days
- Last 180 days

Avoid running all models on the full minute-level dataset at once.

---

## Future Improvements

Possible future improvements:

- Add LSTM or GRU forecasting models
- Add weather-based external features
- Add holiday indicators
- Add real-time forecasting pipeline
- Deploy the dashboard on Streamlit Cloud
- Add automated model retraining
- Add anomaly detection
- Compare multiple households or buildings

---

## Final Conclusion

This project demonstrates a complete time-series forecasting pipeline for household electricity consumption.

It shows how raw energy data can be cleaned, resampled, analyzed, modeled, evaluated, and visualized through an interactive dashboard.

ARIMA provides a reliable statistical baseline. Prophet captures trend and seasonality. XGBoost performs strongly with lag and rolling features, making it a competitive machine learning approach for short-term electricity forecasting.

Overall, this project combines time-series analysis, machine learning, model evaluation, and dashboard development into one professional energy analytics solution.

---

## Author

**Omkar**

Project: Energy Consumption Time Series Forecasting  
Domain: Time-Series Forecasting and Energy Analytics  
Dashboard: Streamlit  
Main ML Model: XGBoost
