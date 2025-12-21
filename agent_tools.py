# agent_tools.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# METRICS (Benchmark only)
# --------------------------------------------------

def safe_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denom = np.where(y_true == 0, 1e-8, y_true)
    mape = (np.abs((y_true - y_pred) / denom).mean()) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


# --------------------------------------------------
# GROWTH & TREND
# --------------------------------------------------

def compute_growth_against_actuals(actual_df, forecast_df):
    """
    % difference between recent actual demand and forecasted demand
    """
    if actual_df is None or forecast_df is None:
        return 0.0

    if actual_df.empty or forecast_df.empty:
        return 0.0

    recent_actual = actual_df["y"].tail(7).mean()
    forecast_avg = forecast_df["yhat"].mean()

    if recent_actual == 0:
        return 0.0

    return round(((forecast_avg - recent_actual) / recent_actual) * 100, 2)


def compute_trend_direction(actual_df, window=14):
    if actual_df is None or len(actual_df) < window * 2:
        return None, "Insufficient data"

    recent = actual_df["y"].tail(window).mean()
    previous = actual_df["y"].iloc[-2*window:-window].mean()

    if previous == 0:
        return None, "Previous demand is zero"

    pct = ((recent - previous) / previous) * 100

    if pct > 5:
        return "UP", f"Demand increased by {round(pct,2)}%"
    elif pct < -5:
        return "DOWN", f"Demand decreased by {round(abs(pct),2)}%"
    else:
        return "FLAT", "Demand stable"


# --------------------------------------------------
# BUSINESS DECISIONS
# --------------------------------------------------

def compute_business_status(growth_pct):
    if growth_pct is None:
        return "UNKNOWN"
    if growth_pct > 10:
        return "DEMAND RISING"
    elif growth_pct < -10:
        return "DEMAND DECLINING"
    return "STABLE DEMAND"


def compute_action_from_status(status):
    if status == "DEMAND RISING":
        return "Increase inventory"
    elif status == "DEMAND DECLINING":
        return "Reduce inventory exposure"
    elif status == "STABLE DEMAND":
        return "Maintain current inventory"
    return "Review data"


# --------------------------------------------------
# FORECAST EXPLAINABILITY
# --------------------------------------------------

def explain_prophet_forecast(forecast_df):
    explanations = []

    if forecast_df is None or forecast_df.empty:
        return ["No forecast available"]

    start = forecast_df["yhat"].iloc[0]
    end = forecast_df["yhat"].iloc[-1]

    if end > start:
        explanations.append("📈 Long-term upward demand trend detected")
    else:
        explanations.append("📉 Long-term downward demand trend detected")

    band = (forecast_df["yhat_upper"] - forecast_df["yhat_lower"]).mean()
    avg = forecast_df["yhat"].mean()

    if band < avg * 0.3:
        explanations.append("✅ Forecast uncertainty is low")
    else:
        explanations.append("⚠️ Forecast uncertainty is high")

    explanations.append("🗓️ Seasonality captured using Prophet")

    return explanations


def compute_risk_label(forecast_df):
    if forecast_df is None or forecast_df.empty:
        return "UNKNOWN"

    band = (forecast_df["yhat_upper"] - forecast_df["yhat_lower"]).mean()
    avg = forecast_df["yhat"].mean()

    if band < avg * 0.25:
        return "LOW"
    elif band < avg * 0.5:
        return "MEDIUM"
    return "HIGH"
