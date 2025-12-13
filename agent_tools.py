# agent_tools.py
import pandas as pd
import numpy as np
from prophet import Prophet
import json
import io

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error

def safe_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
    denom = np.where(y_true == 0, 1e-8, y_true)
    mape = (np.abs((y_true - y_pred) / denom).mean()) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}

def _to_df(serialized):
    """Expect JSON serialized with orient='split'"""
    if isinstance(serialized, str):
        return pd.read_json(serialized, orient='split')
    if isinstance(serialized, dict):
        return pd.read_json(json.dumps(serialized), orient='split')
    return serialized

def compare_models(metrics: dict):
    """
    Select best model based on lowest MAPE, then RMSE, then MAE
    """

    if not metrics:
        return None, "no metrics provided"

    # Filter valid models
    valid = {
        model: vals
        for model, vals in metrics.items()
        if vals and vals.get("MAPE") is not None
    }

    if not valid:
        return None, "no valid metrics"

    # Sort by MAPE → RMSE → MAE
    sorted_models = sorted(
        valid.items(),
        key=lambda x: (
            x[1].get("MAPE", float("inf")),
            x[1].get("RMSE", float("inf")),
            x[1].get("MAE", float("inf"))
        )
    )

    best_model = sorted_models[0][0]
    reason = "lowest MAPE (primary), RMSE & MAE (secondary)"

    return best_model, reason


def compute_growth_from_forecast(forecast_serialized):
    """
    forecast_serialized: serialized df (orient='split') with 'ds' and 'yhat'
    returns dict: initial, final, pct_growth
    """
    df = _to_df(forecast_serialized)
    if df.empty or 'yhat' not in df.columns:
        return {"error": "invalid forecast data"}
    df = df.sort_values('ds').reset_index(drop=True)
    initial = float(df['yhat'].iloc[0])
    final = float(df['yhat'].iloc[-1])
    if initial == 0:
        pct = None
    else:
        pct = ((final - initial) / initial) * 100
    return {"initial": initial, "final": final, "pct_growth": pct}

def compute_trend_direction(actual_df, window=14):
    """
    Compare recent average vs previous average
    """
    if actual_df is None or len(actual_df) < window * 2:
        return None, "Insufficient data for trend analysis"

    recent = actual_df['y'].tail(window).mean()
    previous = actual_df['y'].iloc[-2*window:-window].mean()

    if previous == 0:
        return None, "Previous demand is zero"

    change_pct = ((recent - previous) / previous) * 100

    if change_pct > 5:
        return "UP", f"Demand increased by {round(change_pct,2)}%"
    elif change_pct < -5:
        return "DOWN", f"Demand decreased by {round(abs(change_pct),2)}%"
    else:
        return "FLAT", "Demand is stable"


def compute_growth_against_actuals(actual_df, forecast_df):
    """
    Compute % growth of forecast vs recent actuals.

    actual_df: DataFrame with columns ['ds', 'y']
    forecast_df: DataFrame with column ['yhat']
    """

    if actual_df is None or forecast_df is None:
        return 0.0

    if actual_df.empty or forecast_df.empty:
        return 0.0

    # Use last 7 actual points (or fewer if not available)
    recent_actuals = actual_df['y'].tail(7).mean()

    # Use full forecast average
    forecast_avg = forecast_df['yhat'].mean()

    if recent_actuals == 0:
        return 0.0

    growth_pct = ((forecast_avg - recent_actuals) / recent_actuals) * 100
    return round(float(growth_pct), 2)


def run_prophet_on_serialized(df_serialized, forecast_horizon=30, add_price_reg=False, add_prom_reg=False, freq='D'):
    """
    Runs Prophet on serialized dataframe that has 'ds','y', and optional regressors
    Returns serialized forecast (orient='split') and model metrics stub.
    """
    df = _to_df(df_serialized)
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    train = df.copy()
    m = Prophet(interval_width=0.95)
    if add_price_reg and 'price' in df.columns: m.add_regressor('price')
    if add_prom_reg and 'promotion_flag' in df.columns: m.add_regressor('promotion_flag')
    # fit
    m.fit(train)
    step = 'D' if freq == 'D' else 'W'
    last_date = df['ds'].max()
    future_index = pd.date_range(start=last_date + (pd.Timedelta(days=1) if step == 'D' else pd.Timedelta(weeks=1)),
                                 periods=forecast_horizon, freq=step)
    future = pd.DataFrame({'ds': future_index})
    # append regressors if present in series
    if add_price_reg and 'price' in df.columns:
        price_s = df.set_index('ds')['price']
        future['price'] = price_s.reindex(future_index).fillna(method='ffill').fillna(method='bfill').fillna(0)
    if add_prom_reg and 'promotion_flag' in df.columns:
        promo_s = df.set_index('ds')['promotion_flag']
        future['promotion_flag'] = promo_s.reindex(future_index).fillna(0)
    forecast = m.predict(future)
    # keep ds, yhat, yhat_lower, yhat_upper
    forecast_out = forecast[['ds','yhat','yhat_lower','yhat_upper']].copy()
    return forecast_out.to_json(orient='split')

def run_xgboost_on_serialized(df_serialized, forecast_horizon=30, freq='D'):
    """
    Trains simple XGBoost regressor with lag features and predicts future period.
    Returns serialized forecast
    """
    if not XGB_AVAILABLE:
        return {"error": "XGBoost not available on this environment"}
    df = _to_df(df_serialized)
    df = df.sort_values('ds').reset_index(drop=True)
    # create features - simple lags + rolling
    def add_lag_features(df, lags=[1,7,14], roll_windows=[7,14]):
        df = df.sort_values("ds").reset_index(drop=True)
        for lag in lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        for w in roll_windows:
            df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(window=w, min_periods=1).mean()
        df.fillna(0, inplace=True)
        return df
    xdf = df[['ds','y']].copy()
    if 'price' in df.columns:
        xdf['price'] = df['price'].fillna(method='ffill').fillna(0)
    else:
        xdf['price'] = 0
    if 'promotion_flag' in df.columns:
        xdf['promotion_flag'] = df['promotion_flag'].fillna(0).astype(int)
    else:
        xdf['promotion_flag'] = 0
    xdf = add_lag_features(xdf)
    feat_cols = [c for c in xdf.columns if c not in ['ds','y']]
    X_train = xdf[feat_cols].values
    y_train = xdf['y'].values
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, objective='reg:squarederror', verbosity=0)
    model.fit(X_train, y_train)
    step = 'D' if freq == 'D' else 'W'
    last_date = xdf['ds'].max()
    future_dates = pd.date_range(start=last_date + (pd.Timedelta(days=1) if step=='D' else pd.Timedelta(weeks=1)),
                                 periods=forecast_horizon, freq=step)
    future_df = pd.DataFrame({'ds': future_dates})
    # build combined for lag features generation
    combined = pd.concat([xdf[['ds','y','price','promotion_flag']], future_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['ds']).sort_values('ds').reset_index(drop=True)
    combined = add_lag_features(combined)
    fc_rows = combined[combined['ds'].isin(future_dates)].copy()
    X_fc = fc_rows[feat_cols].values
    y_pred = model.predict(X_fc).round().astype(int)
    # create output
    std = int(max(1, np.std(y_train))) if len(y_train)>0 else 1
    out = pd.DataFrame({
        'ds': future_dates,
        'yhat': y_pred,
        'yhat_lower': (y_pred - std).astype(int),
        'yhat_upper': (y_pred + std).astype(int)
    })
    return out.to_json(orient='split')


def compute_trend_direction(actual_df, window=14):
    """
    Determine demand trend using recent vs previous window averages
    """
    if actual_df is None or len(actual_df) < window * 2:
        return None, "Insufficient data for trend analysis"

    actual_df = actual_df.sort_values("ds")

    recent = actual_df["y"].tail(window).mean()
    previous = actual_df["y"].iloc[-2*window:-window].mean()

    if previous == 0:
        return None, "Previous demand is zero"

    change_pct = ((recent - previous) / previous) * 100

    if change_pct > 5:
        return "UP", f"Demand increased by {round(change_pct, 2)}%"
    elif change_pct < -5:
        return "DOWN", f"Demand decreased by {round(abs(change_pct), 2)}%"
    else:
        return "FLAT", "Demand is stable"
