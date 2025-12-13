import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from datetime import timedelta
from prophet import Prophet
from agent import generate_insights, chat_with_ai

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def is_agent_question(text: str) -> bool:
    decision_keywords = [
        "compare", "pick", "choose", "best", "decide",
        "recommend", "analyze", "forecast", "model",
        "risk", "action", "growth",
        "trend", "trending", "up", "down"
    ]
    text = text.lower().strip()
    return any(word in text for word in decision_keywords)



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

def to_pydt(x):
    if pd.isna(x): return pd.NaT
    if hasattr(x, "to_pydatetime"): return x.to_pydatetime()
    return x

def ensure_ds_py_datetime(df, col='ds'):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[col] = df[col].apply(lambda x: to_pydt(x))
    return df

def make_display_dates(series):
    s = pd.to_datetime(series, errors='coerce')
    return s.dt.strftime("%Y-%m-%d").fillna("")

def add_lag_features(df, lags=[1,7,14], roll_windows=[7,14]):
    df = df.sort_values("ds").reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(window=w, min_periods=1).mean()
    df.fillna(0, inplace=True)
    return df

def finalize_forecast(df_out):
    df = df_out.copy()
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = ensure_ds_py_datetime(df, 'ds')
    for c in ['yhat','yhat_lower','yhat_upper']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').round().fillna(0).astype(int)
    return df

st.set_page_config(layout="wide", page_title="Datalytics AI")

# ====== Hide only Streamlit menu & footer, keep sidebar header =====
hide_st_style = """
    <style>
    .st-emotion-cache-99anic:hover:enabled{
    background-color:#a100ff !important;
    }
    .st-emotion-cache-zh2fnc,.st-emotion-cache-zh2fnc{
    width:70%;
    margin:auto;
    }
    .st-emotion-cache-1vo6xi6{
    width:70%;
    margin:auto;
    }
    .st-emotion-cache-7czcpc > img{
    margin-top:25px !important;
    }
    .st-emotion-cache-uwwqev{
    margin-bottom:55px;
    }
    .st-emotion-cache-1j22a0y{
    margin-top:115px !important;
    }
    .st-emotion-cache-1ffuo7c{
    width:5% !important;
    }
    .st-emotion-cache-scp8yw{
    display:none !important
    }
    .st-emotion-cache-zy6yx3{
    padding-top :1em !important;
    padding-bottom: 5em !important;
    }
    .st-dl{
    background-color:#a100ff !important;
    }
    .st-el {
    background-color: #a100ff !important;
    }
    #MainMenu {visibility: hidden;}   /* Hides the top hamburger menu */
    footer {visibility: hidden;}     /* Hides the footer */
    /* header {visibility: hidden;} */ /* Don't hide header to keep sidebar title visible */
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
st.sidebar.markdown("## ⚙️ Settings")
st.image("logo.png", width=220)


uploaded = st.file_uploader("Upload CSV", type="csv")
if not uploaded:
    st.info("Data Format: date, store_id, SKU_ID, units_sold, inventory_on_hand, price, promotion_flag")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).copy()
df['date'] = df['date'].dt.tz_localize(None)

store_list = sorted(df['store_id'].dropna().unique().tolist())
selected_store = st.selectbox("Select store", store_list)
store_df = df[df['store_id'] == selected_store].copy()

sku_list = sorted(store_df['SKU_ID'].dropna().unique().tolist())
selected_sku = st.selectbox("Select SKU", sku_list)
sku_df = store_df[store_df['SKU_ID'] == selected_sku].copy()

if len(sku_df) < 30:
    st.warning("Not enough rows")
    st.stop()

ts = sku_df[['date','units_sold','inventory_on_hand','price','promotion_flag']].rename(
    columns={'date':'ds','units_sold':'y'}).sort_values('ds').reset_index(drop=True)

ts['price'] = pd.to_numeric(ts['price'], errors='coerce')
ts['promotion_flag'] = pd.to_numeric(ts['promotion_flag'], errors='coerce')

freq = st.radio("Frequency", ("Daily","Weekly"))
if freq == "Weekly":
    ts = ts.set_index('ds').resample('W').agg({
        'y':'sum',
        'inventory_on_hand':'last',
        'price':'mean',
        'promotion_flag':'max'
    }).reset_index()

ts['price'] = ts['price'].fillna(method='ffill').fillna(method='bfill').fillna(0)
ts['promotion_flag'] = ts['promotion_flag'].fillna(0).astype(int)

forecast_horizon = st.sidebar.number_input(
    "Days",
    min_value=7,
    max_value=30,
    value=30,
    step=1
)

add_price_reg = st.sidebar.checkbox("Price", True)
add_prom_reg = st.sidebar.checkbox("Promotion", True)
us_prophet = st.sidebar.checkbox("Use Prophet", True)
us_xgb = st.sidebar.checkbox("Use XGBoost", True if XGB_AVAILABLE else False)

nts = ensure_ds_py_datetime(ts, 'ds').dropna(subset=['ds']).reset_index(drop=True)
ts = nts.copy()

val_days = min(forecast_horizon, max(1, int(len(ts)*0.2)))
train_df = ts.iloc[:-val_days].copy()
val_df = ts.iloc[-val_days:].copy()

results = {}
metrics = {}
model_objects = {}

if us_prophet:
    try:
        m_val = Prophet(interval_width=0.95)
        if add_price_reg and 'price' in ts.columns: m_val.add_regressor('price')
        if add_prom_reg and 'promotion_flag' in ts.columns: m_val.add_regressor('promotion_flag')

        train_prop = train_df[['ds','y']].copy()
        if add_price_reg:
            train_prop['price'] = train_df['price'].fillna(method='ffill').fillna(method='bfill').fillna(0)
            train_prop['price'] = pd.to_numeric(train_prop['price'], errors='coerce').fillna(0)
        if add_prom_reg:
            train_prop['promotion_flag'] = train_df['promotion_flag'].fillna(0)
            train_prop['promotion_flag'] = pd.to_numeric(train_prop['promotion_flag'], errors='coerce').fillna(0).astype(int)

        m_val.fit(train_prop)

        step = 'D' if freq=="Daily" else 'W'
        val_start = train_df['ds'].max() + (pd.Timedelta(days=1) if step=='D' else pd.Timedelta(weeks=1))
        val_dates = pd.date_range(start=val_start, periods=len(val_df), freq=step)
        future_val = pd.DataFrame({'ds': val_dates})

        if add_price_reg:
            price_s = ts.set_index('ds')['price']
            future_val['price'] = price_s.reindex(val_dates).fillna(method='ffill').fillna(method='bfill').fillna(0)
            future_val['price'] = pd.to_numeric(future_val['price'], errors='coerce').fillna(0)
        if add_prom_reg:
            promo_s = ts.set_index('ds')['promotion_flag']
            future_val['promotion_flag'] = promo_s.reindex(val_dates).fillna(0)
            future_val['promotion_flag'] = pd.to_numeric(future_val['promotion_flag'], errors='coerce').fillna(0).astype(int)

        future_val = ensure_ds_py_datetime(future_val, 'ds')
        fc_val = m_val.predict(future_val)
        fc_val = fc_val.set_index('ds')[['yhat']].rename(columns={'yhat':'yhat_prop'})

        joined = val_df.set_index('ds').join(fc_val, how='inner').dropna()
        if len(joined)>0:
            metrics['Prophet'] = safe_metrics(joined['y'], joined['yhat_prop'])
        else:
            metrics['Prophet'] = {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}

        m_full = Prophet(interval_width=0.95)
        if add_price_reg: m_full.add_regressor('price')
        if add_prom_reg: m_full.add_regressor('promotion_flag')

        full_train = ts[['ds','y']].copy()
        if add_price_reg:
            full_train['price'] = ts['price'].fillna(method='ffill').fillna(method='bfill').fillna(0)
            full_train['price'] = pd.to_numeric(full_train['price'], errors='coerce').fillna(0)
        if add_prom_reg:
            full_train['promotion_flag'] = ts['promotion_flag'].fillna(0)
            full_train['promotion_flag'] = pd.to_numeric(full_train['promotion_flag'], errors='coerce').fillna(0).astype(int)

        m_full.fit(full_train)

        last_date = ts['ds'].max()
        future_dates = pd.date_range(start=last_date + (pd.Timedelta(days=1) if step=='D' else pd.Timedelta(weeks=1)),
                                     periods=forecast_horizon,
                                     freq=step)
        future_full = pd.DataFrame({'ds': future_dates})

        if add_price_reg:
            price_s = ts.set_index('ds')['price']
            future_full['price'] = price_s.reindex(future_dates).fillna(method='ffill').fillna(method='bfill').fillna(0)
            future_full['price'] = pd.to_numeric(future_full['price'], errors='coerce').fillna(0)
        if add_prom_reg:
            promo_s = ts.set_index('ds')['promotion_flag']
            future_full['promotion_flag'] = promo_s.reindex(future_dates).fillna(0)
            future_full['promotion_flag'] = pd.to_numeric(future_full['promotion_flag'], errors='coerce').fillna(0).astype(int)

        future_full = ensure_ds_py_datetime(future_full, 'ds')
        forecast_prop = m_full.predict(future_full)
        forecast_prop = forecast_prop[['ds','yhat','yhat_lower','yhat_upper']]

        results['Prophet'] = finalize_forecast(forecast_prop)
        model_objects['Prophet'] = m_full

    except Exception as e:
        st.error(f"Prophet error: {e}")

if us_xgb:
    if not XGB_AVAILABLE:
        st.warning("XGBoost not installed")
    else:
        try:
            xdf = ts[['ds','y','price','promotion_flag']].copy()
            xdf['ds'] = pd.to_datetime(xdf['ds'])
            xdf['price'] = pd.to_numeric(xdf['price'], errors='coerce').fillna(0)
            xdf['promotion_flag'] = pd.to_numeric(xdf['promotion_flag'], errors='coerce').fillna(0)
            xdf['promotion_flag'] = xdf['promotion_flag'].astype(int)
            xdf = add_lag_features(xdf)

            feat_cols = [c for c in xdf.columns if c not in ['ds','y']]
            X_train = xdf[feat_cols].values
            y_train = xdf['y'].values
            model_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=5, objective='reg:squarederror', verbosity=0)
            model_xgb.fit(X_train, y_train)

            step = 'D' if freq=="Daily" else 'W'
            last_date = xdf['ds'].max()
            future_dates = pd.date_range(start=last_date + (pd.Timedelta(days=1) if step=='D' else pd.Timedelta(weeks=1)),
                                         periods=forecast_horizon,
                                         freq=step)

            future_df = pd.DataFrame({'ds': future_dates})
            if 'price' in ts.columns:
                price_s = ts.set_index('ds')['price']
                future_df['price'] = price_s.reindex(future_dates).fillna(method='ffill').fillna(method='bfill').fillna(0)
            else:
                future_df['price'] = 0
            future_df['price'] = pd.to_numeric(future_df['price'], errors='coerce').fillna(0)

            if 'promotion_flag' in ts.columns:
                promo_s = ts.set_index('ds')['promotion_flag']
                future_df['promotion_flag'] = promo_s.reindex(future_dates).fillna(0)
            else:
                future_df['promotion_flag'] = 0
            future_df['promotion_flag'] = pd.to_numeric(future_df['promotion_flag'], errors='coerce').fillna(0).astype(int)

            combined = pd.concat([xdf[['ds','y','price','promotion_flag']], future_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['ds']).sort_values('ds').reset_index(drop=True)
            combined = add_lag_features(combined)

            fc_rows = combined[combined['ds'].isin(future_dates)].copy()
            X_fc = fc_rows[feat_cols].values
            y_pred = model_xgb.predict(X_fc).round().astype(int)

            xgb_out = pd.DataFrame({
                'ds': future_dates,
                'yhat': y_pred,
                'yhat_lower': (y_pred - int(np.std(y_train))).astype(int),
                'yhat_upper': (y_pred + int(np.std(y_train))).astype(int)
            })

            combined_all = pd.concat([xdf[['ds','y','price','promotion_flag']], val_df[['ds','y','price','promotion_flag']]], ignore_index=True)
            combined_all = combined_all.drop_duplicates(subset=['ds']).sort_values('ds').reset_index(drop=True)
            combined_all = add_lag_features(combined_all)
            val_feats = combined_all[combined_all['ds'].isin(val_df['ds'])]
            if len(val_feats)>0:
                val_preds = model_xgb.predict(val_feats[feat_cols].values).round().astype(int)
                metrics['XGBoost'] = safe_metrics(val_feats['y'].values, val_preds)
            else:
                metrics['XGBoost'] = {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}

            results['XGBoost'] = finalize_forecast(xgb_out)
            model_objects['XGBoost'] = model_xgb

        except Exception as e:
            st.error(f"XGBoost error: {e}")

st.header("Forecast Results")

metrics_df = pd.DataFrame.from_dict(metrics, orient='index').reset_index().rename(columns={'index':'model'})
if not metrics_df.empty:
    st.dataframe(metrics_df.set_index('model'))

for name, fc in results.items():
    if name == "XGBoost":
        display_name = "Dataset B"
    else:
        display_name = "Dataset A"
    st.subheader(display_name)
    disp = fc.copy()
    disp['ds'] = make_display_dates(disp['ds'])
    st.dataframe(disp.head(30))
    buf = io.StringIO()
    disp.to_csv(buf, index=False)
    st.download_button(f" ➜] Download", buf.getvalue(), f"{selected_store}_{selected_sku}_{display_name}.csv", "text/csv")

# ====== AI Insights Section ======
# if metrics and results:
#     st.subheader("🤖 AI Analytics")

#     try:
#         # Convert metrics + results to readable text
#         metrics_text = "\n".join([f"{m}: {v}" for m, v in metrics.items()])
#         results_text = "\n".join([f"{name}: {df.head().to_dict()}" for name, df in results.items()])

#         insights = generate_insights(metrics_text, results_text, selected_sku, selected_store)

#         st.success(insights)

#     except Exception as e:
#         st.error(f"AI Insights Error: {e}")


# # ----------------------------------------
# # 💬 CHAT SYSTEM (GROQ AI)
# # ----------------------------------------
# st.subheader("💬 Chat with AI Assistant")

# chat_context = f"""
# Store: {selected_store}
# SKU: {selected_sku}

# Metrics:
# {metrics}

# Insights:
# {insights if 'insights' in locals() else ""}

# Available Models:
# {list(results.keys())}
# """

# user_query = st.text_input("Ask anything about your data, forecast, trends, risks, promotions, or pricing:")

# if user_query:
#     ai_answer = chat_with_ai(user_query, context=chat_context)
#     st.markdown("### 🤖 Reply")
#     st.write(ai_answer)

# --- paste near your AI Insights / Chat section in dashboard.py ---
# from agentic_engine import run_agentic_task
# import json, pandas as pd

from agentic_engine import run_agentic_task

st.subheader("🧠 Autonomous Analysis")

agent_task_default = (
    "Compare Prophet and XGBoost, pick the best model, "
    "compute demand growth vs last actuals, and give 6 action items."
)

task_input = st.text_input("Ask something:", value=agent_task_default)
run_agent_button = st.button("Run")

if run_agent_button:
    # -------------------------
    # 🤖 AGENT MODE
    # -------------------------
    if is_agent_question(task_input):
        with st.spinner("Agent is autonomously analyzing..."):
            agent_output = run_agentic_task(
                prompt=task_input,
                data=sku_df,
                existing_forecasts=results.get(
                    min(metrics, key=lambda m: metrics[m]["MAPE"])
                    if metrics else "XGBoost"
                ),
                metrics=metrics
            )


            st.subheader("🤖 Agent Decision")
            st.json(agent_output)

            st.subheader("🧠 Explanation")
            st.write(agent_output.get("llm_explanation", "No explanation"))

    # -------------------------
    # 💬 CHAT MODE
    # -------------------------
    else:
        with st.spinner("AI Assistant replying..."):
            reply = chat_with_ai(
                user_message=task_input,
                context=f"""
                Store: {selected_store}
                SKU: {selected_sku}
                Metrics: {metrics}
                Available models: {list(results.keys())}
                """
            )

            st.subheader("💬 Assistant")
            st.write(reply)


if results:
    plot_choice = st.selectbox("Plot model", list(results.keys()))
    pr = results[plot_choice].copy()
    actual = ts[['ds','y']]
    plot_df = pd.merge(actual, pr[['ds','yhat']], on='ds', how='outer').sort_values('ds')
    plot_df = plot_df.dropna(subset=['ds'])

    # ===== New Plots: Pie + Bar side by side =====
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Bar chart: Actual vs Forecast for last 10 days
    # Bar chart: Actual vs Forecast for last 10 days
    last_n = 10
    plot_df_bar = plot_df.tail(last_n)
    x = np.arange(len(plot_df_bar))

    width = 0.4
    axes[0].bar(x - width/2, plot_df_bar['y'], width=width, label='Actual', alpha=0.7)
    axes[0].bar(x + width/2, plot_df_bar['yhat'], width=width, label='Forecast', alpha=0.7)
    axes[0].set_title(f'Actual vs Forecast (Last {last_n} Days)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df_bar['ds'].dt.strftime('%Y-%m-%d'), rotation=45)
    axes[0].legend()


    # Pie chart: Total Actual vs Forecast
    total_actual = plot_df['y'].sum()
    total_forecast = plot_df['yhat'].sum()
    axes[1].pie([total_actual, total_forecast], labels=['Actual','Forecast'], autopct='%1.1f%%', colors=['#1f77b4','#ff7f0e'])
    axes[1].set_title('Total Actual vs Forecast')

    plt.tight_layout()
    st.pyplot(fig)


st.markdown(
    """
    <style>
    .footer {
        
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f5f5f5;
        color: #555;
        text-align: center;
        padding: 8px 0;
        font-size: 14px;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
    }
    .footer a {
        color: #a100ff;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Developed by <a style="color:black" href="#">Shaishav Anand</a> @ <a style="color: #a100ff" href="https://www.accenture.com/in-en">Accenture</a> <span style="color:black">2025</span>
    </div>
    """,
    unsafe_allow_html=True
)



