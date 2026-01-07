ture# =========================================================
# Datalytics AI — Store-Level Autonomous Decision System
# =========================================================

import streamlit as st
import pandas as pd
from prophet import Prophet
from agentic_engine import run_portfolio_agent

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(layout="wide", page_title="Datalytics AI")

st.markdown(
    """
<style>

/* ---- Global font polish ---- */
html, body, [class*="css"]  {
    font-family: "Inter", "Segoe UI", sans-serif;
}

/* ---- Section headers ---- */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* ---- Brand accent ---- */
.brand {
    color: #a100ff;
}

/* ---- Card layout ---- */
.metric-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
    border-left: 6px solid #a100ff;
}

/* ---- Insight box ---- */
.insight-box {
    background: linear-gradient(
        135deg,
        rgba(161,0,255,0.08),
        rgba(161,0,255,0.02)
    );
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(161,0,255,0.25);
    margin-top: 10px;
}

/* ---- Decision banner ---- */
.decision-banner {
    background: #a100ff;
    color: white;
    border-radius: 18px;
    padding: 26px;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 10px 30px rgba(161,0,255,0.35);
    margin:16px auto;
}

/* ---- Buttons ---- */
.stButton > button {
    background-color: #a100ff;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.4em;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background-color: #8700d6;
    color: white;
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True
)


st.sidebar.markdown("## ⚙️ Settings")
st.image("image.png", width=200)

# --------------------------------------------------
# FORECAST ADJUSTMENTS (OPTIONAL)
# --------------------------------------------------

st.sidebar.markdown("### 🔧 Forecast Adjustments")

use_price = st.sidebar.checkbox(
    "Consider Price Impact",
    value=False,
    help="Enable if price changes strongly affect demand"
)

use_promotion = st.sidebar.checkbox(
    "Consider Promotion Impact",
    value=False,
    help="Enable if promotions significantly influence sales"
)

# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------

uploaded = st.file_uploader("Upload CSV", type="csv")
if not uploaded:
    st.info(
        "Required columns:\n"
        "date, store_id, SKU_ID, units_sold, price, promotion_flag"
    )
    st.stop()

df = pd.read_csv(uploaded)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

store = st.selectbox("Select Store", sorted(df["store_id"].unique()))
store_df = df[df["store_id"] == store]

# --------------------------------------------------
# EXPLANATION HEADER
# --------------------------------------------------

st.markdown(
    """
### 🧠 What this analysis does
This view analyzes **all products (SKUs)** in the selected store and converts
individual demand forecasts into **one clear business decision**
for inventory planning.
"""
)

# --------------------------------------------------
# FORECAST ALL SKUs (PROPHET)
# --------------------------------------------------

forecasts_by_sku = {}

with st.spinner("Analyzing demand patterns across all products..."):
    for sku, sku_df in store_df.groupby("SKU_ID"):
        if len(sku_df) < 30:
            continue

        # -----------------------------
        # Build training data
        # -----------------------------

        ts = sku_df.rename(
            columns={"date": "ds", "units_sold": "y"}
        )[["ds", "y"]].copy()

        if use_price and "price" in sku_df.columns:
            ts["price"] = pd.to_numeric(
                sku_df["price"], errors="coerce"
            ).fillna(method="ffill").fillna(0)

        if use_promotion and "promotion_flag" in sku_df.columns:
            ts["promotion_flag"] = pd.to_numeric(
                sku_df["promotion_flag"], errors="coerce"
            ).fillna(0).astype(int)

        # -----------------------------
        # Initialize Prophet
        # -----------------------------

        m = Prophet(interval_width=0.95)

        if use_price:
            m.add_regressor("price")

        if use_promotion:
            m.add_regressor("promotion_flag")

        train_cols = ["ds", "y"]
        if use_price:
            train_cols.append("price")
        if use_promotion:
            train_cols.append("promotion_flag")

        m.fit(ts[train_cols])

        # -----------------------------
        # Future dataframe
        # -----------------------------

        future = m.make_future_dataframe(periods=30)

        if use_price:
            future["price"] = ts["price"].iloc[-1]

        if use_promotion:
            future["promotion_flag"] = 0  # conservative assumption

        forecast = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        forecasts_by_sku[sku] = forecast.tail(30)

# --------------------------------------------------
# RUN PORTFOLIO AGENT
# --------------------------------------------------

if st.button("Run Store Analysis"):
    result = run_portfolio_agent(store_df, forecasts_by_sku)

    # --------------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------------

    st.subheader("Executive Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
    <div class="metric-card">
        <div style="font-size:13px;color:#666;">Overall Demand Situation</div>
        <div style="font-size:28px;font-weight:700;">{result["portfolio_status"]}</div>
    </div>
    """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
    <div class="metric-card">
        <div style="font-size:13px;color:#666;">Recommended Inventory Action</div>
        <div style="font-size:26px;font-weight:700;color:#a100ff;">
            {result["portfolio_action"]}
        </div>
    </div>
    """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
    <div class="metric-card">
        <div style="font-size:13px;color:#666;">Products Analyzed</div>
        <div style="font-size:28px;font-weight:700;">{result["sku_count"]}</div>
    </div>
    """,
            unsafe_allow_html=True,
        )


    # --------------------------------------------------
    # DEMAND STATUS (CLEAR LANGUAGE)
    # --------------------------------------------------

    # st.subheader("📊 Demand Situation Across Products")

    status_split = result["status_split"]

    declining = status_split.get("DEMAND DECLINING", 0)
    rising = status_split.get("DEMAND RISING", 0)
    stable = status_split.get("STABLE DEMAND", 0)

    st.markdown(
            f"""
        <div class="insight-box">
            <div style="font-weight:700;font-size:16px;margin-bottom:8px;">
                Demand Situation Across Products
            </div>

            • {declining}% of products show declining demand
            • {rising}% of products show growing demand
            • {stable}% of products have stable demand
        </div>
        """,
            unsafe_allow_html=True,
        )


    # --------------------------------------------------
    # BUSINESS INTERPRETATION (MOST IMPORTANT)
    # --------------------------------------------------

    st.subheader("Recommended Actions")
    st.markdown(
        f"""
    <div class="decision-banner">
        FINAL DECISION: {result["portfolio_action"]}
    </div>
    """,
        unsafe_allow_html=True,
    )


    if declining > 50:
        st.error(
            """
**Demand is weakening across most products.**

✔ Reduce overall inventory exposure  
✔ Avoid bulk purchasing  
✔ Focus on top-performing SKUs  
✔ Monitor sales frequently
"""
        )
    elif rising > 50:
        st.success(
            """
**Demand is strong across most products.**

✔ Increase inventory for strong SKUs  
✔ Prepare suppliers for higher volumes  
✔ Ensure fulfillment capacity
"""
        )
    else:
        st.info(
            """
**Demand is mixed but stable.**

✔ Maintain current inventory levels  
✔ Adjust selectively at SKU level  
✔ Continue monitoring trends
"""
        )

    # --------------------------------------------------
    # SKU LEVEL DETAILS (NO RISK COLUMN)
    # --------------------------------------------------

    with st.expander("📋 View Product-Level Details (Internal Use)"):

        display_table = result["sku_table"].copy()

# 🔥 REMOVE ALL RISK-RELATED COLUMNS (case-safe)
        cols_to_remove = [
            "risk",
            "RISK",
            "Risk",
            "risk_flag",
            "RISK_FLAG",
            "Risk_Flag"
        ]

        display_table = display_table.drop(
            columns=[c for c in cols_to_remove if c in display_table.columns],
            errors="ignore"
        )

        st.dataframe(display_table)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div style="text-align:center;color:#555;padding:10px;">
        Developed by <b>Shaishav Anand</b> • Accenture • © Copyright 2026 
    </div>
    """,
    unsafe_allow_html=True
)

