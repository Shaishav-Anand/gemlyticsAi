# agentic_engine.py

import json
import pandas as pd
from groq import Groq

from agent_tools import (
    compute_growth_against_actuals,
    compute_trend_direction,
    explain_prophet_forecast,
    compute_risk_label,
    compute_business_status,
    compute_action_from_status
)

# --------------------------------------------------
# LLM (EXPLANATION ONLY)
# --------------------------------------------------

# from dotenv import load_dotenv
# load_dotenv()

# # Load API key from .env
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise EnvironmentError("Set GROQ_API_KEY environment variable before running.")

# client = Groq(api_key=GROQ_API_KEY)


# --------------------------------------------------
# SINGLE-SKU AGENT
# --------------------------------------------------

def run_agentic_task(prompt, data=None, existing_forecasts=None, metrics=None):
    """
    Deterministic decision engine.
    LLM is READ-ONLY.
    """

    result = {
        "decision_model": "Prophet",
        "growth_vs_actuals": None,
        "business_status": None,
        "primary_action": None,
        "risk_flag": None,
        "forecast_explanation": [],
        "actions": [],
        "llm_explanation": None
    }

    # -----------------------------
    # Trend
    # -----------------------------
    actual_df = data.rename(columns={"date": "ds", "units_sold": "y"})
    trend, trend_reason = compute_trend_direction(actual_df)

    # -----------------------------
    # Growth
    # -----------------------------
    growth = compute_growth_against_actuals(actual_df, existing_forecasts)
    result["growth_vs_actuals"] = growth

    # -----------------------------
    # Decision
    # -----------------------------
    status = compute_business_status(growth)
    action = compute_action_from_status(status)

    result["business_status"] = status
    result["primary_action"] = action

    # -----------------------------
    # Forecast explanation
    # -----------------------------
    result["forecast_explanation"] = explain_prophet_forecast(existing_forecasts)
    result["risk_flag"] = compute_risk_label(existing_forecasts)

    # -----------------------------
    # Action list
    # -----------------------------
    if growth < -10:
        result["actions"] = [
            "Reduce replenishment",
            "Avoid aggressive promotions",
            "Monitor daily sales"
        ]
    elif growth > 15:
        result["actions"] = [
            "Increase reorder quantities",
            "Confirm supplier capacity",
            "Prevent stock-outs"
        ]
    else:
        result["actions"] = [
            "Maintain current inventory",
            "Monitor weekly trends"
        ]

    # -----------------------------
    # LLM explanation (OPTIONAL)
    # -----------------------------
    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{
                "role": "user",
                "content": f"""
Explain this business decision clearly.
Do NOT change any values.

{json.dumps(result, indent=2)}
"""
            }]
        )
        result["llm_explanation"] = response.choices[0].message.content
    except Exception as e:
        result["llm_explanation"] = f"Explanation unavailable: {e}"

    return result


# --------------------------------------------------
# MULTI-SKU PORTFOLIO AGENT
# --------------------------------------------------

def run_portfolio_agent(store_df, forecasts_by_sku):
    sku_rows = []

    for sku, sku_df in store_df.groupby("SKU_ID"):
        if sku not in forecasts_by_sku:
            continue

        out = run_agentic_task(
            prompt="Auto portfolio analysis",
            data=sku_df,
            existing_forecasts=forecasts_by_sku[sku],
            metrics=None
        )

        sku_rows.append({
            "SKU": sku,
            "Status": out["business_status"],
            "Action": out["primary_action"],
            "Risk": out["risk_flag"],
            "Growth %": out["growth_vs_actuals"]
        })

    if not sku_rows:
        return {"error": "No SKUs processed"}

    df = pd.DataFrame(sku_rows)

    avg_growth = df["Growth %"].mean()

    if avg_growth < -10:
        portfolio_status = "DEMAND DECLINING"
        portfolio_action = "REDUCE OVERALL INVENTORY"
    elif avg_growth > 15:
        portfolio_status = "DEMAND RISING"
        portfolio_action = "INCREASE OVERALL INVENTORY"
    else:
        portfolio_status = "STABLE DEMAND"
        portfolio_action = "MAINTAIN CURRENT LEVELS"

    return {
        "portfolio_status": portfolio_status,
        "portfolio_action": portfolio_action,
        "avg_growth": round(avg_growth, 2),
        "sku_count": len(df),
        "status_split": df["Status"].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "risk_split": df["Risk"].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "sku_table": df
    }
