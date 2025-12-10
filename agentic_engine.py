import os
from groq import Groq, RateLimitError, NotFoundError
import pandas as pd
import json
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Load API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("Set GROQ_API_KEY environment variable before running.")

client = Groq(api_key=GROQ_API_KEY)

# -------------------------------
# HELPER: serialize dataset safely
# -------------------------------
def serialize_data(df: pd.DataFrame, max_rows=50):
    """
    Serialize dataframe into a small JSON summary
    - Keep only last `max_rows` rows
    - Convert datetimes and numeric types for JSON serialization
    """
    if len(df) > max_rows:
        df_small = df.tail(max_rows).copy()
    else:
        df_small = df.copy()

    # Convert all datetime columns to string
    datetime_cols = df_small.select_dtypes(include=['datetime', 'datetimetz']).columns
    for col in datetime_cols:
        df_small[col] = df_small[col].astype(str)

    # Convert to dict
    data_dict = df_small.to_dict(orient="records")

    # Convert numeric types to float
    for row in data_dict:
        for k, v in row.items():
            if isinstance(v, (np.integer, np.floating)):
                row[k] = float(v)

    return json.dumps(data_dict, indent=2)


# -------------------------------
# AI INSIGHTS GENERATOR
# -------------------------------
def generate_insights(metrics, results, sku, store, forecast_df=None):
    prompt = f"""
You are a retail analytics expert. 

Store: {store}
SKU: {sku}

Model performance metrics: {metrics}

Forecast results summary: {results}
"""

    if forecast_df is not None:
        prompt += "\nForecast (last 30 rows): " + serialize_data(forecast_df, max_rows=30)

    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except RateLimitError:
        return "⚠️ AI Insights temporarily unavailable due to rate limit."
    except NotFoundError:
        return "⚠️ Model not found. Please check your Groq account."
    except Exception as e:
        return f"⚠️ AI Insights Error: {str(e)}"


# -------------------------------
# CHAT SYSTEM – AI QUESTION ANSWERING
# -------------------------------
def chat_with_ai(user_message, context="", max_context_chars=5000):
    # Truncate context if too long
    if len(context) > max_context_chars:
        context = context[-max_context_chars:]  # take last part

    prompt = f"""
You are an AI assistant inside a retail forecasting dashboard.
Answer the user's question using the context only.
Be accurate, concise and helpful.

CONTEXT:
{context}

USER QUESTION:
{user_message}
"""
    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI assistant error: {str(e)}"


# -------------------------------
# RUN AGENTIC TASK
# -------------------------------
def run_agentic_task(prompt, data=None, existing_forecasts=None, metrics=None):
    """
    Main wrapper to call Groq agent
    - Trims large datasets automatically
    """
    # Add small serialized datasets to prompt
    if data is not None:
        prompt += "\nDataset summary: " + serialize_data(data, max_rows=30)
    if existing_forecasts is not None:
        prompt += "\nForecast summary: " + serialize_data(existing_forecasts, max_rows=30)
    if metrics is not None:
        prompt += "\nMetrics: " + json.dumps(metrics)

    try:
        msg = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=msg
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Agent Error: {e}"
