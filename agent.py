from groq import Groq
import os

# Load API key from environment variable
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY)

# -------------------------------
# AI INSIGHTS GENERATOR
# -------------------------------
def generate_insights(metrics, results, sku, store):
    prompt = f"""
    You are a retail analytics expert. 
    Generate concise insights based on:

    Store: {store}
    SKU: {sku}

    Model performance:
    {metrics}

    Forecast results:
    {results}

    Give 6–8 bullet points:
    - trend direction  
    - demand spikes or drops  
    - promotion effect  
    - price effect  
    - inventory risk  
    - actionable recommendations  
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# -------------------------------
# CHAT SYSTEM – AI QUESTION ANSWERING
# -------------------------------
def chat_with_ai(user_message, context=""):
    prompt = f"""
    You are an AI assistant inside a retail forecasting dashboard.
    Answer the user's question using the context only.
    Be accurate, concise and helpful.

    CONTEXT:
    {context}

    USER QUESTION:
    {user_message}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
