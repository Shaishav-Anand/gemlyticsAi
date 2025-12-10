from groq import Groq, RateLimitError, NotFoundError
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

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except RateLimitError as e:
        return "⚠️ AI Insights temporarily unavailable due to rate limit. Try again later."
    except NotFoundError:
        return "⚠️ Model not found. Please check your Groq account for available models."
    except Exception as e:
        return f"⚠️ AI Insights Error: {str(e)}"

# -------------------------------
# CHAT SYSTEM – AI QUESTION ANSWERING
# -------------------------------
def chat_with_ai(user_message, context="", max_context_chars=5000):
    """
    Ask a user question using AI assistant with Groq API.
    
    Parameters:
    - user_message: str, question from user
    - context: str, context to help AI answer
    - max_context_chars: int, max characters to include from context to avoid token limits
    """
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

    models_to_try = ["llama-3.3-70b-versatile", "llama-3.3-13b"]  # try smaller model if rate limit/error
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except RateLimitError:
            continue  # try next model
        except NotFoundError:
            continue  # try next model
        except Exception as e:
            continue

    # Fallback if all models fail
    return "⚠️ AI assistant is currently unavailable. Please try again later."
