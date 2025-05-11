import os
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import types

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

def call_llm(query, context_docs):
    # Check if API key is available
    if not gemini_api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."
    
    # Join context documents into a single prompt
    context_text = "\n".join(context_docs)

    system_instruction = "You are an intelligent assistant. Based on the provided context, answer the user's question concisely and accurately. If the answer is not in the context, say that you don't know."

    prompt = f"""Context:
{context_text}

Question:
{query}"""    # Call the Gemini model
    try:
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 300,
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=system_instruction,
        )
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error while calling Gemini LLM: {e}"
