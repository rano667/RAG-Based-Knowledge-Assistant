import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    
    print("API Key loaded:", api_key[:5], "****")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment")
    
    client = Groq(api_key=api_key)
    return client