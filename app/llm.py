import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    
    client = Groq(api_key=api_key)
    
    return client