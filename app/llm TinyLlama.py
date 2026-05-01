from transformers import pipeline

def load_llm():
    generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto"
    )
    
    return generator