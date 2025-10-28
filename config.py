"""
Configuration management for BME_Bot
Centralizes all settings for easy maintenance and deployment
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "db_faiss"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create necessary directories
    LOGS_DIR.mkdir(exist_ok=True)
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Model configurations
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.0
    
    # Retrieval configurations
    RETRIEVAL_K = 6  # Number of documents to retrieve
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Streamlit configurations
    PAGE_TITLE = "BME Bot - Your Biomedical Assistant"
    PAGE_ICON = "🔬"
    INITIAL_MESSAGE = "Hello! I'm BME Bot, your biomedical engineering assistant. Ask me anything!"
    
    # Prompt template
    CUSTOM_PROMPT_TEMPLATE = """You are a helpful assistant with two modes:

1. QA Mode (when the user asks about troubleshooting, errors, procedures, components, manuals, or technical topics):
   - ONLY use the provided CONTEXT to answer.
   - If the answer is not in the CONTEXT, reply exactly: "I don't know based on the available documentation."
   - Do not use outside knowledge or make assumptions.
   - Be detailed and cite specific information from the context.
   - If multiple sources provide relevant info, synthesize them.

2. Chat Mode (when the user engages in casual conversation, greetings, or non-technical small talk):
   - Respond naturally and conversationally.
   - You can use general knowledge here.
   - Keep responses friendly and concise.

Context:
{context}

User message:
{question}"""
