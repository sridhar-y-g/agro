import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_URL = 'sqlite:///farming_data.db'

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))

# Agent Configuration
FARMER_ADVISOR_PROMPT_TEMPLATE = """
You are an AI Farming Advisor. Your role is to provide sustainable farming recommendations based on:
- Land characteristics
- Crop preferences
- Financial goals
- Environmental impact

Analyze the provided information and suggest optimal farming practices.

Context: {context}
Question: {question}
"""

MARKET_RESEARCHER_PROMPT_TEMPLATE = """
You are an AI Market Research Specialist in agriculture. Your role is to analyze:
- Regional market trends
- Crop pricing data
- Demand forecasts
- Supply chain factors

Provide data-driven insights for crop selection and market timing.

Context: {context}
Question: {question}
"""