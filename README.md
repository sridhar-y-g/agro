# Sustainable Farming AI System

A multi-agent AI system that leverages machine learning and data analytics to promote sustainable farming practices through intelligent decision support.

## Features

- **Farmer Advisor Agent**: Provides sustainable farming recommendations based on land characteristics, crop preferences, and environmental impact
- **Market Researcher Agent**: Analyzes market trends and provides data-driven insights for crop selection
- **SQLite Database**: Persistent storage for farming data, market trends, and agent interactions
- **RESTful API**: FastAPI-based interface for system interaction

## Prerequisites

- Python 3.8+
- Ollama (running locally)
- SQLite3

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (optional):
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   API_HOST=0.0.0.0
   API_PORT=8000
   ```

## Usage

1. Start the API server:
   ```bash
   python app.py
   ```

2. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

- `POST /farms/`: Create a new farm
- `POST /farms/{farm_id}/crops/`: Add crops to a farm
- `POST /advisor/query/`: Query the Farmer Advisor agent
- `POST /market/query/`: Query the Market Researcher agent
- `GET /farms/{farm_id}/recommendations/`: Get farm-specific recommendations
- `GET /market/analysis/{crop_name}`: Get market analysis for specific crops

## Architecture

- **Agent Framework**: Implements Farmer Advisor and Market Researcher using Ollama-based LLMs
- **Data Layer**: SQLite database with SQLAlchemy ORM
- **API Layer**: FastAPI-based RESTful interface
- **Integration Layer**: Custom tools for data processing and analysis