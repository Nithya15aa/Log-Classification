# Hybrid Log Classification System

A project showcasing hybrid system to log classification that combines rule-based, machine learning, and LLM-based classification for robust and accurate log categorization.

## Overview

This project implements a three-layer classification pipeline designed for high-volume and diverse log environments:

1. **Regex Classifier** - Fast, deterministic rule-based classification
2. **ML Classifier** - Probabilistic classification using machine learning models
3. **LLM Classifier** - Advanced fallback using Large Language Models for complex cases

Each layer filters logs based on confidence thresholds, ensuring speed when possible while maintaining accuracy for ambiguous cases.


## Project Structure

```
├── app/                          # Main application 
├── models/                       # Trained model artifacts
├── data/                         # Data
├── scripts/                      # Scripts to train ML model
├── requirements.txt              # Python dependencies
├── run_api.sh                    # API startup script
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sejalshitole/Log-Classification.git
cd Log-Classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and set your configuration:
```env
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Running the API

Start the FastAPI server:
```bash
bash run_api.sh
```

or manually:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000/docs`

#### API Endpoints

**Classify a single log:**
```bash
curl -X POST "http://127.0.0.1:8000/v1/logs/analyze" \
  -H "Content-Type: application/json" \
  -d '{"log": "ERROR [2024-01-01 10:00:00] Database connection failed"}'
```

Response:
```json
{
  "label": "ERROR",
  "confidence": 0.98,
  "layer": "regex",
  "llm_explanation": null
}
```

**Batch process CSV file:**
```bash
curl -X POST "http://127.0.0.1:8000/v1/logs/analyze/batch" \
  -F "file=@logs.csv" \
  -F "log_column=message"
```

### API Response Format

```json
{
  "label": "Type of Log",
  "confidence": 0.0-1.0,
  "layer": "regex|ml|llm",
  "llm_explanation": "Optional explanation from LLM"
}
```

### Training the ML Model

Train or retrain the machine learning classifier:
```bash
python -m scripts.train_ml
```



## Classification Pipeline

The hybrid classifier uses a cascading approach:

```
Input Log
    ↓
[1] Regex Classifier (fast, deterministic)
    ├─ Confidence ≥ REGEX_CONFIDENCE? → Return
    └─ Confidence < REGEX_CONFIDENCE? → Next layer
    ↓
[2] ML Classifier (probabilistic)
    ├─ Confidence ≥ ML_CONFIDENCE? → Return
    └─ Confidence < ML_CONFIDENCE? → Next layer
    ↓
[3] LLM Classifier (slow, most accurate)
    └─ Return with explanation
```