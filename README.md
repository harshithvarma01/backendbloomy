# BloomBuddy Backend

A Flask-based API server for BloomBuddy's machine learning models and LLM integration.

## Features

- Machine Learning prediction endpoints for:
  - Diabetes risk assessment
  - Heart disease prediction  
  - Hypertension risk evaluation
- LLM proxy endpoint for Anthropic Claude API
- CORS enabled for frontend integration
- Health check and debug endpoints

## API Endpoints

### Health Check
- `GET /health` - Server health status

### ML Predictions
- `POST /api/predict/diabetes` - Diabetes risk prediction
- `POST /api/predict/heart` - Heart disease prediction
- `POST /api/predict/hypertension` - Hypertension prediction

### LLM Integration
- `POST /api/llm/chat` - Proxy for LLM API calls

### Debug/Info
- `GET /debug/models` - Model loading status
- `GET /api/models/info` - Model information

## Environment Variables

Create a `.env` file with:

```
ANTHROPIC_API_KEY=your_api_key_here
MODELS_DIR=./models
PORT=5000
DEBUG=False
```

## Deployment

This backend is configured for deployment on Render with:
- `Procfile` for process definition
- `runtime.txt` for Python version
- `requirements.txt` for dependencies

## Model Files Required

The following pickle files must be present in the `models/` directory:
- `diabetes_model.pkl` & `diabetes_scaler.pkl`
- `heart_disease_model.pkl` & `heart_scaler.pkl`  
- `hypertension_model.pkl` & `hyper_scaler.pkl`
