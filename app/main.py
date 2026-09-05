import time
import os
import json
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.schemas import *
from src.predict import SepsisPredictor

from app.metrics import (
    PREDICTION_REQUEST_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    MODEL_LOADED_GAUGE)

app = FastAPI(
    title = 'Elvara Health | Early Sepsis warning and deterioration System',
    description = 'Machine Learning Operations (MLOPS) and Clinical Decision Support System',
    version = '1.0.0'
)

@app.get('/')
def root():
    return {
        "message": "Welcome to Elvara Health Sepsis Prediction API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict-risk"
        }
    }


predictor = None

@app.on_event('startup')
def load_model():
    global predictor
    try:
        model_path = os.getenv('MODEL_PATH', 'models/sepsis_model.joblib')
        predictor = SepsisPredictor(model_path=model_path)
        MODEL_LOADED_GAUGE.set(1)
    except Exception as e:
        MODEL_LOADED_GAUGE.set(0)
        print(f'error loading model: {e}')

@app.get('/health', response_model=HealthCheckResponse)
def health_check():
    is_loaded = predictor is not None
    return HealthCheckResponse(
        status='healthy' if is_loaded else 'degraded',
        service= 'elvara-sepsis-cdss',
        model_loaded=is_loaded,
        version= '1.0.0'
    )

@app.post('/predict-risk', response_model = SepsisPredictionResponse)
def predict_risk(request: SepsisPredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail='Sepsis ML Model not loaded.')

    start_time = time.time()
    try:
        patient_dict = request.dict()
        result = predictor.predict_patient(patient_dict)

        latency = time.time() - start_time
        PREDICTION_LATENCY_SECONDS.observe(latency)
        PREDICTION_REQUEST_TOTAL.labels(risk_category=result['risk_category']).inc()

        return SepsisPredictionResponse(
            patient_id=result['patient_id'],
            sepsis_risk_score=result['sepsis_risk_score'],
            risk_category=result['risk_category'],
            prediction_window=result['prediction_window'],
            key_risk_factors=result['key_risk_factors'],
            features_extracted=result.get('features_extracted', {})
            

        )

    except Exception as e:
        raise HTTPException(status_code=500,
        detail=f'prediction error: {e}')

@app.get('/metrics')
def metrics():
    return Response (content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post('/monitoring/drift-report')
def generate_drift_report():
    try:
        from monitoring.drift_monitor import run_drift_analysis
        report_path = run_drift_analysis()
        return {
            "status": "success",
            "report_path": report_path,
            "message": "Drift report generated successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f'Failed to generate drift report: {str(e)}'
        )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8002)                            
