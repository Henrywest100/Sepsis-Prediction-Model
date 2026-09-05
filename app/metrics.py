from prometheus_client import Counter, Histogram, Gauge

# Create metrics
PREDICTION_REQUEST_TOTAL = Counter(
    "elvara_sepsis_prediction_total",
    "Total count of sepsis risk prediction served",
    ["risk_category"]
)

PREDICTION_LATENCY_SECONDS = Histogram(
    'elvara_sepsis_prediction_latency_seconds',
    'time taken to execute feature engineering and model inference',
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

MODEL_LOADED_GAUGE = Gauge(
    'elvara_sepsis_model_loaded',
    '1 if model artefacts is loaded successfully, 0 otherwise'
)

# ✅ Force registration by accessing the metric
# This ensures it's registered at import time
_ = PREDICTION_REQUEST_TOTAL.labels(risk_category="init")