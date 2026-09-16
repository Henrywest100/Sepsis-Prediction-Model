#  Elvara Health — Early Sepsis Warning & Deterioration System

An end-to-end **MLOps pipeline** that predicts sepsis risk within a **6–12 hour window** using patient vitals, lab results, and clinical history.

##  What It Does

Analyzes patient data and returns a **sepsis risk score**, **risk category** (Low/Moderate/High), and **prediction window** to support earlier clinical intervention.

##  Architecture
FastAPI (ML Model) → Prometheus (Metrics) → Grafana (Dashboards)

Deployed on **Railway** with **Docker** containers and **GitHub** CI/CD.



##  Tech Stack

 Layer | Tools 

 ML Model | Scikit-learn, Pandas, NumPy 
API | FastAPI, Uvicorn, Pydantic 
 Deployment | Docker, Railway 
 Monitoring | Prometheus, Grafana 
 ML Monitoring | Evidently AI 


##  Live Endpoints

 Resource | URL 

 API | https://sepsis-prediction-model-production.up.railway.app
 Docs | /docs |
 Health | /health |
 Metrics | /metrics |

 ##  API Example

**POST** `/predict-risk`

```json
{
  "patient_id": 101,
  "age": 65,
  "gender": "Male",
  "comorbidity_count": 2,
  "vitals": [{"heart_rate": 95, "temperature": 38.2, "oxygen_saturation": 94}],
  "labs": [{"white_cell_count": 14.5, "crp": 55.0, "lactate": 2.8}]
}

 ## Response

 {
  "sepsis_risk_score": 0.8542,
  "risk_category": "High",
  "prediction_window": "6 hours"
}

## Monitoring
Prometheus — API latency, prediction counts, model health

Grafana — Dashboards for real-time metrics

Evidently AI — Feature drift & prediction drift detection


Author
Henry Ehis Itua
GitHub: @Henrywest100