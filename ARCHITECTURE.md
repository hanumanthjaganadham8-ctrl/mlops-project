# Crop Recommendation System — Architecture

## 1. Overview
Recommends the most suitable crop from soil nutrients (N, P, K, pH) and climate
(temperature, humidity, rainfall). Output is a ranked list of crops with confidence.

- **Problem type:** multi-class tabular classification
- **Model:** Random Forest / Gradient Boosting inside a scikit-learn Pipeline
- **Serving:** FastAPI REST service (containerized)

## 2. High-Level Architecture

```
                         ┌──────────────────────────┐
  Data Sources           │      DATA LAYER          │
  ─────────────          │                          │
  Soil test labs  ─────► │  Ingestion (batch/API)   │
  Weather API (IMD,      │  Raw store (S3 / GCS)    │
   OpenWeather)  ─────►  │  Validation (schema,     │
  Farmer app / IoT       │   ranges, nulls)         │
   sensors       ─────►  │  Curated store / Feature │
                         │  store (Parquet)         │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │      ML TRAINING         │
                         │  Feature engineering     │
                         │  Train / CV / tuning     │
                         │  Evaluation + quality    │
                         │  gate (acc >= 0.90)      │
                         │  Experiment tracking     │
                         │  (MLflow)                │
                         └────────────┬─────────────┘
                                      │ approved model
                         ┌────────────▼─────────────┐
                         │    MODEL REGISTRY        │
                         │  Versioned artifacts +   │
                         │  metrics.json            │
                         └────────────┬─────────────┘
                                      │
   Mobile / Web App      ┌────────────▼─────────────┐
   ─────────────────     │     SERVING LAYER        │
   Farmer UI  ─────────► │  API Gateway (auth,      │
   Agri extension        │   rate limit)            │
   officer dashboard ──► │  FastAPI inference svc   │
                         │  (Docker on K8s / ECS)   │
                         │  Input validation        │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   MONITORING & MLOPS     │
                         │  Latency / error metrics │
                         │  (Prometheus + Grafana)  │
                         │  Data & prediction drift │
                         │  Feedback loop (actual   │
                         │  yield / farmer choice)  │
                         │  Scheduled retraining    │
                         └──────────────────────────┘
```

## 3. Components

| Layer | Responsibility | Tech (typical) |
|---|---|---|
| Ingestion | Collect soil reports, weather, sensor data | Airflow, Kafka, REST connectors |
| Storage | Raw + curated datasets | S3/GCS, Parquet, PostgreSQL |
| Validation | Schema, range, null checks | Great Expectations / Pydantic |
| Training | Preprocess, CV, tune, evaluate | scikit-learn, Optuna |
| Tracking/Registry | Runs, metrics, model versions | MLflow |
| Serving | Low-latency REST inference | FastAPI, Uvicorn, Docker |
| Gateway | AuthN/Z, throttling, TLS | Nginx / Kong / AWS API Gateway |
| Orchestration | Deploy, autoscale | Kubernetes / ECS |
| CI/CD | Test, build, deploy | GitHub Actions |
| Monitoring | Metrics, logs, drift alerts | Prometheus, Grafana, Evidently |

## 4. ML Pipeline

1. **Ingest** data → validate schema and physical ranges.
2. **Split** 80/20 stratified; 5-fold CV on the training set.
3. **Pipeline** = `StandardScaler` + classifier (preprocessing ships with the model → no train/serve skew).
4. **Compare** candidates (RF vs GB), pick best by CV accuracy.
5. **Quality gate**: model saved only if test accuracy ≥ 0.90.
6. **Persist** `crop_model.joblib` + `metrics.json` (version, metrics, feature importance).

## 5. Inference Contract

`POST /v1/recommend`
```json
{ "N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph": 6.5, "rainfall": 203 }
```
Response:
```json
{
  "recommended_crop": "rice",
  "confidence": 0.97,
  "top_predictions": [{"crop": "rice", "probability": 0.97}, ...]
}
```
Other endpoints: `GET /health`, `GET /model-info`.

## 6. Non-Functional Requirements

- **Latency:** < 100 ms p95 (model loaded once, cached in memory)
- **Scalability:** stateless service, horizontal autoscaling
- **Security:** TLS, API keys/OAuth2 at gateway, input validation, no PII stored
- **Reproducibility:** pinned requirements, fixed seeds, versioned data + models
- **Observability:** structured logs, request metrics, drift alerts
- **Explainability:** feature importance + top-3 alternatives with probabilities

## 7. Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

CI/CD flow: PR → lint + unit tests → train + gate → build image → deploy to staging → smoke test → promote to prod.

## 8. Monitoring & Retraining

- **Data drift:** compare live input distributions vs training (PSI / KS test)
- **Concept drift:** track recommendation vs actual crop/yield feedback
- **Retrain triggers:** drift alert, new season data, monthly schedule
- **Rollback:** registry keeps previous model versions

## 9. Future Enhancements

- Auto-fetch weather and soil data by GPS location
- Add season, region, market price, and yield-profit optimization
- Multilingual farmer UI (Gujarati, Hindi) and voice input
- Satellite imagery (NDVI) as additional features
- SHAP-based explanations per prediction

## 10. Project Structure

```
crop_recommendation/
├── ARCHITECTURE.md
├── requirements.txt
├── generate_data.py   # synthetic data (same schema as Kaggle dataset)
├── train.py           # train + evaluate + save model
├── predict.py         # inference module + CLI
├── api.py             # FastAPI service
├── data/crop_data.csv
└── models/{crop_model.joblib, metrics.json}
```

## 11. Quick Start

```bash
pip install -r requirements.txt
python generate_data.py      # or place the real Kaggle CSV at data/crop_data.csv
python train.py
python predict.py --N 90 --P 42 --K 43 --temperature 21 --humidity 82 --ph 6.5 --rainfall 203
uvicorn api:app --reload
```
