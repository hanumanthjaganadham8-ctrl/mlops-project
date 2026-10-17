# AgriSense — MLOps Crop Recommendation System

An end-to-end Machine Learning Operations (MLOps) system that recommends the most suitable crops based on soil nutrients and meteorological parameters. Built with a trained machine learning pipeline, FastAPI REST API, and an interactive modern web dashboard.

---

## Architecture Overview

- **Problem Type**: Multi-class tabular classification (22 crop varieties)
- **Model**: Random Forest classifier bundled with `StandardScaler` inside a scikit-learn `Pipeline`
- **Quality Gate**: Automated test evaluation (minimum accuracy threshold $\ge 90\%$)
- **Backend**: FastAPI REST service with Pydantic schema validation
- **Frontend**: Clean interactive web dashboard (Fieldwise)
- **Monitoring & Metrics**: Automated metrics persistence (`models/metrics.json`)

---

## Project Structure

```text
├── ARCHITECTURE.md            # Detailed system architecture & design specs
├── Crop_recommendation.csv    # Agronomic dataset (2,200 samples, 22 crops)
├── Dockerfile                 # Container specification
├── README.md                  # Project overview and documentation
├── api.py                     # API entry point alias
├── app.py                     # FastAPI application & route definitions
├── generate_data.py           # Synthetic data generation utility
├── models/
│   ├── crop_model.joblib      # Trained pipeline (StandardScaler + RandomForest)
│   └── metrics.json           # Model evaluation report & feature importances
├── predict.py                 # Inference engine & CLI prediction interface
├── requirements.txt           # Pinned production dependencies
├── test_system.py             # End-to-end API & model verification suite
├── train.py                   # Model training, CV tuning, and evaluation script
└── frontend/
    ├── app.js                 # Dynamic UI logic, validation, & API integration
    ├── index.html             # Fieldwise user interface
    └── styles.css             # Theme, responsive layout, & progress bars
```

---

## Getting Started

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
python -m uvicorn app:app --reload --port 8000
```

- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Model Metrics**: [http://localhost:8000/model-info](http://localhost:8000/model-info)

### 3. Run Inference via CLI

```bash
python predict.py --N 90 --P 42 --K 43 --temperature 21 --humidity 82 --ph 6.5 --rainfall 203
```

### 4. Retrain the Model

```bash
python train.py
```

### 5. Run Verification Tests

```bash
python test_system.py
```
