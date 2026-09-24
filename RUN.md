# DriftGuard — Local Run & Live Demo Guide

Automated MLOps Pipeline for Continuous Model Drift Detection, MLflow Tracking, Zero-Downtime Hot-Swapping, and Real-Time Dashboard.

---

## 🚀 Quickstart (Option 1: Direct Python Execution)

### 1. Setup Virtual Environment & Install Dependencies
```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Initialize Baseline Model & Data
```bash
PYTHONPATH=. python3 -m ml.train_baseline
```

### 3. Start the Services

#### Terminal 1: FastAPI Serving Server + Live Dashboard
```bash
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

#### Terminal 2: MLflow Experiment Tracking & Model Registry
> **Note for macOS**: Port `5001` is used to avoid macOS AirPlay Receiver port `5000` conflicts.
```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## 🌐 Web Dashboards & Port Mapping

| Interface | URL | Purpose |
| :--- | :--- | :--- |
| **DriftGuard Live UI** | [http://127.0.0.1:8000/](http://127.0.0.1:8000/) | Live Drift Radar, Traffic Controls, Hot-Swap & Comparison |
| **Interactive API Docs** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Swagger UI for `/predict`, `/drift/status`, `/metrics` |
| **MLflow Registry** | [http://127.0.0.1:5001/](http://127.0.0.1:5001/) | Experiment runs, model artifacts & production stages |
| **Prometheus Metrics** | [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics) | Live Prometheus metrics scrape feed |

---

## 🎯 Live 3-Minute Hackathon Demo Script

1. **Open the Dashboard**: Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
   - Point out **System State: HEALTHY**, active model version **`v1`** (Production), and the real-time KS p-value progress bars ($p > 0.05$).
2. **Stream Normal Trades**: Click **"Stream Normal Traffic"**.
   - Watch the request counter increment and see the rolling window accumulate data while remaining healthy. Click again to pause.
3. **Inject Drift Batch**: Click **"Inject Drifted Batch (50)"**.
   - Watch the KS test immediately detect distribution shift ($p < 0.05$ across 5 features).
   - The banner turns red: **`DRIFT DETECTED`**.
   - The auto-retraining pipeline triggers automatically in the background.
4. **Zero-Downtime Hot-Swap**:
   - The challenger model is trained, validated against the benchmark, logged to MLflow, and atomically hot-swapped to **`v2`** with zero service interruption.
   - The **Retraining Benchmark Comparison** updates showing before vs after accuracy gains.
5. **Inspect MLflow & Metrics**:
   - Open MLflow at [http://127.0.0.1:5001/](http://127.0.0.1:5001/) to show the tracked runs and Model Registry transitions.
   - Open [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics) to view the Prometheus gauge counters.

---

## 🐳 Quickstart (Option 2: Docker Compose)

```bash
docker-compose up --build
```
- FastAPI & Dashboard: `http://localhost:8000`
- Prometheus Dashboard: `http://localhost:9090`

---

## 🧪 Run Automated Test Suite

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```
All 11 unit & end-to-end integration tests will run and validate the complete pipeline.
