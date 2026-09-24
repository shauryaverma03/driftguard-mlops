# DriftGuard — Autonomous MLOps Pipeline

<div align="center">

**Continuous statistical drift detection, validation-gated retraining, and zero-downtime hot-swap for a live market manipulation classifier.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-3.1+-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io)
[![Tests](https://img.shields.io/badge/Tests-11%20passing-22c55e?style=flat-square)](#tests)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)

</div>

---

> A model trained once and left alone degrades silently the moment the world it's watching changes. DriftGuard detects when live trade patterns drift away from training — then retrains, validates, and hot-swaps without human intervention.

---

## What It Does

DriftGuard wraps a real market-manipulation classifier behind a FastAPI serving layer. Every prediction is logged. On a rolling window of that traffic, a **Kolmogorov–Smirnov (KS) test** compares live feature distributions against the training baseline — per feature, continuously. When enough features breach statistical significance, the pipeline fires automatically:

```
Drift Detected
    │
    ▼
Aggregate Training Data
(baseline + drift-adaptation batch + labeled recent logs)
    │
    ▼
Train Challenger Model
(alternates GradientBoosting / RandomForest across cycles)
    │
    ▼
Validation Gate vs. Current Champion
(head-to-head on held-out drifted test set)
    │
    ├── Challenger loses → rejected, logged, champion stays live
    │
    └── Challenger wins → MLflow run logged → registry staged
                                → artifact saved → atomic hot-swap
```

---

## Architecture

```
Traffic Simulator ──► FastAPI /predict ──► SQLite Feature Log
                                                    │
                                            KS-Test Drift Detector
                                          (per-feature, rolling window)
                                                    │
                                          [drift threshold breached?]
                                                    │
                                          Retraining Service
                                   (thread-locked, 10s timeout guard)
                                                    │
                          Baseline data + drift-adaptation batch + recent logs
                                                    │
                                       Train Challenger (GB / RF)
                                                    │
                                    Validation Gate vs. Champion
                                     (drift-set accuracy comparison)
                                                    │
                          ┌─────────────────────────┴─────────────────────────┐
                    Rejected: logged,                             Approved: MLflow run,
                    champion stays live                         registry promotion, artifact
                                                                  save, atomic hot-swap
                                                                           │
                                                               Prometheus /metrics
                                                                           │
                                                                 Live Dashboard
```

---

## Quick Start

```bash
# 1. Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# 2. Initialize baseline model and training data
PYTHONPATH=. python3 -m ml.train_baseline

# 3. Start services (two terminals)
# Terminal 1 — API Server + Live Dashboard
PYTHONPATH=. uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — MLflow Tracking + Model Registry
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

> **macOS Note**: Port `5000` is reserved by AirPlay. Use `--port 5001` for MLflow.

**Docker Compose (alternative):**
```bash
docker-compose up --build
```

---

## Access Points

| Interface | URL | Description |
|:---|:---|:---|
| **Operations Console** | `http://127.0.0.1:8000/` | Live drift radar, controls, model registry |
| **API Docs** | `http://127.0.0.1:8000/docs` | Swagger UI for all endpoints |
| **MLflow Registry** | `http://127.0.0.1:5001/` | Experiment runs and model stage transitions |
| **Prometheus Metrics** | `http://127.0.0.1:8000/metrics` | Raw scrape endpoint |

---

## 3-Minute Demo Script

**1. Open the console** — observe `HEALTHY` status, active model version (`v1`), and all feature KS p-values comfortably above the `p = 0.05` threshold.

**2. Stream Normal Traffic** — click *Stream Traffic*. Watch the request counter climb while drift status stays `HEALTHY`. Proves the system doesn't false-alarm on ordinary variation.

**3. Inject Drifted Batch** — click *Inject Drift (50)*. Watch KS p-values drop below `0.05` across multiple features in real time. The header banner flips to `DRIFT DETECTED (N/5)`.

**4. Watch the auto-pipeline fire** — the challenger trains, gets validated against the current champion, logs to MLflow, and hot-swaps into production. No manual click required.

**5. Check before/after accuracy** — old model vs. new model accuracy on the drifted benchmark, shown side-by-side with the improvement delta.

**6. Inspect MLflow** — open `:5001` to see the tracked run and the Production stage transition.

**7. Check `/metrics`** — raw Prometheus output showing `driftguard_requests_total`, `driftguard_feature_ks_pvalue`, `driftguard_system_drift_detected`, and more.

---

## API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Service status, active model version, request/retrain counts, uptime |
| `POST` | `/predict` | Score a single trade feature vector; logs it to the rolling window |
| `POST` | `/predict/batch` | Batch inference over multiple trade feature vectors |
| `GET` | `/drift/status` | Live KS-test drift state and per-feature p-values |
| `POST` | `/drift/check` | Force an immediate drift evaluation |
| `POST` | `/simulate/stream` | Start streaming synthetic normal trade traffic |
| `POST` | `/simulate/drift` | Inject a synthetic drifted batch to trigger detection |
| `POST` | `/simulate/stop` | Stop the active traffic stream |
| `POST` | `/retrain/trigger` | Manually run the full retrain → validate → promote pipeline |
| `POST` | `/pipeline/reset` | Clear rolling window, stop streaming, reload baseline model |
| `GET` | `/models/active` | Metadata for the currently serving production model |
| `GET` | `/models` | All registered model versions |
| `GET` | `/events` | Recent pipeline audit events, newest first |
| `GET` | `/drift/comparison` | Latest before/after accuracy comparison on drifted data |
| `GET` | `/metrics` | Prometheus-format metrics scrape endpoint |

---

## Features Monitored

| Feature | Normal Range | Manipulation Signature |
|:---|:---|:---|
| `cancel_ratio` | 0.05 – 0.40 | High (0.80–0.99) in Spoofing / Layering |
| `order_to_cancel_latency_ms` | 500ms – 4000ms | Sub-50ms rapid cancellation bursts |
| `order_size_zscore` | −1.5 to +1.5 | Abnormal size spikes (+3.0 to +9.0) |
| `order_frequency` | 1 – 30 req/min | High-frequency spam (80–250 req/min) |
| `time_between_orders_ms` | 800ms – 5000ms | Microsecond inter-arrival churn |

---

## Key Design Decisions

**Validation gate before every promotion.**
Auto-retraining without a safety check is how you silently ship a worse model. Every challenger is benchmarked against the champion's accuracy on a drifted test set before promotion. If it doesn't win, it's rejected, logged, and the current model keeps serving.

**Blended training data — not just the drift batch.**
Training only on the injected drift batch would let the challenger overfit to that exact pattern. DriftGuard blends the original baseline, a freshly synthesized drift-adaptation batch, and any recently logged labeled live requests to keep the challenger generalizable.

**Thread lock with timeout on the retraining service.**
If drift is detected twice in quick succession, two concurrent retrains could corrupt shared state. The service acquires a lock (10s timeout) before starting and returns `IN_PROGRESS` to any overlapping trigger.

**Alternating model types across retrain cycles.**
Prevents every challenger from being a copy of the same algorithm family — a deliberate choice to avoid the retrain loop reproducing the same biases.

**MLflow over custom version tracking.**
Registry staging (`Staging` → `Production`) is built in, so "hot-swap the model" reduces to "load whatever's tagged Production" with no custom versioning logic to maintain.

---

## Tech Stack

| Layer | Technology |
|:---|:---|
| **API Serving** | FastAPI, Uvicorn |
| **ML Models** | scikit-learn — GradientBoosting, RandomForest |
| **Drift Detection** | `scipy.stats.ks_2samp` (Kolmogorov–Smirnov two-sample test) |
| **Experiment Tracking** | MLflow (tracking server + model registry) |
| **Monitoring** | Prometheus (`/metrics` scrape endpoint) |
| **Storage** | SQLite (feature log, pipeline event log, model registry cache) |
| **Dashboard** | Vanilla HTML/CSS/JS served via FastAPI `StaticFiles` mount |
| **Deployment** | Docker Compose — API + dashboard container, Prometheus container |

---

## Tests

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

11 tests covering:
- Synthetic data generator properties (baseline + drifted distributions)
- FastAPI serving endpoints (`/health`, `/predict` — normal and manipulation patterns)
- End-to-end pipeline: drift detection → retrain trigger → validation gate → hot-swap
- Prometheus metrics endpoint
- Before/after accuracy comparison
- Pipeline audit event log

---

## Repository Layout

```
driftguard-mlops/
├── backend/
│   ├── main.py               # FastAPI app, all API routes
│   ├── drift_detector.py     # KS-test drift engine
│   ├── retraining_service.py # Thread-safe retrain orchestrator
│   ├── simulator.py          # Traffic stream + drift injection
│   ├── model_manager.py      # Thread-safe hot-swap model manager
│   ├── db.py                 # SQLite schema and query layer
│   ├── metrics.py            # Prometheus instrumentation
│   └── config.py             # Environment settings
├── ml/
│   ├── model.py              # Model training + evaluation
│   ├── validator.py          # Validation gate + benchmark comparison
│   ├── mlflow_utils.py       # MLflow logging + registry operations
│   └── train_baseline.py     # One-time baseline initialization script
├── data/
│   └── generator.py          # Synthetic market trade data generator
├── frontend/
│   └── index.html            # Operations console dashboard
├── tests/                    # Unit + integration test suite
├── docker/
│   ├── Dockerfile.backend
│   └── prometheus.yml
├── docker-compose.yml
├── requirements.txt
└── RUN.md                    # Full local run guide and demo script
```

---

## Documentation

| File | Contents |
|:---|:---|
| [`RUN.md`](RUN.md) | Full local run instructions, port mapping, and demo script |
| [`/docs`](http://127.0.0.1:8000/docs) | Interactive API documentation (Swagger UI, served live) |
