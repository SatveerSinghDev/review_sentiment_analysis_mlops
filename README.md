# 🎭 Review Sentiment Analysis — MLOps Pipeline

An end-to-end MLOps project that trains, tracks, deploys, and monitors a **sentiment analysis model for text reviews** (Positive / Negative). The project demonstrates a complete production-grade ML lifecycle: versioned data & code (DVC + S3), experiment tracking & model registry (MLflow + DagsHub), CI/CD (GitHub Actions), containerized deployment (Docker + AWS ECR + Kubernetes/EKS), and live observability (Prometheus + Grafana).

---

## 📌 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Pipeline Stages (DVC)](#-pipeline-stages-dvc)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Running the Pipeline](#-running-the-pipeline)
- [Experiment Tracking & Model Registry](#-experiment-tracking--model-registry)
- [Running the Web App](#-running-the-web-app)
- [Monitoring: Prometheus & Grafana](#-monitoring-prometheus--grafana)
- [Testing](#-testing)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Deployment (Docker + Kubernetes)](#-deployment-docker--kubernetes)
- [License](#-license)

---

## 🏗 Architecture Overview

```
                  ┌──────────────┐        ┌────────────────┐
                  │   Raw Data   │───────▶│  S3 (DVC remote) │
                  └──────┬───────┘        └────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │           DVC Pipeline            │
        │  ingestion → preprocessing →      │
        │  feature engineering → training → │
        │  evaluation → registration        │
        └───────────────┬───────────────────┘
                         │  metrics / artifacts
                         ▼
        ┌──────────────────────────────────┐
        │     MLflow Tracking (DagsHub)      │
        │  experiments · metrics · registry  │
        └───────────────┬───────────────────┘
                         │  Staging → Production
                         ▼
        ┌──────────────────────────────────┐
        │        GitHub Actions CI/CD       │
        │ repro → test → promote → build →  │
        │ push to ECR → deploy to EKS       │
        └───────────────┬───────────────────┘
                         ▼
        ┌──────────────────────────────────┐
        │   Flask App on Kubernetes (EKS)   │
        │      /   /predict   /metrics      │
        └───────────────┬───────────────────┘
                         │  scrape /metrics
                         ▼
        ┌───────────────┐        ┌──────────────┐
        │   Prometheus   │───────▶│   Grafana    │
        └───────────────┘        └──────────────┘
```

---

## 🧰 Tech Stack

| Layer                      | Tool(s) |
|-----------------------------|---------|
| Data & Pipeline Versioning  | [DVC](https://dvc.org/) with **AWS S3** remote |
| Experiment Tracking         | [MLflow](https://mlflow.org/) hosted on **DagsHub** |
| Model Registry              | MLflow Model Registry (Staging → Production) |
| Modeling                    | scikit-learn, NLTK (lemmatization, stopword removal) |
| Web App / Serving           | Flask + Gunicorn |
| Containerization            | Docker |
| Container Registry          | AWS ECR |
| Orchestration               | Kubernetes (AWS EKS) |
| CI/CD                       | GitHub Actions |
| Monitoring                  | Prometheus (metrics scraping) + Grafana (dashboards) |
| Testing                     | `unittest` |

---

## 📂 Project Structure

```
review_sentiment_analysis_mlops/
├── src/
│   ├── data/                  # Data ingestion & preprocessing
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   ├── features/              # Feature engineering (vectorization)
│   │   └── feature_engineering.py
│   ├── model/                 # Training, evaluation, registration
│   │   ├── model_building.py
│   │   ├── model_evaluation.py
│   │   └── register_model.py
│   ├── connections/           # External connectors
│   │   └── s3_connection.py
│   ├── visualization/         # Reporting/plots
│   └── logger/                # Central logging config
│
├── flask_app/                 # Production web app
│   ├── app.py                 # Flask routes + Prometheus metrics
│   ├── preprocessing_utility.py
│   ├── templates/index.html
│   └── requirements.txt
│
├── scripts/
│   └── promote_model.py       # Promotes Staging → Production in MLflow
│
├── tests/
│   ├── test_model.py
│   └── test_flask_app.py
│
├── .github/workflows/ci.yaml  # CI/CD pipeline definition
├── dvc.yaml                   # DVC pipeline stage definitions
├── dvc.lock                   # Locked pipeline state
├── params.yaml                # Pipeline hyperparameters
├── Dockerfile                 # Container build for the Flask app
├── deployment.yaml            # Kubernetes Deployment + Service manifest
├── requirements.txt           # Full pipeline dependencies
└── README.md
```

---

## 🔄 Pipeline Stages (DVC)

The full ML pipeline is defined declaratively in `dvc.yaml` and can be reproduced with a single command:

| Stage                  | Script                                 | Outputs |
|--------------------------|-----------------------------------------|---------|
| `data_ingestion`        | `src/data/data_ingestion.py`            | `data/raw` |
| `data_preprocessing`    | `src/data/data_preprocessing.py`        | `data/interim` |
| `feature_engineering`   | `src/features/feature_engineering.py`   | `data/processed`, `models/vectorizer.pkl` |
| `model_building`        | `src/model/model_building.py`           | `models/model.pkl` |
| `model_evaluation`      | `src/model/model_evaluation.py`         | `reports/metrics.json`, `reports/experiment_info.json` |
| `model_registration`    | `src/model/register_model.py`           | Registers model to MLflow Registry (Staging) |

Key hyperparameters live in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.25

feature_engineering:
  max_features: 50
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- Git
- [DVC](https://dvc.org/doc/install) with S3 support (`dvc[s3]`)
- An AWS account with an S3 bucket (used as the DVC remote)
- A [DagsHub](https://dagshub.com/) account with the repo mirrored there (for MLflow tracking)
- Docker (for containerized deployment)
- `kubectl` + access to an AWS EKS cluster (for Kubernetes deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/review_sentiment_analysis_mlops.git
cd review_sentiment_analysis_mlops

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies (also installs src/ as an editable package)
pip install -r requirements.txt

# Download required NLTK corpora
python -m nltk.downloader stopwords wordnet
```

---

## ⚙️ Configuration

### DVC Remote (S3)

DVC is configured to use an S3 bucket as the remote storage (see `.dvc/config`):

```ini
[core]
    remote = myremote
['remote "myremote"']
    url = s3://<your-bucket-name>
```

Set your AWS credentials via environment variables or the AWS CLI before pulling/pushing data:

```bash
aws configure
# or
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### DagsHub / MLflow

This project uses **DagsHub** as the hosted MLflow tracking server and model registry. Set a single environment variable containing your DagsHub token (used for both username and password by the scripts):

```bash
export CAPSTONE_TEST=<your-dagshub-token>
```

All tracking-related scripts (`model_evaluation.py`, `register_model.py`, `scripts/promote_model.py`, `flask_app/app.py`) point to:

```python
mlflow.set_tracking_uri("https://dagshub.com/<repo_owner>/<repo_name>.mlflow")
```

> Update `repo_owner` / `repo_name` in these files to match your own DagsHub repository.

---

## ▶️ Running the Pipeline

Pull the versioned data and reproduce the full pipeline with DVC:

```bash
# Pull data from the S3 remote
dvc pull

# Run/re-run the entire pipeline (only re-runs stages with changed deps)
dvc repro

# Visualize the pipeline DAG
dvc dag
```

This sequentially runs ingestion → preprocessing → feature engineering → training → evaluation → registration, logging parameters, metrics, and the trained model as an MLflow run on DagsHub.

---

## 📊 Experiment Tracking & Model Registry

Every pipeline run logs to MLflow, hosted on DagsHub:

- **Experiments/runs** — parameters, metrics (`accuracy`, `precision`, `recall`, `roc_auc`), and artifacts.
- **Model Registry** — the trained model is registered under the name `my_model`.
  - `register_model.py` registers the newly trained model and transitions it to the **Staging** stage.
  - `scripts/promote_model.py` archives the current **Production** model and promotes the latest **Staging** model to **Production** (run automatically after tests pass in CI).

View all runs, metrics, and registered model versions in your DagsHub repo under the **Experiments** and **Models** tabs.

---

## 🌐 Running the Web App

The Flask app (`flask_app/app.py`) loads the latest **Production** model from the MLflow Registry and serves predictions.

### Locally

```bash
export CAPSTONE_TEST=<your-dagshub-token>
cd flask_app
python app.py
```

The app runs on **http://localhost:5001**.

### With Docker

```bash
docker build -t review-sentiment-app .
docker run -p 5001:5001 -e CAPSTONE_TEST=<your-dagshub-token> review-sentiment-app
```

### Endpoints

| Route       | Method | Description |
|-------------|--------|-------------|
| `/`         | GET    | Renders the input form (`index.html`) |
| `/predict`  | POST   | Accepts `text`, cleans/vectorizes it, and returns the predicted sentiment |
| `/metrics`  | GET    | Exposes Prometheus-formatted metrics for scraping |

---

## 📈 Monitoring: Prometheus & Grafana

The Flask app is instrumented with `prometheus_client` and exposes a dedicated `/metrics` endpoint using a custom `CollectorRegistry`, so only app-specific metrics are exposed (not default process/GC metrics).

### Custom metrics exposed

| Metric                          | Type      | Labels                | Description |
|-----------------------------------|-----------|------------------------|--------------|
| `app_request_count`              | Counter   | `method`, `endpoint`   | Total requests received per endpoint |
| `app_request_latency_seconds`    | Histogram | `endpoint`             | Request latency distribution per endpoint |
| `model_prediction_count`         | Counter   | `prediction`           | Count of predictions grouped by predicted class |

### Setting up Prometheus

Add a scrape job pointing at the app's `/metrics` endpoint. Example `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "review-sentiment-app"
    static_configs:
      - targets: ["<app-host>:5001"]   # e.g. localhost:5001 or the k8s Service DNS
```

Run Prometheus (locally, for example):

```bash
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

Prometheus UI: **http://localhost:9090** — verify the `review-sentiment-app` target is `UP` under **Status → Targets**.

### Setting up Grafana

```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

1. Open Grafana at **http://localhost:3000** (default login `admin` / `admin`).
2. Add **Prometheus** as a data source, pointing at `http://<prometheus-host>:9090`.
3. Build dashboard panels using the metrics above, for example:
   - Request rate: `rate(app_request_count[1m])`
   - p95 latency: `histogram_quantile(0.95, rate(app_request_latency_seconds_bucket[5m]))`
   - Prediction class distribution: `sum by (prediction) (model_prediction_count)`

This gives real-time visibility into traffic, latency, and the live class distribution of model predictions in production — useful for spotting drift or unusual traffic patterns.

---

## ✅ Testing

Unit tests cover both the trained model and the Flask app:

```bash
# Validate the registered model (loads latest model version and sanity-checks predictions)
python -m unittest tests/test_model.py

# Validate Flask routes (/ and /predict)
python -m unittest tests/test_flask_app.py
```

Both test suites require `CAPSTONE_TEST` to be set, since they load the model from the MLflow Registry on DagsHub.

---

## 🔁 CI/CD Pipeline

Defined in `.github/workflows/ci.yaml` and triggered on every push. The pipeline:

1. Checks out the code and sets up Python 3.10 (with pip caching).
2. Installs dependencies.
3. Runs `dvc repro` to reproduce the full pipeline (ingestion → registration).
4. Runs model unit tests (`tests/test_model.py`).
5. Promotes the newly trained model from **Staging** to **Production** (`scripts/promote_model.py`).
6. Runs Flask app tests (`tests/test_flask_app.py`).
7. Authenticates to **AWS ECR** and builds the Docker image.
8. Tags and pushes the image to ECR.
9. Updates the `kubeconfig` for the target **EKS** cluster.
10. Creates/updates the `capstone-secret` Kubernetes secret from `CAPSTONE_TEST`.
11. Applies `deployment.yaml` to deploy the updated app to EKS.

### Required GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `CAPSTONE_TEST` | DagsHub token for MLflow tracking/registry access |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `AWS_REGION` | AWS region (e.g. `us-east-1`) |
| `AWS_ACCOUNT_ID` | AWS account ID (for ECR image URI) |
| `ECR_REPOSITORY` | ECR repository name |

---

## 📦 Deployment (Docker + Kubernetes)

- **Dockerfile** builds a slim Python 3.10 image, installs the Flask app's dependencies, downloads required NLTK corpora, and serves the app with **Gunicorn** on port `5001`.
- **`deployment.yaml`** defines:
  - A `Deployment` (`flask-app`) with 2 replicas, resource requests/limits, and the `CAPSTONE_TEST` env var sourced from a Kubernetes `Secret`.
  - A `LoadBalancer` `Service` (`flask-app-service`) exposing port `5001`.

To deploy manually to an existing EKS cluster:

```bash
aws eks update-kubeconfig --region us-east-1 --name flask-app-cluster
kubectl create secret generic capstone-secret --from-literal=CAPSTONE_TEST=<your-dagshub-token>
kubectl apply -f deployment.yaml
kubectl get svc flask-app-service   # grab the external LoadBalancer URL
```

For cluster-level monitoring, deploy `kube-prometheus-stack` (Prometheus + Grafana + Alertmanager) via Helm and add a `ServiceMonitor`/scrape config targeting the `flask-app-service` `/metrics` endpoint, so Grafana dashboards stay up to date automatically as pods scale.

---

## 📄 License

See `LICENSE` for details.

---

<p align="center"><small>Project scaffolding based on the <a href="https://drivendata.github.io/cookiecutter-data-science/" target="_blank">Cookiecutter Data Science</a> template.</small></p>
