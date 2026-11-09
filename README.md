# YouTube Comment Insights Engine

> An end-to-end NLP and MLOps backend for classifying YouTube comments and turning the results into useful visual insights.

[![CI/CD](https://github.com/sahilshore/youtube-comment-insights-engine/actions/workflows/cicd.yaml/badge.svg)](https://github.com/sahilshore/youtube-comment-insights-engine/actions/workflows/cicd.yaml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/API-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![DVC](https://img.shields.io/badge/data%20%26%20pipelines-DVC-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/experiment%20tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)

The engine is the backend for the companion Chrome extension:

**[sahilshore/yt-chrome-plugin-frontend](https://github.com/sahilshore/yt-chrome-plugin-frontend)**

Together, the two repositories provide a complete workflow:

```text
YouTube video
    │
    ▼
Chrome extension ── YouTube Data API ──► up to 500 comments
    │
    ▼
Flask API ── preprocessing + TF-IDF ──► registered LightGBM model
    │
    ├── sentiment predictions
    ├── sentiment distribution chart
    ├── monthly sentiment trend graph
    └── comment word cloud
```

## What this project does

- Ingests and cleans labelled comment data.
- Applies a reproducible DVC pipeline for ingestion, preprocessing, model building, evaluation, and registration.
- Normalizes text using lowercasing, whitespace cleanup, punctuation filtering, selected stop-word removal, and lemmatization.
- Represents text with a TF-IDF vectorizer using configurable n-grams.
- Trains a three-class LightGBM classifier with labels:
  - `1` — positive
  - `0` — neutral
  - `-1` — negative
- Tracks experiments and model artifacts with MLflow.
- Loads the exact vectorizer associated with the registered model version.
- Exposes prediction and analytics endpoints through Flask.
- Supports Docker packaging and GitHub Actions CI/CD.

> **Important:** this repository contains the code and pipeline metadata, but the trained model/vectorizer outputs are generated or restored through DVC/MLflow. A plain `git clone` is not sufficient to run the API until those artifacts and MLflow access are available.

## Table of contents

- [Quick start](#quick-start)
- [Choose your setup path](#choose-your-setup-path)
- [How the pipeline works](#how-the-pipeline-works)
- [Run the Flask API](#run-the-flask-api)
- [API reference](#api-reference)
- [Connect the Chrome extension](#connect-the-chrome-extension)
- [Train and reproduce the model](#train-and-reproduce-the-model)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Repository guide](#repository-guide)
- [Troubleshooting](#troubleshooting)
- [Security and production notes](#security-and-production-notes)
- [Development roadmap](#development-roadmap)
- [Contributing](#contributing)
- [License](#license)

## Quick start

### Prerequisites

- Python 3.10
- Git
- Access to the configured DVC remote if restoring pipeline outputs
- Access to the MLflow tracking/model registry used by the deployment
- A YouTube Data API v3 key for the companion extension
- Docker, only if you want to build the container

### 1. Clone the backend

```bash
git clone https://github.com/sahilshore/youtube-comment-insights-engine.git
cd youtube-comment-insights-engine
```

### 2. Create an isolated environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies

The root dependency file is a pinned, UTF-16 encoded environment export. Use Python to install it reliably:

```bash
python - <<'PY'
from pathlib import Path
import subprocess
import sys

requirements = Path("requirements.txt").read_text(encoding="utf-16")
Path(".requirements.utf8.txt").write_text(requirements, encoding="utf-8")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", ".requirements.utf8.txt"])
Path(".requirements.utf8.txt").unlink()
PY
```

For API-only development, the smaller service manifest is easier to use:

```bash
python -m pip install -r flask_app/requirements.txt
python -m nltk.downloader stopwords wordnet
```

### 4. Restore or build artifacts

If the DVC remote is available:

```bash
dvc pull
dvc repro
```

If the model is already registered in MLflow, make sure the API's configured model name/version and tracking server are reachable before starting it.

### 5. Start the API

```bash
python flask_app/app.py
```

Verify that it is running:

```bash
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{"status":"healthy"}
```

## Choose your setup path

<details>
<summary><strong>I only want to inspect or develop the ML pipeline</strong></summary>

Install the root requirements, then run each stage directly:

```bash
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py
```

The recommended reproducible command is:

```bash
dvc repro
```

</details>

<details>
<summary><strong>I only want to run the prediction API</strong></summary>

Install `flask_app/requirements.txt`, download the NLTK resources, restore the model from MLflow, and start:

```bash
python -m nltk.downloader stopwords wordnet
python flask_app/app.py
```

The API loads the registered model and its matching `tfidf_vectorizer.pkl` during application startup.

</details>

<details>
<summary><strong>I want to use the Chrome extension</strong></summary>

1. Start the backend on `http://localhost:5000`.
2. Configure a valid YouTube Data API key in the frontend repository.
3. Open `chrome://extensions`.
4. Enable **Developer mode**.
5. Select **Load unpacked**.
6. Choose the root directory of `yt-chrome-plugin-frontend`.
7. Open a YouTube watch page and launch the extension.

The extension fetches the top-level comments, sends them to `/predict_with_timestamps`, and requests the three generated visualizations.

</details>

## How the pipeline works

The DVC graph in `dvc.yaml` defines these stages:

| Stage | Script | Result |
|---|---|---|
| Data ingestion | `src/data/data_ingestion.py` | `data/raw/train.csv`, `data/raw/test.csv` |
| Data preprocessing | `src/data/data_preprocessing.py` | `data/interim/train_processed.csv`, `data/interim/test_processed.csv` |
| Model building | `src/model/model_building.py` | `lgbm_model.pkl`, `tfidf_vectorizer.pkl` |
| Model evaluation | `src/model/model_evaluation.py` | MLflow metrics/artifacts and `experiment_info.json` |
| Model registration | `src/model/register_model.py` | Registered `yt_chrome_plugin_model` version |

Current model parameters are in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.20

model_building:
  max_features: 10000
  ngram_range: [1, 3]
  learning_rate: 0.08
  max_depth: 20
  n_estimators: 367
```

The notebooks document the exploratory work and model comparisons, including TF-IDF/BoW experiments and Optuna-tuned KNN, LightGBM, logistic regression, Naive Bayes, random forest, SVM, and XGBoost experiments.

## Run the Flask API

The application listens on `0.0.0.0:5000` and enables CORS for the current browser-extension workflow.

```bash
python flask_app/app.py
```

### Prediction labels

| Value | Meaning |
|---:|---|
| `1` | Positive |
| `0` | Neutral |
| `-1` | Negative |

The prediction path preprocesses each comment, transforms it with the original TF-IDF vocabulary, converts the sparse matrix into the feature-named DataFrame expected by MLflow, and predicts in batches of 256.

## API reference

All JSON endpoints expect `Content-Type: application/json`. Image endpoints return `image/png`.

### `GET /health`

Health check.

```bash
curl http://localhost:5000/health
```

### `POST /predict`

Predict a list of comments without timestamps.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"comments":["This video is excellent","I disagree with this"]}'
```

Response shape:

```json
[
  {"comment":"This video is excellent","sentiment":"1"},
  {"comment":"I disagree with this","sentiment":"-1"}
]
```

### `POST /predict_with_timestamps`

Predict comments while preserving their timestamps.

Request shape:

```json
{
  "comments": [
    {
      "text": "Great explanation",
      "timestamp": "2024-01-15T12:00:00Z",
      "authorId": "channel-id"
    }
  ]
}
```

Response shape:

```json
[
  {
    "comment": "Great explanation",
    "sentiment": "1",
    "timestamp": "2024-01-15T12:00:00Z"
  }
]
```

### `POST /generate_chart`

Create a sentiment distribution pie chart.

```json
{
  "sentiment_counts": {
    "1": 42,
    "0": 13,
    "-1": 8
  }
}
```

### `POST /generate_wordcloud`

Create a word cloud from a list of raw comments.

```json
{"comments":["The explanation was clear","The examples were useful"]}
```

### `POST /generate_trend_graph`

Create a monthly positive/neutral/negative percentage graph.

```json
{
  "sentiment_data": [
    {"timestamp":"2024-01-15T12:00:00Z","sentiment":1},
    {"timestamp":"2024-02-02T12:00:00Z","sentiment":0}
  ]
}
```

## Connect the Chrome extension

The companion frontend is intentionally kept in a separate repository:

```bash
git clone https://github.com/sahilshore/yt-chrome-plugin-frontend.git
```

Its current flow is:

1. Read the active YouTube watch URL.
2. Extract the 11-character video ID.
3. Fetch up to 500 top-level comments through the YouTube Data API.
4. Call `/predict_with_timestamps`.
5. Calculate total comments, unique commenters, average comment length, and normalized average sentiment.
6. Call `/generate_chart`, `/generate_trend_graph`, and `/generate_wordcloud`.
7. Render the images and the first 25 prediction results in the popup.

The current frontend expects:

```text
Backend: http://localhost:5000
Frontend API route: POST /predict_with_timestamps
Image routes: /generate_chart, /generate_trend_graph, /generate_wordcloud
```

## Train and reproduce the model

### Reproduce the complete pipeline

```bash
dvc repro
```

### Inspect the pipeline

```bash
dvc dag
dvc status
```

### Push or pull DVC artifacts

The repository is configured with an S3 DVC remote named `myremote`:

```bash
dvc pull
dvc push
```

Set AWS credentials through your local AWS configuration or environment variables. Do not commit credentials.

### Run the focused checks

```bash
pytest Scripts/test_load_model.py
pytest Scripts/test_model_signature.py
pytest Scripts/test_model_performance.py
pytest -v Scripts/test_flask_api.py
```

The model checks depend on the configured MLflow server, registered model, artifacts, and AWS/DVC access.

## Docker

The `Dockerfile` packages the Flask service and installs NLTK resources:

```bash
docker build -t youtube-comment-insights-engine .
docker run --rm -p 5000:5000 youtube-comment-insights-engine
```

Before building, confirm that the files copied by the Dockerfile (`tfidf_vectorizer.pkl` and the API runtime dependencies) are present in the build context. In production, prefer injecting model configuration and artifacts through a deployment secret/configuration mechanism rather than baking credentials or mutable model versions into the image.

## CI/CD

`.github/workflows/cicd.yaml` currently runs on pushes and performs:

1. Python 3.10 setup and dependency installation.
2. `dvc repro` and `dvc push`.
3. Model loading, signature, and performance tests.
4. Model promotion.
5. Flask startup and API tests.
6. Docker build and push to AWS ECR.

The workflow expects AWS credentials and access to the configured DVC, MLflow, and ECR infrastructure through GitHub Actions secrets.

## Repository guide

```text
.
├── .github/workflows/cicd.yaml   # CI/CD and deployment automation
├── .dvc/                         # DVC configuration
├── dvc.yaml                      # Reproducible ML pipeline
├── dvc.lock                      # Locked dependency/output state
├── params.yaml                   # Data and model parameters
├── src/data/                     # Ingestion and text preprocessing
├── src/model/                    # Training, evaluation, and registration
├── flask_app/app.py              # Prediction and visualization API
├── flask_app/requirements.txt    # API-focused dependencies
├── Scripts/                      # Operational checks and deployment helpers
├── notebooks/                    # EDA and experiment history
├── docs/                         # Sphinx documentation source
├── Dockerfile                    # Flask service image
├── Makefile                      # Environment, data, lint, and sync helpers
└── reports/                      # Generated report/figure locations
```

## Troubleshooting

<details>
<summary><strong>The API fails while loading the model</strong></summary>

The application loads the registered MLflow model at import time. Check:

- The MLflow tracking URI is reachable.
- The registered model `yt_chrome_plugin_model` exists.
- The configured model version exists.
- The MLflow run contains `tfidf_vectorizer.pkl`.
- The vectorizer vocabulary matches the model input schema.

</details>

<details>
<summary><strong>DVC cannot pull or reproduce the pipeline</strong></summary>

Check AWS credentials, the configured region, S3 permissions, and the remote defined in `.dvc/config`. Then run:

```bash
dvc remote list
dvc status
dvc pull -v
```

</details>

<details>
<summary><strong>The extension reports a network or CORS error</strong></summary>

Start the backend first, confirm `curl http://localhost:5000/health`, and confirm the extension uses the same host and port. For a deployed backend, update the frontend API URL and the extension manifest permissions to match the HTTPS origin.

</details>

<details>
<summary><strong>The trend graph is empty or fails</strong></summary>

The endpoint expects ISO-compatible timestamps and integer sentiment values from `-1`, `0`, and `1`. It groups records by month, so very small or same-month datasets may produce only one plotted point.

</details>

## Security and production notes

Before exposing this service publicly:

- **Rotate any YouTube API key that has ever been committed to the frontend repository.** Store it outside source control and restrict it by API, referrer, quota, and environment.
- Move MLflow URLs, model name/version, and deployment settings to environment variables.
- Restrict CORS to the deployed extension/application origin instead of allowing every origin.
- Add request-size limits, schema validation, rate limiting, authentication, and structured error responses.
- Avoid logging comment text or author identifiers unless there is a clear retention policy.
- Pin and regularly review dependencies; the root and API manifests currently contain different versions.
- Add a `.dockerignore` and use a non-root container user.
- Keep AWS credentials only in GitHub Actions secrets or an external secret manager.

## Development roadmap

The next improvements that would make the project easier to operate and extend are:

1. Replace hard-coded MLflow configuration with environment variables.
2. Add a single shared preprocessing module used by training and inference.
3. Add typed request validation for every Flask endpoint.
4. Add unit tests for preprocessing, empty inputs, malformed timestamps, and model/vectorizer compatibility.
5. Add a versioned API contract and OpenAPI specification.
6. Add a frontend production configuration instead of a localhost-only API URL.
7. Add observability for latency, prediction counts, model version, and failures without recording raw comments.
8. Use a single source of truth for runtime dependencies and artifact packaging.

## Contributing

1. Create a feature branch.
2. Keep changes focused and update the relevant documentation.
3. Run the smallest applicable test set, plus API tests when changing `flask_app/app.py`.
4. Run `dvc repro` when changing pipeline code or parameters.
5. Never commit API keys, AWS credentials, model registry credentials, or private comment data.
6. Open a pull request with the motivation, validation performed, and any model/data impact.

## License

This project includes a `LICENSE` file. See that file for the applicable terms.

## Maintainer

Built and maintained by [sahilshore](https://github.com/sahilshore).
