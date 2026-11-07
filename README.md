# Credit Card Fraud Detection – End-to-End ML Engineering Project

An end-to-end machine learning project that detects potentially fraudulent credit card transactions using **XGBoost**, explains model predictions using **SHAP**, and serves predictions through a **FastAPI REST API** packaged with **Docker**.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem. Fraudulent transactions represent a very small proportion of all transactions, making accurate detection challenging.

This project develops a machine learning solution to identify fraudulent transactions while balancing fraud detection (recall) and false alarms (precision).

The trained model is integrated into a REST API, enabling transaction-level fraud predictions.

## Key Features

- Exploratory Data Analysis and data quality checks
- Leakage-aware, group-based train, validation, and test splitting
- Logistic Regression baseline and XGBoost model comparison
- Classification threshold optimization
- SHAP-based global and individual prediction explanations
- Model serialization and reload verification
- FastAPI prediction service
- Pydantic request validation
- Automated API testing with Pytest
- Docker containerization and deployment

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| API Development | FastAPI, Pydantic |
| Model Serving | Uvicorn |
| Testing | Pytest, HTTPX |
| Deployment | Docker |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Dataset

**Credit Card Fraud Detection Dataset**

Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset characteristics:

- 284,807 transactions
- 492 fraudulent transactions
- 284,315 legitimate transactions
- 30 input features
- 1 target column (`Class`)

Target variable:

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

The dataset contains anonymized PCA-transformed features (`V1`–`V28`), along with transaction time and amount.

The raw dataset is excluded from this GitHub repository. Download it from Kaggle and place `creditcard.csv` inside `data/raw/` to reproduce the notebook workflow.

## Machine Learning Workflow

### 1. Data Understanding and Preprocessing

- Inspected dataset dimensions, data types, and missing values.
- Analyzed class imbalance and transaction distributions.
- Investigated duplicate records and duplicate groups.
- Used group-aware splitting to prevent identical feature records from appearing across training, validation, and test sets.
- Scaled `Amount` and `Time` for Logistic Regression.
- Kept original feature values for XGBoost.

### 2. Model Development

Two models were evaluated:

- Logistic Regression – baseline classification model.
- XGBoost – gradient-boosted tree-based classification model.

XGBoost achieved stronger fraud detection performance on the validation dataset.

### 3. Threshold Optimization

The default classification threshold of 0.50 was evaluated against alternative thresholds.

A threshold of **0.20** was selected using validation results to balance precision and recall through the F1-score.

The test dataset was kept separate from model selection and threshold tuning.

### 4. Explainable AI with SHAP

SHAP was used to understand:

- Overall feature importance.
- How individual features influence fraud predictions.
- Why selected transactions receive high or low fraud scores.

Important model features included `V14`, `V4`, `V12`, `V8`, and `Amount`.

Note: `V1`–`V28` are anonymized PCA-transformed features and cannot be directly interpreted as original transaction attributes.

## Model Performance

### Final Held-Out Test Results

The final XGBoost model was evaluated on an untouched test dataset containing 56,962 transactions, including 99 fraudulent transactions.

| Metric | Result |
|---|---:|
| Fraud Precision | 92.13% |
| Fraud Recall | 82.83% |
| Fraud F1-Score | 87.23% |
| PR-AUC | 0.8611 |
| ROC-AUC | 0.9887 |
| Accuracy | 99.96% |

Classification threshold: **0.20**

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,856 | 7 |
| Actual Fraud | 17 | 82 |

The model correctly detected 82 of the 99 fraudulent test transactions, with 7 legitimate transactions incorrectly flagged as fraud.

PR-AUC is particularly important here because the dataset is highly imbalanced.

## Project Structure

```text
fraud-detection-ml/
│
├── models/
│   ├── fraud_xgboost.json
│   └── model_metadata.json
│
├── notebooks/
│   └── 01_fraud_detection.ipynb
│
├── src/
│   └── api.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

The dataset and local virtual environment are excluded from version control.

## API Development with FastAPI

The trained XGBoost model is served through a REST API developed using FastAPI.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API health check |
| POST | `/predict` | Predict transaction fraud probability |

### Prediction Response

The prediction endpoint returns:

- `fraud_probability` – Model-estimated fraud probability.
- `prediction` – 0 or 1.
- `result` – Legitimate or Fraud.
- `risk_level` – Low, Medium, or High.
- `threshold` – Configured classification threshold.

Risk levels are application-defined labels based on prediction probability, not calibrated financial risk categories.

### Interactive API Documentation

After starting the API, access:

`http://localhost:8000/docs`

Swagger UI allows users to submit transaction features and inspect prediction responses.

The API expects all 30 model input features.

## Docker Deployment

The application is containerized using Docker to provide a consistent runtime environment.

### Build Docker Image

```bash
docker build -t fraud-detection-api:1.0 .
```

### Run Docker Container

```bash
docker run -d --name fraud-api -p 8000:8000 fraud-detection-api:1.0
```

### Check Running Container

```bash
docker ps
```

### View Container Logs

```bash
docker logs fraud-api
```

### Access API Documentation

Open:

`http://localhost:8000/docs`

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Kushala-D/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
```

Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install Runtime Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
python -m uvicorn src.api:app --reload
```

Visit `http://localhost:8000/docs`.

## Automated Testing

API tests were implemented using Pytest and FastAPI TestClient.

Test coverage includes:

- API health check.
- Fraud transaction prediction.
- Legitimate transaction prediction.
- Missing input validation.
- Negative transaction time and amount validation.

**Latest test result: 5 passed.**

Run the tests locally after installing development dependencies:

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Limitations and Future Improvements

- The dataset is historical and anonymized; results do not establish live production performance.
- The anonymized PCA features limit direct business interpretation.
- The current risk labels are heuristic and are not calibrated risk estimates.
- The model would require fresh data, calibration, drift monitoring, and operational validation before real-world deployment.

Potential improvements:

- Model monitoring and data drift detection.
- Prediction logging and performance tracking.
- Model versioning and experiment tracking with MLflow.
- CI/CD automation using GitHub Actions.
- Cloud deployment and scalable inference.
- Model calibration and cost-sensitive threshold selection.

## Author

**Kushala D**

GitHub: [Kushala-D](https://github.com/Kushala-D)

Project Repository: [fraud-detection-ml](https://github.com/Kushala-D/fraud-detection-ml)