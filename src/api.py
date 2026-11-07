
from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Locate model files
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fraud_xgboost.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# --------------------------------------------------
# 2. Load model metadata
# --------------------------------------------------

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

FEATURES = metadata["features"]
THRESHOLD = metadata["threshold"]


# --------------------------------------------------
# 3. Load trained XGBoost model
# --------------------------------------------------

model = XGBClassifier()
model.load_model(str(MODEL_PATH))


# --------------------------------------------------
# 4. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered credit card fraud detection service",
    version="1.0.0"
)


# --------------------------------------------------
# 5. Define input validation schema
# --------------------------------------------------

class TransactionInput(BaseModel):
    Time: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


# --------------------------------------------------
# 6. Health-check endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "model": metadata["model_name"]
    }


# --------------------------------------------------
# 7. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: TransactionInput):

    try:
        # Convert input into a DataFrame
        transaction_data = transaction.model_dump()

        input_df = pd.DataFrame(
            [transaction_data],
            columns=FEATURES
        )

        # Predict fraud probability
        fraud_probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        # Apply saved threshold
        prediction = int(
            fraud_probability >= THRESHOLD
        )

        # Determine risk category
        if fraud_probability >= 0.80:
            risk_level = "High"
        elif fraud_probability >= THRESHOLD:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "fraud_probability": round(fraud_probability, 6),
            "prediction": prediction,
            "result": "Fraud" if prediction == 1 else "Legitimate",
            "risk_level": risk_level,
            "threshold": THRESHOLD
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
