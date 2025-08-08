# scripts/predict.py

# This script serves the final fraud detection system.
# It uses a hybrid approach, combining a hard-coded rule for specific fraud patterns with the pre-trained XGBoost model for more nuanced cases.

import xgboost as xgb
import pandas as pd
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
# import numpy as np


# 1. Pydantic Model for Input Validation ==================================================
class Transaction(BaseModel):
    type: str = Field(..., pattern="^(CASH_OUT|TRANSFER)$")
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0
    step: int = 1


# 2. Load The Model ==================================================
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading machine learning model...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    model_path = os.path.join(PROJECT_ROOT, "models", "fraud_detection_model.xgb")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    ml_models["fraud_detection_model"] = model
    print("Model loaded successfully.")

    yield

    ml_models.clear()
    print("Model resources cleared.")


# 3. Initialize FastAPI App ==================================================
app = FastAPI(
    title="Fraud Detection API by Christopher Manlongat",
    description="An API to predict fraudulent transactions using a hybrid rule-based/ML approach.",
    version="4.0.0",  # Final Version!
    lifespan=lifespan
)


# 4. Define the Prediction Endpoint ==================================================
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    model = ml_models["fraud_detection_model"]
    transaction_data = transaction.dict()

    # --- Hybrid System Logic ---
    # Rule-Based "Safety Net": Check for the undeniable "black hole" pattern first.
    is_black_hole = (
            transaction_data['oldbalanceDest'] == transaction_data['newbalanceDest'] and
            transaction_data['amount'] > 0
    )

    if is_black_hole:
        return {
            "predicted_class": "Fraud",
            "fraud_probability": 1.0,
            "reason": "Rule-based override: Black hole transaction detected."
        }

    # If the rule doesn't trigger, proceed with the ML model prediction.
    df = pd.DataFrame([transaction_data])

    # --- Preprocessing for the ML Model ---
    df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df['isBlackHoleTransaction'] = 0  # It's 0 because the rule above would have caught it if it were 1

    if df['type'][0] == 'TRANSFER':
        df['type_TRANSFER'] = 1
    else:
        df['type_TRANSFER'] = 0

    df = df.drop('type', axis=1)

    # Ensure column order and presence match the final training script
    training_columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
        'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest',
        'isBlackHoleTransaction', 'type_TRANSFER'
    ]
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]

    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)[:, 1]

    predicted_class = 'Fraud' if prediction[0] == 1 else 'Legitimate'
    fraud_probability = float(prediction_proba[0])

    return {
        "predicted_class": predicted_class,
        "fraud_probability": fraud_probability,
        "reason": "Prediction from XGBoost model." # included reason
    }

# 5. Run the API via Uvicorn ==================================================
if __name__ == "__main__":
    print("Starting API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
