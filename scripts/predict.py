# scripts/predict.py

# This script serves the final, production-ready fraud detection system.
# It includes:
# - A hybrid rule-based/ML prediction model.
# - Structured JSON logging for every request.
# - API key authentication for security.
# - A tracking system using a Discord webhook.

import xgboost as xgb
import pandas as pd
import os
import uvicorn
from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import logging
from pythonjsonlogger import jsonlogger
import datetime
import requests

# --- 1. Structured Logging Setup ==================================
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
log.addHandler(logHandler)


# 2. Pydantic Model for Input Validation ==================================
class Transaction(BaseModel):
    type: str = Field(..., pattern="^(CASH_OUT|TRANSFER)$")
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0
    step: int = 1


# 3. API Key Security Setup ==================================
API_KEY = os.environ.get("API_KEY", "your-secret-key")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# 4 Tracking System Setup ==================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")  # Get webhook from environment


async def get_api_key(key: str = Security(api_key_header)):
    if key == API_KEY:
        return key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


# 5. Load The Model ==================================
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading machine learning model...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    model_path = os.path.join(PROJECT_ROOT, "models", "fraud_detection_model.xgb")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    ml_models["fraud_detection_model"] = model
    log.info("Model loaded successfully.")

    yield

    ml_models.clear()
    log.info("Model resources cleared.")


# 6. Initialize FastAPI App ==================================
app = FastAPI(
    title="Fraud Detection API",
    description="An API to predict fraudulent transactions using a hybrid rule-based/ML approach.",
    version="5.0.0",  # Final Version with Tracking!
    lifespan=lifespan
)


# 7. Define the Prediction Endpoint with Sec ==================================
@app.post("/predict", dependencies=[Depends(get_api_key)])
def predict_fraud(transaction: Transaction):
    model = ml_models["fraud_detection_model"]
    transaction_data = transaction.dict()

    # Rule-Based "Safety Net"
    is_black_hole = (
            transaction_data['oldbalanceDest'] == transaction_data['newbalanceDest'] and
            transaction_data['amount'] > 0
    )

    if is_black_hole:
        result = {
            "predicted_class": "Fraud",
            "fraud_probability": 1.0,
            "reason": "Rule-based override: Black hole transaction detected."
        }
    else:
        # Proceed with the ML model prediction
        df = pd.DataFrame([transaction_data])
        df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
        df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
        df['isBlackHoleTransaction'] = 0
        df['type_TRANSFER'] = 1 if df['type'][0] == 'TRANSFER' else 0
        df = df.drop('type', axis=1)

        training_columns = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
            'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest',
            'isBlackHoleTransaction', 'type_TRANSFER'
        ]
        df = df[training_columns]

        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)[:, 1]

        result = {
            "predicted_class": 'Fraud' if prediction[0] == 1 else 'Legitimate',
            "fraud_probability": float(prediction_proba[0]),
            "reason": "Prediction from XGBoost model."
        }

    # Log the input and the final result
    log.info({
        "request_data": transaction_data,
        "prediction_result": result
    })

    # --- Send Tracking Notification ---
    if DISCORD_WEBHOOK_URL:
        try:
            message = {
                "content": f"Prediction Made!",
                "embeds": [{
                    "title": f"Result: {result['predicted_class']}",
                    "color": 15158332 if result['predicted_class'] == 'Fraud' else 3066993,
                    # Red for Fraud, Green for Legitimate
                    "fields": [
                        {"name": "Reason", "value": result['reason'], "inline": False},
                        {"name": "Probability", "value": f"{result['fraud_probability']:.4f}", "inline": True},
                        {"name": "Type", "value": transaction_data['type'], "inline": True},
                        {"name": "Amount", "value": f"${transaction_data['amount']:,.2f}", "inline": True},
                    ],
                    "footer": {"text": f"Timestamp: {datetime.datetime.now()}"}
                }]
            }
            requests.post(DISCORD_WEBHOOK_URL, json=message, timeout=5)
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to send Discord notification: {e}")

    return result


# 8. Run the API via Uvicorn ==================================
if __name__ == "__main__":
    if not DISCORD_WEBHOOK_URL:
        log.warning("DISCORD_WEBHOOK_URL environment variable not set. Tracking is disabled.")

    log.info("Starting API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
