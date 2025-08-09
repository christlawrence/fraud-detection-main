# scripts/monitor_model.py
#
# Objective:
# To simulate a model monitoring process. This script evaluates the model
# on a hold-out test set and logs the key performance metrics. In a real
# system, this would run periodically (e.g., daily) to detect model drift.

import xgboost as xgb
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime


def monitor_performance():
    """
    Evaluates the model on the test set and logs performance metrics.
    """
    print("Starting model performance monitoring...")

    # --- 1. Load Model and Test Data ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "PS_20174392719_1491204439457_log.csv")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    MODEL_PATH = os.path.join(MODELS_DIR, "fraud_detection_model.xgb")

    df = pd.read_csv(DATA_PATH)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print("Model and data loaded.")

    # --- 2. Recreate the Test Set (ensuring consistency) ---
    df_filtered = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()
    df_filtered["errorBalanceOrig"] = (df_filtered["newbalanceOrig"] + df_filtered["amount"]) - df_filtered[
        "oldbalanceOrg"]
    df_filtered["errorBalanceDest"] = (df_filtered["oldbalanceDest"] + df_filtered["amount"]) - df_filtered[
        "newbalanceDest"]
    df_filtered['isBlackHoleTransaction'] = (
                (df_filtered['oldbalanceDest'] == df_filtered['newbalanceDest']) & (df_filtered['amount'] > 0)).astype(
        int)
    df_filtered = pd.get_dummies(df_filtered, columns=["type"], prefix="type", drop_first=True)
    y = df_filtered["isFraud"]
    X = df_filtered.drop(["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], axis=1)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # --- 3. Make Predictions and Calculate Metrics ---
    print("Calculating performance metrics...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # --- 4. Log the Metrics ---
    # In a real system, this would write to a database or a monitoring service.
    # Here, we'll just print it and save to a JSON file.
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc
    }

    print("\n--- Model Performance Metrics ---")
    print(json.dumps(metrics, indent=4))

    # --- 5. Simple Alerting Logic ---
    # Example of a simple check for performance degradation.
    if precision < 0.95:
        print("\nALERT: Model precision has dropped below the 95% threshold!")
    if recall < 0.98:
        print("\nALERT: Model recall has dropped below the 98% threshold!")

    # Save to a log file
    log_path = os.path.join(PROJECT_ROOT, "reports", "monitoring_log.json")
    with open(log_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"\nMetrics appended to {log_path}")
    print("--- Monitoring complete ---")


if __name__ == "__main__":
    monitor_performance()
