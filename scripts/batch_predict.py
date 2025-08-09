# scripts/batch_predict.py

# To score a batch of transactions from a CSV file and save the results.
# This is useful for offline analysis, rescoring historical data, etc.

import xgboost as xgb
import pandas as pd
import os
import argparse

def batch_predict(input_path: str, output_path: str):
    print("Starting batch prediction...")

    # 1. Load Model and Data ==================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    model_path = os.path.join(PROJECT_ROOT, "models", "fraud_detection_model.xgb")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("Model loaded successfully.")

    try:
        df = pd.read_csv(input_path)
        print(f"Data loaded from {input_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input data file not found at {input_path}")
        return

    # 2. Feature Engineering (must be identical to training) ==================================
    print("Performing feature engineering...")
    df_processed = df.copy()
    df_processed["errorBalanceOrig"] = (df_processed["newbalanceOrig"] + df_processed["amount"]) - df_processed[
        "oldbalanceOrg"]
    df_processed["errorBalanceDest"] = (df_processed["oldbalanceDest"] + df_processed["amount"]) - df_processed[
        "newbalanceDest"]
    df_processed['isBlackHoleTransaction'] = ((df_processed['oldbalanceDest'] == df_processed['newbalanceDest']) & (
                df_processed['amount'] > 0)).astype(int)
    df_processed = pd.get_dummies(df_processed, columns=["type"], prefix="type", drop_first=True, dummy_na=False)

    # Ensure type_TRANSFER column exists if no transfers were in the batch
    if 'type_TRANSFER' not in df_processed.columns:
        df_processed['type_TRANSFER'] = 0

    training_columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
        'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest',
        'isBlackHoleTransaction', 'type_TRANSFER'
    ]
    df_processed = df_processed[training_columns]

    # 3. Make Predictions ==================================
    print("Making predictions...")
    predictions = model.predict(df_processed)
    prediction_probas = model.predict_proba(df_processed)[:, 1]

    # 4. Save Results ==================================
    df['predicted_class'] = predictions
    df['fraud_probability'] = prediction_probas

    # Re-map class from 0/1 to Legitimate/Fraud for clarity
    df['predicted_class'] = df['predicted_class'].map({0: 'Legitimate', 1: 'Fraud'})

    df.to_csv(output_path, index=False)
    print(f"Predictions saved successfully to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict fraud from a CSV file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV file with predictions.")
    args = parser.parse_args()

    batch_predict(args.input, args.output)
