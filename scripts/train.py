# scripts/train.py

# This script trains the final fraud detection model.
# It includes all engineered features and tuned hyperparameters for maximum performance.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
import xgboost as xgb
import os
#import numpy as np

print("Starting the model training script...")

# --- Paths---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 1. Load Data =================================
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "PS_20174392719_1491204439457_log.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")

# 2. Final Feature Engineering & Preprocessing =================================
print("Performing final, comprehensive feature engineering...")

df_filtered = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()

# --- Combine all features for maximum signal ---
df_filtered["errorBalanceOrig"] = (df_filtered["newbalanceOrig"] + df_filtered["amount"]) - df_filtered["oldbalanceOrg"]
df_filtered["errorBalanceDest"] = (df_filtered["oldbalanceDest"] + df_filtered["amount"]) - df_filtered["newbalanceDest"]
df_filtered['isBlackHoleTransaction'] = (
    (df_filtered['oldbalanceDest'] == df_filtered['newbalanceDest']) &
    (df_filtered['amount'] > 0)
).astype(int)

df_filtered = pd.get_dummies(df_filtered, columns=["type"], prefix="type", drop_first=True)

y = df_filtered["isFraud"]
X = df_filtered.drop(["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

# 3. Train-Test Split =================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

# 4. Model Training w/ Tuned Hyperparameters =================================
print("Training the final XGBoost model with tuned hyperparameters...")
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    seed=42,
    # --- Hyperparameter Tuning ---
    n_estimators=300,  # More trees
    max_depth=7,   # Deeper trees to learn complex interactions
    #learning_rate
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train, y_train)
print("Model training complete.")

# 5. Save the Model =================================
model_path = os.path.join(MODELS_DIR, "fraud_detection_model.xgb")
model.save_model(model_path)
print(f"Model saved to {model_path}")

# 6. Model Evaluation =================================
print("Evaluating the model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Needed for PR Curve

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
plt.title("Confusion Matrix", fontsize=16)
plt.ylabel("Actual")
plt.xlabel("Predicted")
cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"Confusion matrix saved to {cm_path}")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auprc = auc(recall, precision)
plt.figure(figsize=(10, 7))
plt.plot(recall, precision, color="blue", label=f"AUPRC = {auprc:.4f}")
plt.title("Precision-Recall Curve", fontsize=16)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
pr_curve_path = os.path.join(REPORTS_DIR, "precision_recall_curve.png")
plt.savefig(pr_curve_path)
print(f"Precision-Recall curve saved to {pr_curve_path}")
plt.close()

print("\n--- Train Script Finished ---")
