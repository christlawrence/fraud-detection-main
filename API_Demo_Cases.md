# API Test Scenarios & Demonstration Cases

This document provides a comprehensive set of test cases for the Fraud Detection API. The scenarios are designed to showcase the accuracy, intelligence, and robustness of the final hybrid system.

Each scenario includes the JSON payload to send to the `/predict` endpoint and a brief narrative explaining the expected outcome and its significance. These can be tested using the interactive API documentation available at `http://127.0.0.1:8000/docs` when the server is running.

-----

## Scenario Set 1: The Baseline (Validating Legitimate Activity)

**Goal:** Prove the system is safe for normal customers and can correctly distinguish between suspicious and merely unusual legitimate behavior.

### Test 1.1: A Standard CASH\_OUT

**Description:** A completely normal cash-out where all balances are arithmetically correct and the amount is reasonable.

**JSON Payload:**

```json
{
  "type": "CASH_OUT",
  "amount": 5200.50,
  "oldbalanceOrg": 78000,
  "newbalanceOrig": 72799.50,
  "oldbalanceDest": 25000,
  "newbalanceDest": 30200.50
}
```

**Expected Result:** `Predicted Class: Legitimate`

**Analysis:** The system correctly passes this through the ML model, which identifies it as legitimate, ensuring a good customer experience.

### Test 1.2: The "Legitimate but Unusual" Account Drain

**Description:** A user drains their account with a large transfer, which can be a fraud indicator. However, the recipient's balance increases correctly.

**JSON Payload:**

```json
{
  "type": "TRANSFER",
  "amount": 500000,
  "oldbalanceOrg": 500000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 25000,
  "newbalanceDest": 525000
}
```

**Expected Result:** `Predicted Class: Legitimate`

**Analysis:** The model is smart enough to see that even though the originator's balance went to zero, the transaction itself is arithmetically sound, and it correctly classifies it as legitimate. This prevents false positives on valid, large transactions.

-----

## Scenario Set 2: The Hybrid System in Action

**Goal:** Show how the layered system catches different types of fraud, from blatant to subtle, highlighting both the rule-based safety net and the intelligence of the ML model.

### Test 2.1: The "Black Hole" TRANSFER (Caught by the Rule)

**Description:** A subtle case where money is sent, but the recipient's balance does not change. This is an undeniable sign of fraud that a model might miss.

**JSON Payload:**

```json
{
  "type": "TRANSFER",
  "amount": 1500,
  "oldbalanceOrg": 20000,
  "newbalanceOrig": 18500,
  "oldbalanceDest": 100,
  "newbalanceDest": 100
}
```

**Expected Result:** `Predicted Class: Fraud, Reason: Rule-based override due to black hole transaction.`

**Analysis:** The hybrid system's rule-based safety net detects that the recipient's balance didn't change. This triggers the 'black hole' rule, and the transaction is immediately flagged as fraud, guaranteeing this critical pattern is never missed.

### Test 2.2: The "Anomalous Zero-Balance" CASH\_OUT (Caught by the Model)

**Description:** A transaction that drains an account to a recipient who had a zero balance and still has a zero balance after the transaction.

**JSON Payload:**

```json
{
  "type": "CASH_OUT",
  "amount": 10000,
  "oldbalanceOrg": 10000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 0
}
```

**Expected Result:** `Predicted Class: Fraud, Reason: Prediction from XGBoost model.`

**Analysis:** This case passes the specific rule check but is still highly suspicious. The tuned XGBoost model, sensitive to nuanced patterns, recognizes the combination of features as a high-risk pattern and correctly flags it as fraud.

### Test 2.3: The "Probing" Fraud Attempt

**Description:** A very small transaction sent to a zero-balance account, a common pattern for fraudsters testing stolen account details before attempting a larger theft.

**JSON Payload:**

```json
{
  "type": "CASH_OUT",
  "amount": 1.00,
  "oldbalanceOrg": 100.00,
  "newbalanceOrig": 99.00,
  "oldbalanceDest": 0,
  "newbalanceDest": 0
}
```

**Expected Result:** `Predicted Class: Fraud`

**Analysis:** While the amount is insignificant, the pattern is something our ML model has learned to associate with fraudulent probing activity, and it correctly flags it as high-risk.

-----

## Scenario Set 3: Production Readiness & Robustness

**Goal:** Show that the service is professional and stable enough to handle real-world errors gracefully.

### Test 3.1: Invalid Input

**Description:** A request containing data that violates the defined Pydantic model constraints (e.g., a negative amount).

**JSON Payload:**

```json
{
  "type": "CASH_OUT",
  "amount": -100,
  "oldbalanceOrg": 1000,
  "newbalanceOrig": 1100,
  "oldbalanceDest": 50,
  "newbalanceDest": 50
}
```

**Expected Result:** A `422 Unprocessable Entity` HTTP error.

**Analysis:** The API does not crash. Thanks to the Pydantic validation layer, it immediately rejects the invalid request with a clear error message, ensuring the system's stability and reliability.