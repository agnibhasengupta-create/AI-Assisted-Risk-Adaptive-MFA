"""
AI-Assisted Risk-Adaptive MFA
--------------------------------
Synthetic authentication experiment.

This program:
1. Generates synthetic authentication events.
2. Creates a ground-truth risk label.
3. Trains a Decision Tree classifier.
4. Evaluates the classifier on unseen test data.
5. Compares AI predictions with a rule-based baseline.
6. Calculates security and usability metrics.
7. Saves the dataset and results for graph generation.

No real accounts, credentials, users, or attack infrastructure are used.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

RANDOM_SEED = 42
N_SAMPLES = 10000

np.random.seed(RANDOM_SEED)


# ============================================================
# 2. GENERATE SYNTHETIC AUTHENTICATION DATA
# ============================================================

def generate_dataset(n):

    data = pd.DataFrame({

        # 1 = known device, 0 = unknown device
        "known_device":
            np.random.binomial(1, 0.75, n),

        # 1 = unusual location
        "location_anomaly":
            np.random.binomial(1, 0.15, n),

        # 1 = unusual login time
        "time_anomaly":
            np.random.binomial(1, 0.20, n),

        # 1 = suspicious network
        "network_risk":
            np.random.binomial(1, 0.12, n),

        # 1 = abnormal login frequency
        "frequency_anomaly":
            np.random.binomial(1, 0.10, n),

        # 1 = known compromise indicator
        "compromise_indicator":
            np.random.binomial(1, 0.04, n)
    })

    # --------------------------------------------------------
    # Introduce realistic relationships between variables
    # --------------------------------------------------------

    # Unknown devices are somewhat more likely to have
    # location anomalies.
    for i in range(n):

        if data.loc[i, "known_device"] == 0:
            if np.random.random() < 0.30:
                data.loc[i, "location_anomaly"] = 1

        # Suspicious networks increase probability of
        # abnormal login frequency.
        if data.loc[i, "network_risk"] == 1:
            if np.random.random() < 0.35:
                data.loc[i, "frequency_anomaly"] = 1

    # --------------------------------------------------------
    # Ground-truth risk score
    # --------------------------------------------------------

    score = (
        (1 - data["known_device"]) * 2
        + data["location_anomaly"] * 2
        + data["time_anomaly"] * 1
        + data["network_risk"] * 2
        + data["frequency_anomaly"] * 2
        + data["compromise_indicator"] * 4
    )

    # --------------------------------------------------------
    # Add controlled noise.
    #
    # This is important:
    # real-world risk is not perfectly deterministic.
    # --------------------------------------------------------

    noise = np.random.normal(0, 0.8, n)

    noisy_score = score + noise

    # --------------------------------------------------------
    # Convert score into risk classes
    # --------------------------------------------------------

    risk_level = np.select(
        [
            noisy_score <= 2.5,
            noisy_score <= 5.5
        ],
        [
            "Low",
            "Medium"
        ],
        default="High"
    )

    data["risk_score"] = noisy_score.round(2)
    data["risk_level"] = risk_level

    return data


# ============================================================
# 3. GENERATE DATASET
# ============================================================

df = generate_dataset(N_SAMPLES)

print("\n==============================")
print("SYNTHETIC DATASET")
print("==============================")

print(df.head())

print("\nNumber of authentication events:")
print(len(df))

print("\nRisk distribution:")
print(df["risk_level"].value_counts())


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

features = [
    "known_device",
    "location_anomaly",
    "time_anomaly",
    "network_risk",
    "frequency_anomaly",
    "compromise_indicator"
]

X = df[features]

y = df["risk_level"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)


# ============================================================
# 6. TRAIN AI MODEL
# ============================================================

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=RANDOM_SEED
)

model.fit(X_train, y_train)


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. AI MODEL METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)


# ============================================================
# 9. PRINT AI RESULTS
# ============================================================

print("\n==============================")
print("AI MODEL RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix")
print("Rows = Actual")
print("Columns = Predicted")

print(pd.DataFrame(
    cm,
    index=["Actual Low", "Actual Medium", "Actual High"],
    columns=["Predicted Low", "Predicted Medium", "Predicted High"]
))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 10. RULE-BASED BASELINE
# ============================================================

def rule_based_prediction(row):

    score = (
        (1 - row["known_device"]) * 2
        + row["location_anomaly"] * 2
        + row["time_anomaly"] * 1
        + row["network_risk"] * 2
        + row["frequency_anomaly"] * 2
        + row["compromise_indicator"] * 4
    )

    if score <= 2:
        return "Low"

    elif score <= 5:
        return "Medium"

    else:
        return "High"


baseline_predictions = X_test.apply(
    rule_based_prediction,
    axis=1
)


# ============================================================
# 11. BASELINE METRICS
# ============================================================

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_precision = precision_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)


print("\n==============================")
print("RULE-BASED BASELINE")
print("==============================")

print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1-score : {baseline_f1:.4f}")


# ============================================================
# 12. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# ============================================================

# We define "High" as the security-critical class.

actual_high = (y_test == "High")
predicted_high = (y_pred == "High")

true_positive = np.sum(
    actual_high & predicted_high
)

false_positive = np.sum(
    ~actual_high & predicted_high
)

false_negative = np.sum(
    actual_high & ~predicted_high
)

true_negative = np.sum(
    ~actual_high & ~predicted_high
)


print("\n==============================")
print("HIGH-RISK DETECTION")
print("==============================")

print(f"True Positives : {true_positive}")
print(f"False Positives: {false_positive}")
print(f"False Negatives: {false_negative}")
print(f"True Negatives : {true_negative}")


# ============================================================
# 13. HIGH-RISK RECALL
# ============================================================

high_risk_recall = recall_score(
    y_test,
    y_pred,
    labels=["High"],
    average="macro",
    zero_division=0
)

print(f"\nHigh-risk recall: {high_risk_recall:.4f}")


# ============================================================
# 14. AUTHENTICATION FRICTION
# ============================================================

# Our adaptive policy:
#
# Low    -> Standard authentication
# Medium -> Additional MFA
# High   -> Phishing-resistant MFA
#
# For usability analysis we focus on legitimate
# (non-high-risk) events that are unnecessarily escalated
# to the strongest authentication level.

legitimate_events = (y_test != "High")

unnecessarily_escalated = (
    legitimate_events & (y_pred == "High")
)

friction_rate = (
    unnecessarily_escalated.sum()
    / legitimate_events.sum()
)

print("\n==============================")
print("USABILITY / FRICTION")
print("==============================")

print(
    f"Unnecessary high-risk escalations: "
    f"{unnecessarily_escalated.sum()}"
)

print(
    f"Authentication friction rate: "
    f"{friction_rate:.4f}"
)


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(importance)


# ============================================================
# 16. SAVE DATA FOR GRAPH GENERATOR
# ============================================================

df.to_csv(
    "synthetic_authentication_dataset.csv",
    index=False
)

results = pd.DataFrame({
    "Model": [
        "Rule-based",
        "AI Decision Tree"
    ],

    "Accuracy": [
        baseline_accuracy,
        accuracy
    ],

    "Precision": [
        baseline_precision,
        precision
    ],

    "Recall": [
        baseline_recall,
        recall
    ],

    "F1": [
        baseline_f1,
        f1
    ]
})

results.to_csv(
    "model_comparison.csv",
    index=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n==============================")
print("FILES CREATED")
print("==============================")

print("1. synthetic_authentication_dataset.csv")
print("2. model_comparison.csv")
print("3. feature_importance.csv")

print("\nExperiment completed.")
