# ============================================================
# PROJECT 2: Credit Card Fraud Detection
# Stack: Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn
#
# WHAT IT DOES
#   1. Loads transaction data (synthetic imbalanced data out of the box)
#   2. EDA - class distribution, amount analysis (fraud vs legit)
#   3. Handles class imbalance with SMOTE (training data only!)
#   4. Trains Logistic Regression, Random Forest, XGBoost
#   5. Evaluation - Precision, Recall, F1, Precision-Recall AUC
#      (PR-AUC is more informative than ROC-AUC for rare fraud)
#
# TO USE A REAL DATASET (e.g. Kaggle "creditcard.csv", European cardholders):
#   df = pd.read_csv("creditcard.csv")   # columns: Time, V1..V28, Amount, Class
#   Just replace the "LOAD DATA" section. Target column is "Class" (1 = fraud).
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMG_DIR, exist_ok=True)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, classification_report,
                             precision_recall_curve, auc)

RANDOM_STATE = 42

# ---------------- 1. LOAD DATA ----------------
# --- Option A: synthetic imbalanced data (~2% fraud) ---
def make_fraud_data(n=10000, fraud_rate=0.02, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def block(k, fraud):
        # Legit and fraud transactions come from different (but overlapping) distributions
        shift = 2.5 if fraud else 0.0
        data = {f"V{i}": rng.normal(shift if i in (1, 3, 4) else 0, 1.5, k) for i in range(1, 9)}
        data["Amount"] = np.round(rng.lognormal(mean=4.5 if fraud else 3.2, sigma=1.0, size=k), 2)
        data["Time"] = rng.integers(0, 172800, k)  # seconds in 2 days
        data["Class"] = int(fraud)
        return pd.DataFrame(data)

    df = pd.concat([block(n_legit, False), block(n_fraud, True)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

df = make_fraud_data()
# --- Option B: real data ---
# df = pd.read_csv("creditcard.csv")   # target column must be "Class" (1 = fraud)

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["Class"].value_counts())
print("\nFraud rate:", round(df["Class"].mean() * 100, 2), "%")
print("\nAvg amount (legit vs fraud):\n", df.groupby("Class")["Amount"].mean().round(2))

# ---------------- 2. EDA ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["Class"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"])
axes[0].set_title("Class Distribution (0 = Legit, 1 = Fraud)"); axes[0].set_xticklabels(["Legit", "Fraud"], rotation=0)
for label, color in [(0, "steelblue"), (1, "tomato")]:
    axes[1].hist(np.log1p(df.loc[df["Class"] == label, "Amount"]), bins=40, alpha=0.6,
                 label="Fraud" if label else "Legit", color=color)
axes[1].set_title("Amount Distribution (log scale)"); axes[1].legend(); axes[1].set_xlabel("log1p(Amount)")
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "fraud_eda.png")); plt.close()

# ---------------- 3. SPLIT + SCALE ----------------
X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# ---------------- 4. SMOTE (train only - never resample the test set!) ----------------
print("\nBefore SMOTE:", y_train.value_counts().to_dict())
X_train_sm, y_train_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
print("After SMOTE: ", pd.Series(y_train_sm).value_counts().to_dict())

# ---------------- 5. TRAIN & EVALUATE 3 MODELS ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                         eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
}

plt.figure(figsize=(7, 6))
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | PR-AUC: {pr_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(rec, prec, label=f"{name} (AUC = {auc(rec, prec):.3f})")

plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves - Fraud Detection")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "fraud_precision_recall.png")); plt.close()

print("\nSaved plots: fraud_eda.png, fraud_precision_recall.png")
print("\nNOTE: for fraud, recall = catching real fraud; precision = not blocking legit customers.")
print("Tune the decision threshold (default 0.5) based on business cost of each error type.")
