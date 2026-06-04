"""
TwinVEIL — Module 2, Step 3: logistic-regression baseline.

Reads  data/processed/labelled.csv
Writes models/lr_baseline.pkl
       models/lr_weights.npy   (shape (4,), float64)
       models/lr_bias.npy      (scalar, float64)
       reports/step3_metrics.txt
       reports/confusion_matrix.png
       reports/step3_verification.txt
Also copies the model artefacts into twinveil-server/models/ for Step 4.
"""
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

LABELLED = "data/processed/labelled.csv"
FEATURES = ["speed_norm", "longitudinal_accel_norm",
            "steering_angle_norm", "brake_pressure_norm"]
TARGETS = {"Accuracy": 0.90, "Precision": 0.75, "Recall": 0.80,
           "F1": 0.78, "ROC-AUC": 0.90}


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv(LABELLED)
    X = df[FEATURES].values
    y = df["label"].values

    # Same split as §2.4 (deterministic, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------- §3.2 Training ----------------
    clf = LogisticRegression(
        solver="lbfgs", max_iter=500, C=1.0,
        class_weight="balanced", random_state=42,
    )
    clf.fit(X_train, y_train)
    print("Coefficients (w):", clf.coef_[0])
    print("Intercept  (b):", clf.intercept_[0])

    joblib.dump(clf, "models/lr_baseline.pkl")
    np.save("models/lr_weights.npy", clf.coef_[0].astype(np.float64))   # (4,)
    np.save("models/lr_bias.npy", np.float64(clf.intercept_[0]))        # scalar

    # ---------------- §3.3 Evaluation ----------------
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }
    report = classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    lines = ["=== §3.3 Evaluation (held-out test set) ==="]
    for k, v in metrics.items():
        ok = "PASS" if v >= TARGETS[k] else "FAIL"
        lines.append(f"{k:10s}: {v:.4f}   (target >= {TARGETS[k]:.2f})  [{ok}]")
    lines.append("\nConfusion matrix [rows=true, cols=pred] (Normal, Anomaly):")
    lines.append(np.array2string(cm))
    tn, fp, fn, tp = cm.ravel()
    lines.append(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    lines.append("\n" + report)
    text = "\n".join(lines)
    with open("reports/step3_metrics.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.title("TwinVEIL LR Baseline — Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    plt.clf()

    # ---------------- §3.4 Weight-export verification ----------------
    w = np.load("models/lr_weights.npy")
    b = float(np.load("models/lr_bias.npy"))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    vlines = ["=== §3.4 Weight-export verification (manual vs sklearn) ==="]
    all_ok = True
    for i in range(5):
        z_manual = X_test[i] @ w + b
        score_manual = sigmoid(z_manual)
        score_sklearn = clf.predict_proba(X_test[i:i + 1])[0, 1]
        match = abs(score_manual - score_sklearn) < 1e-5
        all_ok &= match
        vlines.append(f"Sample {i}: manual={score_manual:.6f}  sklearn={score_sklearn:.6f}  match={match}")
    vlines.append(f"\nAll 5 match: {all_ok}")
    vtext = "\n".join(vlines)
    with open("reports/step3_verification.txt", "w", encoding="utf-8") as f:
        f.write(vtext + "\n")
    print("\n" + vtext)

    # ---------------- Stage artefacts for Step 4 ----------------
    os.makedirs("twinveil-server/models", exist_ok=True)
    for fn in ["lr_weights.npy", "lr_bias.npy"]:
        shutil.copy(f"models/{fn}", f"twinveil-server/models/{fn}")
    shutil.copy("models/scaler_bounds.json", "twinveil-server/models/scaler_bounds.json")
    print("\nCopied model artefacts into twinveil-server/models/")

    overall = all(v >= TARGETS[k] for k, v in metrics.items())
    print(f"\nAll metric targets met: {overall}")


if __name__ == "__main__":
    main()
