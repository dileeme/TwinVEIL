"""
TwinVEIL — Module 2, Step 2: feature engineering, anomaly injection,
normalisation, and train/test split.

Reads  data/processed/cleaned.csv
Writes data/processed/labelled.csv      (label + 4 *_norm columns)
       models/scaler_bounds.json         (fixed domain bounds)
       reports/step2_injection.txt       (rate, class counts, final thresholds)
       reports/step2_normalisation.txt   (describe() of *_norm columns)
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IN = "data/processed/cleaned.csv"
LABELLED = "data/processed/labelled.csv"
FEATURES = ["speed_norm", "longitudinal_accel_norm",
            "steering_angle_norm", "brake_pressure_norm"]

# §2.3 fixed domain bounds (NOT data-driven, so the same transform is reusable
# at inference time without recomputing from the dataset).
BOUNDS = {
    "speed":              (0,    40),
    "longitudinal_accel": (-8,    5),
    "steering_angle":     (-500, 500),
    "brake_pressure":     (0,     1),
}


def inject_labels(df):
    """§2.2 anomaly-label injection with the spec's threshold-tuning logic."""
    r1_speed, r1_accel = 15.0, -3.5   # hard-brake-at-speed
    r2_steer, r2_accel = 200.0, 0.5   # steering oscillation
    r3_accel, r3_speed = 3.0, 5.0     # acceleration spike

    def apply(r1_speed, r1_accel):
        lab = pd.Series(0, index=df.index)
        lab[(df["speed"] > r1_speed) & (df["longitudinal_accel"] < r1_accel)] = 1
        lab[(df["steering_angle"].abs() > r2_steer) & (df["longitudinal_accel"].abs() < r2_accel)] = 1
        lab[(df["longitudinal_accel"] > r3_accel) & (df["speed"] < r3_speed)] = 1
        return lab

    notes = []
    lab = apply(r1_speed, r1_accel)
    rate = lab.mean()
    notes.append(f"initial anomaly rate (defaults): {rate:.4f}")

    if rate < 0.02:
        r1_speed = 10.0
        lab = apply(r1_speed, r1_accel)
        rate = lab.mean()
        notes.append(f"rate < 2% -> loosened rule1 speed to {r1_speed}; new rate {rate:.4f}")
    elif rate > 0.08:
        r1_accel = -4.0
        lab = apply(r1_speed, r1_accel)
        rate = lab.mean()
        notes.append(f"rate > 8% -> tightened rule1 accel to {r1_accel}; new rate {rate:.4f}")

    final = {
        "rule1_hard_brake_at_speed": {"speed>": r1_speed, "accel<": r1_accel},
        "rule2_steering_oscillation": {"abs(steering)>": r2_steer, "abs(accel)<": r2_accel},
        "rule3_acceleration_spike": {"accel>": r3_accel, "speed<": r3_speed},
    }
    return lab, rate, final, notes


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv(IN)

    # ---------------- §2.2 Anomaly injection ----------------
    label, rate, thresholds, notes = inject_labels(df)
    df["label"] = label
    counts = df["label"].value_counts().sort_index()

    inj = []
    inj.append("=== §2.2 Anomaly Injection ===")
    inj.extend(notes)
    inj.append(f"\nfinal anomaly rate: {rate:.4f}")
    inj.append(f"class counts: normal(0)={int(counts.get(0,0)):,}  anomaly(1)={int(counts.get(1,0)):,}")
    inj.append("\nfinal thresholds used:")
    inj.append(json.dumps(thresholds, indent=2))
    with open("reports/step2_injection.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(inj) + "\n")
    print("\n".join(inj))

    # ---------------- §2.3 Normalisation to [-1, 1] ----------------
    for col, (lo, hi) in BOUNDS.items():
        df[col + "_norm"] = 2 * (df[col] - lo) / (hi - lo) - 1
        df[col + "_norm"] = df[col + "_norm"].clip(-1, 1)

    with open("models/scaler_bounds.json", "w", encoding="utf-8") as f:
        json.dump(BOUNDS, f, indent=2)

    desc = df[FEATURES].describe().to_string()
    with open("reports/step2_normalisation.txt", "w", encoding="utf-8") as f:
        f.write("=== §2.3 Normalisation: describe() of *_norm columns ===\n")
        f.write(desc + "\n")
        in_range = (df[FEATURES].ge(-1) & df[FEATURES].le(1)).all().all()
        f.write(f"\nall *_norm values within [-1, 1]: {bool(in_range)}\n")
    print("\n" + desc)

    # Persist labelled dataset (label + 4 _norm columns + originals)
    df.to_csv(LABELLED, index=False)
    print(f"\nWrote {LABELLED}: {len(df):,} rows")

    # ---------------- §2.4 Train/Test split (sanity print) ----------------
    X = df[FEATURES].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train:", X_train.shape, "| Anomaly rate:", round(float(y_train.mean()), 4))
    print("Test: ", X_test.shape, "| Anomaly rate:", round(float(y_test.mean()), 4))


if __name__ == "__main__":
    main()
