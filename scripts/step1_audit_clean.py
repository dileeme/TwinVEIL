"""
TwinVEIL — Module 2, Step 1.3 (Data Audit) + Step 1.4 (Cleaning).

Reads data/interim/chunk1.csv, writes:
  - reports/step1_audit.txt  (info, describe, nulls, duplicates, per-rule counts)
  - reports/hist_<col>.png   (one histogram per feature, pre-clean)
  - data/processed/cleaned.csv

Cleaning rules applied in the exact order specified by §1.4. Row count is
recorded after each rule. Missing values are dropped, never imputed.
"""
import io
import os
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

IN = "data/interim/chunk1.csv"
OUT = "data/processed/cleaned.csv"
AUDIT = "reports/step1_audit.txt"
FEATURES = ["speed", "longitudinal_accel", "steering_angle", "brake_pressure"]


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(IN)
    log = []  # lines for the audit report

    # ---------------- §1.3 Data Audit ----------------
    buf = io.StringIO()
    df.info(buf=buf)
    log.append("=== df.info() ===")
    log.append(buf.getvalue())
    log.append("=== df.describe() ===")
    log.append(df.describe().to_string())
    log.append("\n=== Nulls per column ===")
    log.append(df.isnull().sum().to_string())
    log.append(f"\nDuplicates (full-row): {df.duplicated().sum()}")
    log.append(f"Duplicate timestamps:  {df['timestamp'].duplicated().sum()}")
    log.append(f"Null timestamps:       {df['timestamp'].isnull().sum()}")

    # Histograms (pre-clean), one per feature
    for col in FEATURES:
        df[col].hist(bins=60)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(f"reports/hist_{col}.png", dpi=120)
        plt.clf()
    log.append("\nSaved histograms: " + ", ".join(f"reports/hist_{c}.png" for c in FEATURES))

    # ---------------- §1.4 Cleaning ----------------
    log.append("\n=== Cleaning (row counts after each rule) ===")
    log.append(f"Rule 0  start                              : {len(df):,}")

    # Rule 1: sort by timestamp; drop duplicate or null timestamps
    df = df.sort_values("timestamp", kind="mergesort")
    df = df[df["timestamp"].notna()]
    df = df.drop_duplicates(subset="timestamp", keep="first")
    log.append(f"Rule 1  sort + drop dup/null timestamps    : {len(df):,}")

    # Rule 2: drop speed < 0 or speed > 60 m/s
    df = df[(df["speed"] >= 0) & (df["speed"] <= 60)]
    log.append(f"Rule 2  drop speed<0 or >60                : {len(df):,}")

    # Rule 3: clip longitudinal_accel to [-10, +8] (sensor faults)
    df["longitudinal_accel"] = df["longitudinal_accel"].clip(-10, 8)
    log.append(f"Rule 3  clip accel to [-10, 8]  (no drop)  : {len(df):,}")

    # Rule 4: drop rows where any of the 4 feature columns is NaN
    df = df.dropna(subset=FEATURES)
    log.append(f"Rule 4  drop NaN in any feature            : {len(df):,}")

    # Rule 5: brake_pressure raw int 0-255 -> /255.0 (check dtype first)
    if pd.api.types.is_integer_dtype(df["brake_pressure"]) and df["brake_pressure"].max() > 1:
        df["brake_pressure"] = df["brake_pressure"] / 255.0
        log.append("Rule 5  brake_pressure was raw int -> /255.0")
    else:
        log.append(f"Rule 5  brake_pressure dtype={df['brake_pressure'].dtype}, "
                   f"max={df['brake_pressure'].max():.3f} -> already 0-1, no change")

    # Drop the timestamp now that rows are sorted (index only, per §1.2)
    df = df.reset_index(drop=True)
    df.to_csv(OUT, index=False)
    log.append(f"\nWrote {OUT}: {len(df):,} rows")

    report = "\n".join(log)
    with open(AUDIT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[audit written to {AUDIT}]")


if __name__ == "__main__":
    main()
