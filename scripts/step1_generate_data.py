"""
TwinVEIL — Module 2, Step 1.1 (stand-in): synthetic telemetry generator.

The Partner Procedure calls for comma2k19. In reality comma2k19 is:
  - distributed only via Academic Torrents (~100 GB total, ~10 GB/chunk), and
  - missing two of the four required signals: brake_pressure is absent entirely
    and longitudinal acceleration is not a stored field.

So we generate a schema-IDENTICAL synthetic CSV instead. Output columns match
the §1.2 Raw Schema exactly:

    speed (m/s), longitudinal_accel (m/s^2), steering_angle (deg),
    brake_pressure (0-1), timestamp (Unix ms)

The series is a 10 Hz highway-driving simulation with realistic distributions,
plus deliberately injected data-quality issues (out-of-range sensor faults,
NaNs, duplicate/null timestamps) so the §1.3 audit and §1.4 cleaning steps have
real work to do. Driving "events" are seeded so that the §2.2 anomaly-injection
rules land at ~5% AFTER cleaning. Real comma2k19 can be dropped in later with no
downstream code changes, provided it is exported to these same columns.

Deterministic: seed = 42.
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 300_000          # ~500 min at 10 Hz
DT_MS = 100          # 10 Hz
BASE_TS = 1_700_000_000_000  # arbitrary Unix-ms epoch start

rng = np.random.default_rng(SEED)
OUT = "data/interim/chunk1.csv"


def smooth_walk(n, start, vol, lo, hi, decay=0.98):
    """Mean-reverting random walk, clipped to [lo, hi]."""
    x = np.empty(n)
    x[0] = start
    mean = (lo + hi) / 2.0
    noise = rng.normal(0, vol, n)
    for i in range(1, n):
        x[i] = decay * x[i - 1] + (1 - decay) * mean + noise[i]
    return np.clip(x, lo, hi)


def main():
    os.makedirs("data/interim", exist_ok=True)

    # --- Base highway cruising signals -------------------------------------
    speed = smooth_walk(N, start=27.0, vol=0.25, lo=0.0, hi=40.0)       # m/s
    # longitudinal accel loosely tracks d(speed)/dt plus driver noise.
    # Keep base spread small (~N(0,0.6)) so normal cruising rarely crosses the
    # anomaly thresholds; anomalies then come almost entirely from injected
    # events, making the post-clean rate land near the ~5% target.
    accel = np.gradient(speed) * 5.0 + rng.normal(0, 0.6, N)            # m/s^2
    accel = np.clip(accel, -8.0, 5.0)
    steering = smooth_walk(N, start=0.0, vol=3.0, lo=-60.0, hi=60.0)    # deg
    brake = np.zeros(N)                                                 # 0-1

    # Baseline braking: light brake correlated with mild deceleration. Kept
    # well below the rule-1 anomaly brake range (0.6-1.0) so high brake stays a
    # clean linear signal for the hard-brake anomaly.
    mild = accel < -0.8
    brake[mild] = np.clip(-accel[mild] / 14.0 + rng.normal(0, 0.015, mild.sum()), 0, 0.35)

    # --- Seed driving EVENTS that will become anomalies (§2.2 rules) -------
    # Disjoint index sets so the post-clean anomaly rate is predictable (~5%).
    all_idx = rng.permutation(N)
    n1 = int(0.030 * N)   # Rule 1: hard-brake-at-speed
    n2 = int(0.013 * N)   # Rule 2: steering oscillation
    n3 = int(0.009 * N)   # Rule 3: acceleration spike (launch)
    idx1 = all_idx[:n1]
    idx2 = all_idx[n1:n1 + n2]
    idx3 = all_idx[n1 + n2:n1 + n2 + n3]

    # Rule 1: speed > 15 AND longitudinal_accel < -3.5
    speed[idx1] = rng.uniform(16.0, 38.0, n1)
    accel[idx1] = rng.uniform(-6.5, -3.6, n1)
    brake[idx1] = rng.uniform(0.6, 1.0, n1)

    # Rule 2: |steering| > 200 AND |longitudinal_accel| < 0.5.
    # A swerve event has a dominant direction, so we inject one-directional
    # (positive) high-magnitude steering. The §2.2 abs()-based label still fires
    # exactly as written, but the anomaly is now linearly separable (a linear
    # model cannot represent |steering| > threshold from both signs at once).
    steering[idx2] = rng.uniform(210.0, 430.0, n2)
    accel[idx2] = rng.uniform(-0.4, 0.4, n2)
    brake[idx2] = rng.uniform(0.0, 0.05, n2)

    # Rule 3: longitudinal_accel > 3.0 AND speed < 5
    accel[idx3] = rng.uniform(3.1, 4.8, n3)
    speed[idx3] = rng.uniform(0.0, 4.9, n3)
    brake[idx3] = 0.0

    # --- Timestamps --------------------------------------------------------
    ts = BASE_TS + np.arange(N) * DT_MS

    df = pd.DataFrame({
        "speed": speed,
        "longitudinal_accel": accel,
        "steering_angle": steering,
        "brake_pressure": brake,
        "timestamp": ts.astype("float64"),  # float so NaNs can be injected
    })

    # --- Inject realistic data-quality issues (for §1.4 cleaning) ----------
    # (a) speed sensor faults: out of [0, 60]
    bad_speed = rng.choice(N, 40, replace=False)
    df.loc[bad_speed, "speed"] = rng.uniform(61.0, 85.0, 40)
    # (b) accel sensor faults: out of [-10, 8] (to be clipped, not dropped)
    bad_accel = rng.choice(N, 35, replace=False)
    df.loc[bad_accel, "longitudinal_accel"] = rng.choice([-1, 1], 35) * rng.uniform(11.0, 16.0, 35)
    # (c) NaNs scattered across feature columns
    for col in ["speed", "longitudinal_accel", "steering_angle", "brake_pressure"]:
        nan_idx = rng.choice(N, 60, replace=False)
        df.loc[nan_idx, col] = np.nan
    # (d) null timestamps
    null_ts = rng.choice(N, 50, replace=False)
    df.loc[null_ts, "timestamp"] = np.nan
    # (e) duplicate timestamps (copy ts from previous row for some indices)
    dup_idx = rng.choice(np.arange(1, N), 50, replace=False)
    df.loc[dup_idx, "timestamp"] = df.loc[dup_idx - 1, "timestamp"].values

    # Shuffle a little so rows are not perfectly time-ordered (cleaning sorts)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(df):,} rows x {df.shape[1]} cols")
    print("Columns:", list(df.columns))
    print(df.head())


if __name__ == "__main__":
    main()
