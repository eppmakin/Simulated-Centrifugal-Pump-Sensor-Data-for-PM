"""Chaos-mode pump sensor generator.

The original dataset was i.i.d. noise with a label you could read straight off
one feature. This one simulates pumps that degrade over time, lie to you,
drop data, and fail in bursts. If your model gets AUC 1.0 on this, you cheated.

Output: chaos_pump_sensor_data.csv.gz (data)
        chaos_pump_rul_benchmark.csv.gz (data + rul, for scoring only)
Columns: timestamp, vibration, temperature, flow_rate, pressure,
         power_consumption, failure
`failure` = 1 only for the final FAILURE_WINDOW minutes of a pump's life,
i.e. the moment maintenance got called, not the whole degradation.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(1337)

N_PUMPS = 12
FAILURE_WINDOW = 60          # minutes flagged as failure at end of life
MINUTES_PER_DAY = 1440

# per-pump baseline personalities
PUMP_BASELINES = {
    "vibration": (5.0, 0.8),
    "temperature": (60.0, 2.0),
    "flow_rate": (300.0, 15.0),
    "pressure": (10.0, 0.7),
    "power_consumption": (5.0, 0.3),
}

# feature-wise degradation gain at the moment of death (multiplied by wear 0->1)
DEGRADATION = {
    "vibration": (10.0, 3.0),        # bearings: huge vibration growth
    "temperature": (15.0, 4.0),      # heating rises with wear
    "flow_rate": (-40.0, 10.0),      # flow drops as impeller wears
    "pressure": (3.0, 1.0),          # pressure fluctuates more, slight rise
    "power_consumption": (3.0, 1.2), # motor works harder to keep flow
}

DEGRADATION_NOISE = {
    "vibration": 0.9,
    "temperature": 0.6,
    "flow_rate": 0.5,
    "pressure": 0.4,
    "power_consumption": 0.25,
}


def seasonality(minutes):
    """Daily load cycle: plant runs harder during day shifts."""
    hour = (minutes // 60) % 24
    return 0.85 + 0.15 * np.sin((hour - 6) / 24 * 2 * np.pi)


def simulate_pump(pump_id, n_days):
    """A pump lives, degrades, dies, gets rebuilt (imperfectly), repeats."""
    n = n_days * MINUTES_PER_DAY
    t = np.arange(n)

    wear = np.zeros(n)
    failure = np.zeros(n, dtype=int)
    fail_starts = []

    pos = 0
    repair_quality = 1.0
    while pos < n:
        life = int(RNG.integers(2 * MINUTES_PER_DAY, 6 * MINUTES_PER_DAY) * repair_quality)
        life = min(life, n - pos - 1)
        if life <= FAILURE_WINDOW + 120:
            break
        ramp_start = int(life * RNG.uniform(0.5, 0.8))

        seg = np.zeros(life)
        seg[:ramp_start] = RNG.beta(1.5, 8, ramp_start) * 0.15
        ramp = np.arange(ramp_start, life) - ramp_start
        seg[ramp_start:] = 0.15 + 0.85 * (ramp / max(1, life - ramp_start)) ** 2.2
        wear[pos:pos + life] = seg

        fail_starts.append(pos + life - FAILURE_WINDOW)
        failure[pos + life - FAILURE_WINDOW:pos + life] = 1

        pos += life
        # imperfect repair: next life is shorter, baseline never fully returns
        repair_quality = max(0.4, repair_quality * RNG.uniform(0.75, 0.95))

    # remaining-useful-life: minutes until the next failure window starts
    fail_starts = np.array(fail_starts)
    idx = np.searchsorted(fail_starts, t, side="left")
    next_fail = np.full(n, np.inf)
    ok = idx < len(fail_starts)
    next_fail[ok] = fail_starts[idx[ok]]
    rul = np.where(np.isfinite(next_fail), np.maximum(next_fail - t, 0), np.inf)

    season = seasonality(t)
    rows = {}
    for feat, (base_mu, base_sd) in PUMP_BASELINES.items():
        mu = base_mu * RNG.uniform(0.9, 1.1)  # unit personality
        deg_mu, deg_sd = DEGRADATION[feat]
        signal = (
            mu * (season if feat in ("flow_rate", "power_consumption") else 1.0)
            + wear * RNG.normal(deg_mu, deg_sd * 0.3)
            + RNG.normal(0, base_sd, n)
            + wear * RNG.normal(0, deg_sd, n) * DEGRADATION_NOISE[feat]
        )
        # sensor lies: sticking, spikes (dropout handled later)
        stick_mask = RNG.random(n) < 0.01
        if stick_mask.any():
            idx2 = np.where(stick_mask)[0]
            prev = np.maximum(idx2 - 1, 0)
            signal[idx2] = signal[prev]
        spike_mask = RNG.random(n) < 0.004
        signal[spike_mask] *= RNG.uniform(1.8, 3.0, spike_mask.sum())
        rows[feat] = signal

    df = pd.DataFrame(rows)
    df["pressure"] = df["pressure"] + 0.05 * df["flow_rate"] - 0.15 * df["vibration"]

    # cascading events: a big vibration slam often knocks temperature/flow up too
    slam = RNG.random(n) < 0.002
    df.loc[slam, "temperature"] += RNG.normal(8, 2, slam.sum())
    df.loc[slam, "vibration"] += RNG.normal(4, 1, slam.sum())

    df["failure"] = failure
    df["pump_id"] = pump_id
    df["minute"] = t
    df["rul"] = rul
    return df


def main():
    frames = []
    for i in range(N_PUMPS):
        days = int(RNG.integers(45, 75))
        frames.append(simulate_pump(f"P{i:02d}", days))
    df = pd.concat(frames, ignore_index=True)

    # global timestamp axis
    start = pd.Timestamp("2024-01-01 00:00:00")
    df["timestamp"] = start + pd.to_timedelta(df["minute"], unit="m")

    # missing data: entire sensors drop out in blocks
    for feat in PUMP_BASELINES:
        drop = RNG.random(len(df)) < 0.03
        df.loc[drop, feat] = np.nan

    # slow sensor drift on temperature (calibration rot) for some pumps
    for pid in df.pump_id.unique()[:4]:
        m = df.pump_id == pid
        df.loc[m, "temperature"] = df.loc[m, "temperature"] + df.loc[m, "minute"] * 0.00012

    df = df.sort_values(["timestamp", "pump_id"]).reset_index(drop=True)
    keep = ["timestamp", "pump_id", "vibration", "temperature", "flow_rate",
            "pressure", "power_consumption", "failure"]
    bench = ["timestamp", "pump_id", "rul"] + keep[2:]
    df[keep].to_csv("chaos_pump_sensor_data.csv.gz", index=False, compression="gzip")
    df[bench].to_csv("chaos_pump_rul_benchmark.csv.gz", index=False, compression="gzip")
    print(f"wrote {len(df)} rows, {df.failure.sum()} failure minutes")
    print(f"failure rate: {df.failure.mean():.4f}, NaN cells: {df[keep[2:-1]].isna().sum().sum()}")


if __name__ == "__main__":
    main()
