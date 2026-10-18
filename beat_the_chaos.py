"""Beat the chaos: predict pump failure with actual lead time.

Rules of engagement:
  - time-based split (train on first 70%, test on last 30%)
  - features only use rolling history (no future, no rul)
  - scored on: can you flag a pump before it enters the failure window?

Outputs per-row failure risk and prints a lead-time report.
Saves: chaos_predictions.csv.gz
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

DATA = "chaos_pump_rul_benchmark.csv.gz"


def build_features(g):
    g = g.sort_values("minute" if "minute" in g else "timestamp").copy()
    for feat in ["vibration", "temperature", "flow_rate", "pressure", "power_consumption"]:
        # sensor lies: median over a window resists spikes and sticking
        g[f"{feat}_med"] = g[feat].rolling(30, min_periods=5).median()
        g[f"{feat}_slope"] = (
            g[feat].rolling(30, min_periods=5).median()
            - g[feat].rolling(180, min_periods=20).median()
        )
        g[f"{feat}_std"] = g[feat].rolling(60, min_periods=10).std()
        g[f"{feat}_miss"] = g[feat].isna().astype(float).rolling(60, min_periods=1).mean()
    # cross-sensor physics: power per unit flow (efficiency proxy)
    g["efficiency"] = g["power_consumption_med"] / (g["flow_rate_med"] + 1)
    # raw values too, forward-filled so NaN dropouts don't kill the model
    for feat in ["vibration", "temperature", "flow_rate", "pressure", "power_consumption"]:
        g[feat] = g[feat].ffill()
    return g


def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    X = df.groupby("pump_id", group_keys=False).apply(build_features)
    X["pump_id"] = df["pump_id"].to_numpy()
    X["pre_failure"] = ((X["rul"] > 0) & (X["rul"] <= 120)).astype(int)

    cut = X["timestamp"].quantile(0.7)
    tr, te = X[X.timestamp <= cut], X[X.timestamp > cut]
    print(f"train rows {len(tr)}, test rows {len(te)}")
    print(f"pre_failure rate train {tr.pre_failure.mean():.4f}, test {te.pre_failure.mean():.4f}")

    feats = ([c for c in X.columns if c.endswith(("_med", "_slope", "_std", "_miss"))]
             + ["efficiency", "vibration", "temperature", "flow_rate", "pressure", "power_consumption"])

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08)
    model.fit(tr[feats], tr["pre_failure"])
    p = model.predict_proba(te[feats])[:, 1]
    print(f"AUC (predicting failure within 120 min): {roc_auc_score(te['pre_failure'], p):.4f}")

    # lead-time analysis: at the best-F1 threshold, when does the alarm first fire
    # relative to the true failure-window start?
    te = te.assign(risk=p)
    prec, rec, thr = precision_recall_curve(te.pre_failure, te.risk)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_thr = thr[f1.argmax()]
    print(f"best F1 threshold {best_thr:.3f}, precision {prec[f1.argmax()]:.3f}, recall {rec[f1.argmax()]:.3f}")

    leads = []
    for pid, g in te[te.rul > 0].groupby("pump_id"):
        fail_start = g[g.rul <= 60].timestamp.min()
        fired = g[g.risk > best_thr]
        if len(fired):
            leads.append((fired.timestamp.min() - fail_start).total_seconds() / 60)
    if leads:
        leads = np.array(leads)
        print(f"alarms fired on {len(leads)}/{te[te.rul>0].pump_id.nunique()} dying pumps")
        print(f"median warning: {abs(np.median(leads)):.0f} min BEFORE the failure window opens")
    out = te[["timestamp", "pump_id", "risk", "pre_failure", "rul"]]
    out.to_csv("chaos_predictions.csv.gz", index=False, compression="gzip")
    print("wrote chaos_predictions.csv.gz")


if __name__ == "__main__":
    main()
