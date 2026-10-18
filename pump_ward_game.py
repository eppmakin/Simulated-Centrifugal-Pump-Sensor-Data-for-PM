"""PUMP WARD: you are the night-shift maintenance tech.

12 pumps. One shift. Sensors stream in. The plant loses $50/min for every
minute a pump is shut down early for "no reason", but if a pump enters its
failure window while running, that's a $10,000 blown seal.

Commands:
  watch <pump-id>   stream 30 min of recent history + model risk score
  shut <pump-id>    take a pump offline
  status            dashboard of all pumps
  done              end the shift, get scored

Beat the model. Or trust it. Your call.
"""

import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

EARLY_SHUTDOWN_COST_PER_MIN = 50
FAILURE_COST = 10_000
WINDOW = 60


def spark(values, width=30, lo=None, hi=None):
    lo = lo if lo is not None else np.nanmin(values)
    hi = hi if hi is not None else np.nanmax(values)
    lo, hi = lo - 1e-9, hi + 1e-9
    chars = "▁▂▃▄▅▆▇█"
    idx = np.floor((values - lo) / (hi - lo) * (len(chars) - 1)).astype(int)
    return "".join(chars[i] for i in np.clip(idx, 0, len(chars) - 1))


def main():
    preds = pd.read_csv("chaos_predictions.csv.gz", parse_dates=["timestamp"])
    pumps = sorted(preds.pump_id.unique())

    now = preds.timestamp.max() - pd.Timedelta(minutes=1)
    shift_start = preds.timestamp.quantile(0.9)
    shut_down = {}
    blown = set()
    score = 0

    print("=" * 62)
    print("PUMP WARD — night shift. 12 pumps. Good luck.")
    print("=" * 62)

    while True:
        try:
            cmd = input("\n> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        now = min(now + pd.Timedelta(minutes=30), preds.timestamp.max())

        if cmd[0] == "status":
            rows = []
            for p in pumps:
                g = preds[(preds.pump_id == p) & (preds.timestamp <= now)].tail(1)
                if g.empty:
                    continue
                r = g.iloc[0]
                state = "OFF" if p in shut_down else ("BLOWN" if p in blown else "RUN")
                rows.append({"pump": p, "state": state,
                             "risk": round(r.risk, 3), "rul_hint": "?" if r.rul > 1000 else "<1d"})
            print(pd.DataFrame(rows).to_string(index=False))

        elif cmd[0] == "watch" and len(cmd) > 1:
            p = cmd[1].upper()
            g = preds[(preds.pump_id == p) & (preds.timestamp <= now)].tail(30)
            if g.empty:
                print("no data yet for", p)
                continue
            data = pd.read_csv(
                "chaos_pump_sensor_data.csv.gz", parse_dates=["timestamp"],
                usecols=["timestamp", "pump_id", "vibration", "temperature", "flow_rate", "power_consumption"]
            )
            gg = data[(data.pump_id == p) & (data.timestamp <= now)].tail(60)
            for feat in ["vibration", "temperature", "flow_rate", "power_consumption"]:
                v = gg[feat].tail(30).to_numpy()
                v = v[~np.isnan(v)]
                if len(v):
                    print(f"  {feat:<18} {spark(v)}  last={v[-1]:.2f}")
            print(f"  model risk: {g.risk.iloc[-1]:.3f}   (shut threshold is your call)")

        elif cmd[0] == "shut" and len(cmd) > 1:
            p = cmd[1].upper()
            if p not in pumps or p in shut_down:
                print("can't shut", p)
                continue
            g = preds[(preds.pump_id == p) & (preds.timestamp <= now)].tail(1)
            rul = g.rul.iloc[-1] if not g.empty else np.inf
            shut_down[p] = now
            if 0 < rul <= WINDOW * 3:
                print(f"  {p} down — turns out it was dying. Nice save.")
            else:
                early = rul if rul != np.inf else 10_000
                cost = min(early, 720) * EARLY_SHUTDOWN_COST_PER_MIN
                score -= cost
                print(f"  {p} down early. Plant loses ${cost:,} (charged to your shift).")

        elif cmd[0] == "done":
            for p in pumps:
                if p in shut_down or p in blown:
                    continue
                g = preds[(preds.pump_id == p) & (preds.timestamp <= now)].tail(1)
                if g.empty:
                    continue
                # simulate forward: did it fail while you watched?
                fut = preds[(preds.pump_id == p) & (preds.timestamp > now) & (preds.rul <= WINDOW)]
                if not fut.empty:
                    blown.add(p)
                    score -= FAILURE_COST
                    print(f"  {p} BLEW A SEAL. -$10,000")
            break
        else:
            print(__doc__.split("Commands:")[0])

    print("\n" + "=" * 62)
    if blown:
        print(f"pumps destroyed: {len(blown)} ({', '.join(sorted(blown))})")
    print(f"final shift score: ${score:,}  (0 is perfect)")
    if score >= 0 and not blown:
        print("...did you just read the rul column? cheater. (fine. win's a win.)")
    elif not blown:
        print("no blown seals. the night crew talks about you for a week.")
    else:
        print("the day shift is going to have questions.")
    sys.exit(0)


if __name__ == "__main__":
    main()
