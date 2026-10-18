# Simulated Centrifugal Pump Sensor Data — Chaos Edition

Predictive maintenance dataset where the data fights back.

## Why this repo exists: a warning about label leakage

The original `pump_sensor_data_large.csv` is a trap. It's i.i.d. noise with a
label you can read straight off one feature: `vibration > 10 ⇒ failure`.
Any model — logistic regression, a decision stump, one `if` statement —
scores AUC 1.0. There is zero time structure: no degradation, no trends,
nothing to predict, just a threshold hidden in a column.

It's kept here as a warning. If your predictive-maintenance pipeline scores
perfectly on data like that, your data is broken, not your model.

## The chaos dataset

`generate_chaos.py` (seeded, deterministic — seed 1337) simulates 12 pumps
over ~45–75 days at 1-minute resolution: **1,002,240 rows**, 23,100 failure
minutes, ~150k NaN sensor cells. Failure is flagged only for the final 60
minutes of a pump's life — the moment maintenance got called, not the whole
degradation.

The mechanics:

- **Wear ramps.** Each pump lives 2–6 days, then dies. Wear is quiet for most
  of a life, then ramps hard (superlinear) toward the end. Bearings shake,
  impellers lose flow, motors burn power to compensate.
- **Imperfect repairs.** After each death the pump is rebuilt — but each
  rebuild is worse than the last. Lives get shorter; baselines drift up.
- **6 a.m. seasonality traps.** Flow and power follow a daily plant-load
  cycle. A naive model reads the morning ramp-up as impending failure.
- **Lying sensors.** Stuck readings (1% of samples), 2–3× spikes (0.4%),
  cascading slam events that hit vibration and temperature together, whole-sensor
  dropouts (3% NaN per feature), and slow temperature calibration drift on
  four of the pumps.

Vibration over one pump life (median-filtered, left to right — you can watch
it die):

```
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▂▂▂▂▂▂▂▂▂▂▂▂▂▃▃▃▃▃▃▃▃▄▃▄▄▄▄▅▅▅▆▆▆▆▆▇▇▇█████
```

Outputs:

| File | Contents |
|---|---|
| `chaos_pump_sensor_data.csv.gz` | timestamp, pump_id, 5 sensors, failure label |
| `chaos_pump_rul_benchmark.csv.gz` | same + `rul` (minutes to next failure window) — for scoring only |
| `chaos_predictions.csv.gz` | per-row failure risk from the benchmark model |

## The benchmark

`beat_the_chaos.py` — rules of engagement: time-based split (train first 70%,
test last 30%), rolling-history features only (no future, no `rul`), scored on
whether you flag a pump before it enters the failure window.

Results (histogram gradient boosting on rolling medians/slopes/stds/missingness
plus a power-per-flow efficiency proxy):

- **AUC 0.975** predicting failure within 120 minutes
- Alarms fired on **12/12** dying pumps in the test window
- **~2 h (116 min) median warning** before the failure window opens
- Best-F1 threshold: precision 0.625, recall 0.832

## Running it

```bash
pip install pandas scikit-learn numpy

python3 generate_chaos.py     # regenerate the data (deterministic, seed 1337)
python3 beat_the_chaos.py    # train + score the benchmark model
python3 pump_ward_game.py     # play the night shift (needs the two steps above)
```

If your own model gets AUC 1.0 on this dataset, you leaked the `rul` column
somewhere. Check yourself.

## PUMP WARD

You are the night-shift maintenance tech. 12 pumps, sensors streaming in.
Shutting a healthy pump down early costs the plant **$50/min** of lost
production. A pump that enters its failure window while running blows a seal:
**–$10,000**.

```
status            dashboard of all pumps
watch <pump-id>   sparkline history + model risk score
shut <pump-id>    take a pump offline
done              end the shift, get scored
```

Beat the model. Or trust it. Your call. (If you finish with a perfect score
and no blown seals... did you just read the `rul` column? Cheater. Win's a win.)
