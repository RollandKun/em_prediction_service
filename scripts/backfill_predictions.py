# -*- coding: utf-8 -*-
"""One-shot: backfill predictions table from latest features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from config import settings
from pipeline.inference import run_inference
from pipeline.data_loader import load_for_inference
from pipeline.feature_engine import build_features
from pipeline.output import save_outputs


def rebuild_features(grid_lag=0, forward_extend=0):
    loaded = load_for_inference(forward_extend=forward_extend)
    (dt_arr, df, solar, wind, hydro, load, price,
     bidspace, reserve, nonmarket, tieline, load_tie) = loaded
    result = build_features(
        dt_arr, df, solar, wind, hydro, load, price,
        bidspace, reserve, nonmarket, tieline, load_tie,
        grid_lag=grid_lag,
    )
    save_outputs(result, grid_lag=grid_lag)
    return result

def build(pp, dt, period):
    res = {}
    for i in range(len(pp)):
        d = (pd.Timestamp(dt[i]) + pd.Timedelta(hours=24)).strftime('%Y-%m-%d')
        res.setdefault(d, np.full(96, np.nan))
        p = int(period[i])
        if 0 <= p < 96:
            res[d][p] = pp[i]
    return res

print("Rebuilding latest features...")
rebuild_features(grid_lag=0, forward_extend=0)
rebuild_features(grid_lag=192, forward_extend=192)

print("Running inference...")
result_normal = run_inference(grid_lag=0)
normal = build(result_normal['price_pred'], result_normal['dt'], result_normal['period'])
result_lag = run_inference(grid_lag=192)
lag = build(result_lag['price_pred'], result_lag['dt'], result_lag['period'])

engine = create_engine(settings.database_url_sync, echo=False)
total = 0
normal_label = settings.feature_version
lag_label = settings.feature_version + '_lag192'
last_normal_date = max(normal) if normal else None

if last_normal_date:
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(
                text("DELETE FROM predictions "
                     "WHERE model_version = :v AND target_time::date <= :d"),
                {'v': lag_label, 'd': last_normal_date},
            )
            conn.commit()
    lag = {d: arr for d, arr in lag.items() if d > last_normal_date}
    print(f"Lag192 gap-fill starts after {last_normal_date}: {len(lag)} dates")

for label, preds in [(normal_label, normal), (lag_label, lag)]:
    for d_str in sorted(preds):
        arr = preds[d_str]
        records = []
        for p in range(96):
            v = arr[p]
            h, m = divmod(p, 4); m *= 15
            records.append(dict(
                target_time=pd.Timestamp(f'{d_str} {h:02d}:{m:02d}:00').to_pydatetime(),
                predicted_price=0.0 if np.isnan(v) else float(v),
                model_version=label,
                season='dry' if pd.Timestamp(d_str).month <= 4 else 'wet',
                period=p,
            ))
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("DELETE FROM predictions WHERE target_time::date = :d AND model_version = :v"),
                             {'d': d_str, 'v': label})
                conn.execute(text("INSERT INTO predictions (target_time,predicted_price,model_version,season,period) "
                                  "VALUES (:target_time,:predicted_price,:model_version,:season,:period)"), records)
                conn.commit()
        print(f'  {d_str}: {len(records)} ({label})')
        total += len(records)

engine.dispose()
print(f'Done: {total} predictions')
