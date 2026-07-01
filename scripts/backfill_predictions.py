# -*- coding: utf-8 -*-
"""One-shot: backfill predictions table from latest features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from config import settings
from pipeline.inference import run_inference

def build(pp, dt, period):
    res = {}
    for i in range(len(pp)):
        d = (pd.Timestamp(dt[i]) + pd.Timedelta(hours=24)).strftime('%Y-%m-%d')
        res.setdefault(d, np.full(96, np.nan))
        p = int(period[i])
        if 0 <= p < 96:
            res[d][p] = pp[i]
    return res

print("Running inference...")
normal = build(*[run_inference(grid_lag=0)[k] for k in ['price_pred','dt','period']])
lag    = build(*[run_inference(grid_lag=192)[k] for k in ['price_pred','dt','period']])

engine = create_engine(settings.database_url_sync, echo=False)
total = 0

for label, preds in [('v14', normal), ('v14_lag192', lag)]:
    for d_str in sorted(preds):
        arr = preds[d_str]
        records = []
        for p in range(96):
            v = arr[p]
            if not np.isnan(v) and v > 0:
                h, m = divmod(p, 4); m *= 15
                records.append(dict(
                    target_time=pd.Timestamp(f'{d_str} {h:02d}:{m:02d}:00').to_pydatetime(),
                    predicted_price=float(v), model_version=label,
                    season='dry' if pd.Timestamp(d_str).month <= 4 else 'wet', period=p))
        if records:
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
