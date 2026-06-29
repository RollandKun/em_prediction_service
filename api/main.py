# -*- coding: utf-8 -*-
"""
em_prediction_service — FastAPI application
============================================
Prediction API for Sichuan electricity spot market price forecasting.

Endpoints:
    GET  /health                          Health check
    GET  /api/v1/predictions?date=...     Get predictions for a date
    GET  /api/v1/predictions/latest       Get latest available predictions
    GET  /api/v1/models                   List active models 

Startup: loads all models + pre-computes predictions for all dates in DB.
"""
import sys
import io
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime, date
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Logging ──
logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from pipeline.inference import (
    load_stage1_models, load_stage2_models,
    predict_stage1, build_stage2_features, blend_weights,
)

from api.schemas import (
    PredictionPoint, PredictionSummary, PredictionsResponse,
    ModelInfo, ModelsResponse, HealthResponse,
)

# ── Global state (populated at startup) ──
state = {
    "models_s1": {},
    "models_s2": {},
    "predictions_cache": {},    # date_str → list[96 prices]
    "feat_names": None,
    "X_full": None,
    "period": None,
    "price": None,
    "dry_mask": None,
    "wet_mask": None,
    "dt_arr": None,
    "db_ok": False,
}

# ── Period → time mapping ──
def _period_to_time(p: int) -> str:
    h = p // 4
    m = (p % 4) * 15
    return f"{h:02d}:{m:02d}"

def _period_to_segment(p: int) -> str:
    if p <= 35 or p >= 88:
        return "base"
    elif p <= 67:
        return "valley"
    else:
        return "peak"

# ── Startup ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and pre-compute predictions at startup."""
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("FastAPI startup — loading models...")
    logger.info(f"  feature_version={settings.feature_version}  model_dir={settings.model_dir}")

    # ── Check DB ──
    t1 = time.time()
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.database_url_sync, echo=False)
        with engine.connect() as conn:
            r = conn.execute(text("SELECT COUNT(*) FROM grid_data")).scalar()
            dmin = conn.execute(text("SELECT MIN(datetime) FROM grid_data")).scalar()
            dmax = conn.execute(text("SELECT MAX(datetime) FROM grid_data")).scalar()
            state["db_ok"] = True
            logger.info(f"  DB: {r} rows, {dmin} → {dmax}  ({time.time()-t1:.1f}s)")
        engine.dispose()
    except Exception as e:
        logger.error(f"  DB connection failed: {e}")
        state["db_ok"] = False

    # ── Load features ──
    t2 = time.time()
    feat_path = PROJECT_ROOT / "pipeline" / "output" / "features_15min_dry.npz"
    if feat_path.exists():
        data = np.load(feat_path, allow_pickle=True)
        state["X_full"] = data["X"]
        state["feat_names"] = data["feat_names"]
        state["period"] = data["period"]
        state["price"] = data["price"]
        state["dry_mask"] = data["dry_mask"]
        state["wet_mask"] = data["wet_mask"]
        state["dt_arr"] = data["dt"]
        dry_n = state["dry_mask"].sum(); wet_n = state["wet_mask"].sum()
        logger.info(f"  Features: {data['X'].shape}  dry={dry_n} wet={wet_n}  "
                    f"path={feat_path.name}  ({time.time()-t2:.1f}s)")
    else:
        logger.error(f"  Features NOT FOUND: {feat_path}")
        state["db_ok"] = False

    # ── Load models ──
    t3 = time.time()
    state["models_s1"] = load_stage1_models()
    state["models_s2"] = load_stage2_models()
    n_s1, n_s2 = len(state["models_s1"]), len(state["models_s2"])
    s1_keys = sorted(state["models_s1"].keys())
    s2_keys = sorted(state["models_s2"].keys())
    logger.info(f"  Models: S1={n_s1} {s1_keys}  S2={n_s2} {s2_keys}  "
                f"({time.time()-t3:.1f}s)")
    if n_s1 < 8:
        missing_s1 = set(f"{v}_{s}" for v in ['solar','hydro','wind','load'] for s in ['dry','wet']) - set(s1_keys)
        logger.error(f"  Missing Stage1 models: {sorted(missing_s1)}")
    if n_s2 < 6:
        missing_s2 = set(f"{seg}_{s}" for seg in ['valley','peak','base'] for s in ['dry','wet']) - set(s2_keys)
        logger.error(f"  Missing Stage2 models: {sorted(missing_s2)}")

    # ── Pre-compute ──
    if state["X_full"] is not None and n_s1 >= 8 and n_s2 >= 6:
        t4 = time.time()
        logger.info("  Pre-computing all predictions...")
        _precompute_predictions()
        n_dates = len(state["predictions_cache"])
        # Quick sanity: check first/last date and price range
        dates_sorted = sorted(state["predictions_cache"].keys())
        first_date, last_date = dates_sorted[0], dates_sorted[-1]
        all_prices = np.concatenate([state["predictions_cache"][d] for d in dates_sorted])
        logger.info(f"  Cache: {n_dates} dates ({first_date} → {last_date})  "
                    f"price range=[{all_prices.min():.1f}, {all_prices.max():.1f}]  "
                    f"({time.time()-t4:.1f}s)")
    else:
        logger.warning(f"  Skipping pre-computation: "
                       f"features={'OK' if state['X_full'] is not None else 'MISSING'}, "
                       f"S1={n_s1}/8, S2={n_s2}/6")

    logger.info(f"Startup complete — {time.time()-t0:.1f}s total")
    logger.info("=" * 50)

    yield  # app runs here

    # Shutdown
    logger.info("Shutting down — clearing state")
    state.clear()


def _precompute_predictions():
    """Run full inference and cache by date."""
    X_full = state["X_full"]
    feat_names = state["feat_names"]
    period = state["period"]
    price = state["price"]
    dry_mask = state["dry_mask"]
    wet_mask = state["wet_mask"]
    dt_arr = state["dt_arr"]
    m1 = state["models_s1"]
    m2 = state["models_s2"]
    n = X_full.shape[0]

    # Stage1
    oof_s = np.full(n, np.nan); oof_h = np.full(n, np.nan)
    oof_w = np.full(n, np.nan); oof_l = np.full(n, np.nan)

    for season, mask in [('dry', dry_mask), ('wet', wet_mask)]:
        s, h, w, l = predict_stage1(m1, X_full, feat_names, season)
        oof_s[mask] = s[mask]; oof_h[mask] = h[mask]
        oof_w[mask] = w[mask]; oof_l[mask] = l[mask]
    for name, a in [('solar', oof_s), ('hydro', oof_h), ('wind', oof_w), ('load', oof_l)]:
        nan_count = np.isnan(a).sum()
        if nan_count > 0:
            logger.warning(f"  Stage1 {name}: {nan_count} NaN OOF → filled with 0")
        a[np.isnan(a)] = 0.0

    # Stage2 — use feature indices from training metadata
    if not m2:
        logger.error("No Stage2 models loaded — cannot run inference")
        return
    first_m2 = next(iter(m2.values()))
    safe_idx = first_m2.get('safe_indices') if isinstance(first_m2, dict) else None
    X_s2 = build_stage2_features(X_full, feat_names, oof_s, oof_h, oof_w, oof_l,
                                  period, safe_indices=safe_idx)

    lag96 = np.roll(price, 96); lag96[:96] = np.nan
    lag672 = np.roll(price, 672); lag672[:672] = np.nan
    anchor = (lag96 + lag672) / 2.0

    price_pred = np.full(n, np.nan)

    for season, mask in [('dry', dry_mask), ('wet', wet_mask)]:
        idx = np.where(mask)[0]
        if len(idx) == 0: continue

        seg_preds = {}
        for seg in ['valley', 'peak', 'base']:
            key = f"{seg}_{season}"
            if key in m2:
                m = m2[key]
                model = m['model'] if isinstance(m, dict) else m
                seg_preds[seg] = model.predict(X_s2[idx])
            else:
                seg_preds[seg] = np.zeros(len(idx))

        w = blend_weights(period[idx])
        blended = (w[:, 0] * seg_preds['valley'] +
                   w[:, 1] * seg_preds['peak'] +
                   w[:, 2] * seg_preds['base'])

        if season == 'dry':
            price_pred[idx] = anchor[idx] + blended
        else:
            price_pred[idx] = blended

    # Cache by date
    nan_price = np.isnan(price_pred).sum()
    if nan_price > 0:
        logger.warning(f"  price_pred: {nan_price} NaN values → filled with 0")
    price_pred[np.isnan(price_pred)] = 0.0

    for i in range(n):
        # Cache by TARGET date (dt + 24h), not input date.
        # price[t+96] predicts the day after the feature timestamp.
        ts = pd.Timestamp(dt_arr[i])
        d = (ts + pd.Timedelta(hours=24)).strftime('%Y-%m-%d')

        if d not in state["predictions_cache"]:
            state["predictions_cache"][d] = np.full(96, np.nan)
        p = period[i]
        if 0 <= p < 96:
            state["predictions_cache"][d][p] = price_pred[i]

    # Fill NaN with 0 for completeness (shouldn't happen — log if it does)
    for d in state["predictions_cache"]:
        arr = state["predictions_cache"][d]
        nan_n = np.isnan(arr).sum()
        if nan_n > 0:
            logger.warning(f"  predictions_cache[{d}]: {nan_n} NaN periods → filled with 0")
        arr[np.isnan(arr)] = 0.0


# ── App ──
app = FastAPI(
    title="EM Prediction Service",
    description="Sichuan electricity spot market price forecasting API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Path normalization (collapse // → / before routing) ──
@app.middleware("http")
async def normalize_double_slash(request: Request, call_next):
    """Collapse repeated slashes in path so //api/foo matches /api/foo."""
    path = request.url.path
    if "//" in path:
        # Replace the raw_path in scope so downstream routing sees normalized path
        normalized = path
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        request.scope["path"] = normalized
        request.scope["raw_path"] = normalized.encode()
    return await call_next(request)


# ── Request logging middleware ──
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    dt_ms = (time.time() - t0) * 1000
    if "/api/" in request.url.path or request.url.path == "/health":
        logger.info(f"{request.method} {request.url.path} "
                    f"→ {response.status_code} ({dt_ms:.0f}ms)")
    return response


# ── Routes ──

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — DB connection, model count, data range."""
    date_range = None
    if state["dt_arr"] is not None:
        d0 = state["dt_arr"][0]
        d1 = state["dt_arr"][-1]
        d0_str = str(d0)[:10] if hasattr(d0, 'strftime') else str(d0)[:10]
        d1_str = str(d1)[:10] if hasattr(d1, 'strftime') else str(d1)[:10]
        date_range = f"{d0_str} → {d1_str}"

    return HealthResponse(
        status="healthy" if state["db_ok"] else "degraded",
        db_connected=state["db_ok"],
        models_loaded=len(state["models_s1"]) + len(state["models_s2"]),
        feature_version=settings.feature_version,
        data_date_range=date_range,
    )


@app.get("/api/v1/predictions", response_model=PredictionsResponse)
async def get_predictions(date_str: str = Query(None, alias="date")):
    """Get 96 price predictions for a specific date."""
    cache = state["predictions_cache"]
    if not cache:
        logger.error("GET /predictions: cache is empty")
        raise HTTPException(503, "Predictions not yet computed. Check model loading.")

    if date_str is None:
        date_str = max(cache.keys())

    if date_str not in cache:
        available = sorted(cache.keys())
        logger.warning(f"GET /predictions: date={date_str} not found, "
                       f"available={available[0]}→{available[-1]}")
        raise HTTPException(
            404,
            f"Date '{date_str}' not found. Available: {available[0]} → {available[-1]}"
        )

    prices = cache[date_str]
    # Sanity: warn if all zeros (indicates upstream failure)
    if np.all(prices == 0):
        logger.warning(f"GET /predictions: date={date_str} all 96 periods are 0 — possible inference failure")

    prices = cache[date_str]
    preds = []
    for p in range(96):
        preds.append(PredictionPoint(
            period=p,
            time=_period_to_time(p),
            price=round(float(prices[p]), 1),
            segment=_period_to_segment(p),
        ))

    # Summary
    peak_idx = int(np.argmax(prices))
    valley_idx = int(np.argmin(prices))
    summary = PredictionSummary(
        avg_price=round(float(np.mean(prices)), 1),
        peak_price=round(float(prices[peak_idx]), 1),
        peak_period=peak_idx,
        valley_price=round(float(prices[valley_idx]), 1),
        valley_period=valley_idx,
    )

    return PredictionsResponse(
        date=date_str,
        model_version=settings.feature_version,
        generated_at=datetime.now().isoformat(),
        predictions=preds,
        summary=summary,
    )


@app.get("/api/v1/predictions/latest", response_model=PredictionsResponse)
async def get_latest_predictions():
    """Get predictions for the most recent available date."""
    if not state["predictions_cache"]:
        raise HTTPException(503, "No predictions available.")
    latest = max(state["predictions_cache"].keys())
    return await get_predictions(date_str=latest)


@app.get("/api/v1/models", response_model=ModelsResponse)
async def list_models():
    """List active models."""
    model_list = []
    for key in sorted(state["models_s1"].keys()):
        model_list.append(ModelInfo(
            version_name=f"stage1_{key}",
            model_type=f"stage1_{key.split('_')[0]}",
            is_active=True,
        ))
    for key in sorted(state["models_s2"].keys()):
        model_list.append(ModelInfo(
            version_name=f"stage2_{key}",
            model_type="stage2_price",
            is_active=True,
        ))
    return ModelsResponse(models=model_list)


@app.get("/api/v1/chart", responses={200: {"content": {"image/png": {}}}})
async def prediction_chart(date_str: str = Query(None, alias="date")):
    """Return PNG chart of 96-period price prediction curve."""
    cache = state["predictions_cache"]
    if not cache:
        raise HTTPException(503, "Predictions not yet computed.")

    if date_str is None:
        date_str = max(cache.keys())
    if date_str not in cache:
        raise HTTPException(404, f"Date '{date_str}' not found.")

    prices_arr = np.asarray(cache[date_str], dtype=float)
    # Segment: period 0-35=base, 36-67=valley, 68-87=peak, 88-95=base
    seg_map = np.full(96, 'base', dtype=object)
    seg_map[36:68] = 'valley'
    seg_map[68:88] = 'peak'

    # ── Plot ──
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 5))
    seg_colors = {'valley': '#4CAF50', 'peak': '#F44336', 'base': '#2196F3'}
    seg_labels = {'valley': '午谷', 'peak': '晚峰', 'base': '基荷'}

    for seg in ['base', 'valley', 'peak']:
        mask = seg_map == seg
        ax.fill_between(range(96), 0, prices_arr, where=mask,
                        color=seg_colors[seg], alpha=0.3, label=seg_labels[seg])

    ax.plot(range(96), prices_arr, 'k-', lw=1.2, alpha=0.8)
    ax.set_xlim(0, 95)
    ax.set_xticks(range(0, 96, 4))
    ax.set_xticklabels([f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)][::4],
                       rotation=45, fontsize=7)
    ax.set_ylabel('元/MWh')
    ax.set_title(f'电价预测 — {date_str}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')


# ── Entrypoint ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
