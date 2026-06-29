# -*- coding: utf-8 -*-
"""
pipeline/inference.py — Full prediction chain (Stage1 + Stage2)
===============================================================
Loads all models, runs the complete inference pipeline:
  Features → Stage1 (4 vars) → Stage2 features (80 dims) → 3-period blend → 96 prices

Usage:
    python -m pipeline.inference                     # Predict all data
    python -m pipeline.inference --verify            # Compare with price_oof.npz
"""
import sys
import time
import logging
import warnings
import pickle
import argparse
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

MODEL_DIR = Path(settings.model_dir)
FEAT_DIR = PROJECT_ROOT / "pipeline" / "output"

# Stage2 models (local project)
_STAGE2_MODEL_DIR = MODEL_DIR
_STAGE2_OOF = FEAT_DIR / "price_oof.npz"


# ====================================================================
# Model loading
# ====================================================================

def load_stage1_models():
    """Load all 8 Stage1 models. Returns {var_season: model_info}."""
    models = {}
    for var in ['solar', 'hydro', 'wind', 'load']:
        for season in ['dry', 'wet']:
            key = f"{var}_{season}"
            path = MODEL_DIR / f"stage1_{var}_{season}.pkl"
            if path.exists():
                with open(path, 'rb') as f:
                    info = pickle.load(f)
                    models[key] = info
                    strategy = info.get('strategy', '?') if isinstance(info, dict) else 'raw'
                    n_feat = len(info.get('feat_indices', [])) if isinstance(info, dict) else '?'
                    logger.debug(f"  S1 {key}: strategy={strategy}, "
                                f"features={n_feat}, path={path.name}")
            else:
                logger.warning(f"  S1 {key}: model file not found — {path}")
    return models


def load_stage2_models():
    """Load 6 Stage2 period models (valley/peak/base × dry/wet)."""
    models = {}
    for season in ['dry', 'wet']:
        for seg in ['valley', 'peak', 'base']:
            key = f"{seg}_{season}"
            path = _STAGE2_MODEL_DIR / f"price_{seg}_{season}.pkl"
            if path.exists():
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    models[key] = obj
                    has_model = 'model' in obj
                    has_safe = 'safe_indices' in obj
                    logger.debug(f"  S2 {key}: has_model={has_model}, "
                                f"has_safe_indices={has_safe}, path={path.name}")
                else:
                    models[key] = {'model': obj, 'feat_names': None, 'safe_indices': None}
                    logger.debug(f"  S2 {key}: raw model (no metadata), path={path.name}")
            else:
                logger.warning(f"  S2 {key}: model file not found — {path}")
    return models


# ====================================================================
# Stage1 prediction
# ====================================================================

def predict_stage1(models, X_full, feat_names, season):
    """Run Stage1 predictions for one season.

    Returns oof_s, oof_h, oof_w, oof_l arrays (same length as X).
    """
    all_names = list(feat_names)
    n = X_full.shape[0]
    preds = {}

    t0 = time.time()
    for var in ['solar', 'hydro', 'wind', 'load']:
        key = f"{var}_{season}"
        if key not in models:
            logger.warning(f"  predict_stage1: {key} not in loaded models → all NaN")
            preds[var] = np.full(n, np.nan)
            continue

        info = models[key]
        fidx = info['feat_indices']
        strategy = info['strategy']
        model = info['model']

        X_var = X_full[:, fidx]
        resid = model.predict(X_var)

        if strategy == 'direct absolute':
            preds[var] = resid
        elif 'lag_672' in strategy:
            hydro_t_idx = all_names.index('B_hydro[t]')
            hydro_t = X_full[:, hydro_t_idx]
            lag_672 = np.roll(hydro_t, 672)
            lag_672[:672] = np.nan
            preds[var] = resid + lag_672
        elif 'lag_96' in strategy:
            idx_map = {'hydro': 'B_hydro[t]', 'wind': 'B_wind[t]', 'load': 'B_load[t]'}
            t_idx = all_names.index(idx_map[var])
            var_t = X_full[:, t_idx]
            lag_96 = np.roll(var_t, 96)
            lag_96[:96] = np.nan
            preds[var] = resid + lag_96
        else:
            preds[var] = resid

    # Log per-variable stats
    for var in ['solar', 'hydro', 'wind', 'load']:
        a = preds[var]
        valid = ~np.isnan(a)
        if valid.any():
            logger.debug(f"  S1 {var}/{season}: mean={a[valid].mean():.0f} "
                        f"range=[{a[valid].min():.0f}, {a[valid].max():.0f}] "
                        f"nan={np.isnan(a).sum()}")
        else:
            logger.warning(f"  S1 {var}/{season}: ALL NaN — model missing or failed")

    logger.info(f"  Stage1 {season}: done ({time.time()-t0:.1f}s)")
    return preds['solar'], preds['hydro'], preds['wind'], preds['load']


# ====================================================================
# Stage2 feature construction (identical to train_price_stage2.py)
# ====================================================================

def build_stage2_features(X_full, feat_names, oof_s, oof_h, oof_w, oof_l, period,
                           safe_indices=None):
    """Build Stage2 input from Stage1 predictions + safe features.

    If safe_indices is provided (from training metadata), use it directly.
    Otherwise, select dynamically based on naming rules.
    """
    all_names = list(feat_names)
    period = np.asarray(period, dtype=int)

    if safe_indices is None:
        # Dynamic selection (fallback, matches training logic)
        allowed_a = {'A_price_lag_96', 'A_price_lag_192', 'A_price_lag_288', 'A_price_lag_672',
                     'A_price_ma_24h', 'A_price_ma_7d', 'A_price_vol_24h', 'A_price_max_30d'}
        allowed_b_prefixes = ('B_solar_lag_96', 'B_solar_lag_672', 'B_wind_lag_96', 'B_wind_lag_672',
                              'B_hydro_lag_96', 'B_hydro_lag_672', 'B_load_lag_96', 'B_load_lag_672')
        allowed_c = {'C_net_load_lag_96', 'C_surplus_lag_96'}
        allowed_j = {'J_rad_pp_x_period_cos', 'J_rain72h_x_flood'}

        safe_indices = []
        for i, n in enumerate(all_names):
            include = False
            if n in allowed_a: include = True
            elif n.startswith('B_') and any(n.startswith(p) for p in allowed_b_prefixes): include = True
            elif n in allowed_c: include = True
            elif n.startswith('D_') and n.endswith('_pp'): include = True
            elif n.startswith('F_'): include = True
            elif n.startswith('G_'): include = True
            elif n.startswith('H_'): include = True
            elif n.startswith('I_'): include = True
            elif n in allowed_j: include = True
            elif n.startswith('L_'): include = True
            elif n.startswith('M_'): include = True
            if include: safe_indices.append(i)

    # Build X: 4 OOF + safe features + 6 interactions
    cols = [oof_s.reshape(-1, 1), oof_h.reshape(-1, 1),
            oof_w.reshape(-1, 1), oof_l.reshape(-1, 1),
            X_full[:, safe_indices]]

    period_cos = np.cos(2 * np.pi * period / 96)
    slot_valley = ((period >= 36) & (period <= 67)).astype(np.float32)
    slot_peak   = ((period >= 68) & (period <= 87)).astype(np.float32)

    cols.extend([
        (oof_s * period_cos).reshape(-1, 1),
        (oof_s * slot_valley).reshape(-1, 1),
        (oof_l * slot_peak).reshape(-1, 1),
        (oof_s * oof_l).reshape(-1, 1),
        ((oof_s + oof_w) * slot_valley).reshape(-1, 1),
        (oof_l * period_cos).reshape(-1, 1),
    ])

    return np.concatenate(cols, axis=1)


def blend_weights(period):
    """Soft blend weights for valley/peak/base period transitions."""
    n = len(period)
    w = np.zeros((n, 3))
    for i, p in enumerate(period):
        if   p <= 31:                         w[i] = [0, 0, 1]
        elif p <= 35: t = (p - 32) / 4;       w[i] = [t, 0, 1 - t]
        elif p <= 63:                         w[i] = [1, 0, 0]
        elif p <= 67: t = (p - 64) / 4;       w[i] = [1 - t, t, 0]
        elif p <= 83:                         w[i] = [0, 1, 0]
        elif p <= 87: t = (p - 84) / 4;       w[i] = [0, 1 - t, t]
        else:                                 w[i] = [0, 0, 1]
    return w


# ====================================================================
# Main prediction
# ====================================================================

def run_inference():
    """Full inference chain. Returns dict with all predictions."""
    t_total = time.time()
    logger.info("=" * 50)
    logger.info("Inference Pipeline (Stage1 + Stage2)")

    # 1. Load features
    t0 = time.time()
    feat_path = FEAT_DIR / "features_15min_dry.npz"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}")

    data = np.load(feat_path, allow_pickle=True)
    X_full = data['X']
    feat_names = data['feat_names']
    period = data['period']
    price = data['price']
    dry_mask = data['dry_mask']
    wet_mask = data['wet_mask']
    dt_arr = data['dt']

    n = X_full.shape[0]
    logger.info(f"  Features: {n} rows × {X_full.shape[1]} dims  "
                f"({time.time()-t0:.1f}s)")

    # 2. Load models
    t0 = time.time()
    m1 = load_stage1_models()
    m2 = load_stage2_models()
    logger.info(f"  Models: S1={len(m1)}, S2={len(m2)}  ({time.time()-t0:.1f}s)")

    # 3. Stage1
    t0 = time.time()
    oof_s = np.full(n, np.nan); oof_h = np.full(n, np.nan)
    oof_w = np.full(n, np.nan); oof_l = np.full(n, np.nan)

    for season, mask in [('dry', dry_mask), ('wet', wet_mask)]:
        s, h, w, l = predict_stage1(m1, X_full, feat_names, season)
        oof_s[mask] = s[mask]; oof_h[mask] = h[mask]
        oof_w[mask] = w[mask]; oof_l[mask] = l[mask]

    for name, a in [('solar', oof_s), ('hydro', oof_h), ('wind', oof_w), ('load', oof_l)]:
        nan_n = np.isnan(a).sum()
        if nan_n > 0:
            logger.warning(f"  Stage1 {name}: {nan_n} NaN OOF → filled with 0")
        a[np.isnan(a)] = 0.0

    print(f"    Solar: mean={np.mean(oof_s):.0f} MW")
    print(f"    Hydro: mean={np.mean(oof_h):.0f} MW")
    print(f"    Wind:  mean={np.mean(oof_w):.0f} MW")
    print(f"    Load:  mean={np.mean(oof_l):.0f} MW")

    # 4. Stage2 — use feature indices from training metadata
    t0 = time.time()
    if not m2:
        logger.error("No Stage2 models loaded — cannot run inference")
        raise RuntimeError("No Stage2 models loaded")
    first_m2 = next(iter(m2.values()))
    safe_idx = first_m2.get('safe_indices') if isinstance(first_m2, dict) else None
    X_s2 = build_stage2_features(X_full, feat_names, oof_s, oof_h, oof_w, oof_l,
                                  period, safe_indices=safe_idx)
    logger.info(f"  Stage2 input: {X_s2.shape[1]} dims  ({time.time()-t0:.1f}s)")

    lag96 = np.roll(price, 96); lag96[:96] = np.nan
    lag672 = np.roll(price, 672); lag672[:672] = np.nan
    anchor = (lag96 + lag672) / 2.0

    price_pred = np.full(n, np.nan)
    resid_pred = np.full(n, np.nan)

    for season, mask in [('dry', dry_mask), ('wet', wet_mask)]:
        idx = np.where(mask)[0]
        if len(idx) == 0: continue

        is_dry = (season == 'dry')
        seg_preds = {}
        for seg in ['valley', 'peak', 'base']:
            key = f"{seg}_{season}"
            if key in m2:
                m = m2[key]
                model = m['model'] if isinstance(m, dict) else m
                seg_preds[seg] = model.predict(X_s2[idx])
            else:
                logger.warning(f"  S2 {key}: model missing → using zeros")
                seg_preds[seg] = np.zeros(len(idx))

        w = blend_weights(period[idx])
        blended = (w[:, 0] * seg_preds['valley'] +
                   w[:, 1] * seg_preds['peak'] +
                   w[:, 2] * seg_preds['base'])

        if is_dry:
            resid_pred[idx] = blended
            price_pred[idx] = anchor[idx] + blended
        else:
            price_pred[idx] = blended
            resid_pred[idx] = blended - anchor[idx]

        # Per-season summary
        valid_p = ~np.isnan(price_pred[idx])
        if valid_p.any():
            logger.info(f"  S2 {season}: pred mean={price_pred[idx][valid_p].mean():.1f} "
                       f"range=[{price_pred[idx][valid_p].min():.1f}, "
                       f"{price_pred[idx][valid_p].max():.1f}]")

    nan_price = np.isnan(price_pred).sum()
    if nan_price > 0:
        logger.warning(f"  price_pred: {nan_price} NaN → filled with 0")
    price_pred[np.isnan(price_pred)] = 0.0
    logger.info(f"  Inference done: {time.time()-t_total:.1f}s total")

    return {
        'price_pred': price_pred, 'resid_pred': resid_pred,
        'price': price, 'price_lag96': lag96, 'anchor': anchor,
        'oof_s': oof_s, 'oof_h': oof_h, 'oof_w': oof_w, 'oof_l': oof_l,
        'dry_mask': dry_mask, 'wet_mask': wet_mask,
        'dt': dt_arr, 'period': period,
    }


# ====================================================================
# Verification
# ====================================================================

def verify(result):
    """Compare inference output with price_oof.npz."""
    if not _STAGE2_OOF.exists():
        print(f"\n  [WARN] Reference OOF not found: {_STAGE2_OOF}")
        return False

    print("\n" + "=" * 60)
    print("  Verification: inference vs price_oof.npz")
    print("=" * 60)

    ref = np.load(_STAGE2_OOF, allow_pickle=True)
    ref_price = ref['oof_price']

    our_price = result['price_pred']
    valid = ~np.isnan(ref_price) & ~np.isnan(our_price)

    if valid.sum() == 0:
        print("  [FAIL] No valid overlap!")
        return False

    abs_diff = np.abs(ref_price[valid] - our_price[valid])
    mae = np.mean(abs_diff)
    max_d = np.max(abs_diff)
    r2 = 1 - np.sum((ref_price[valid] - our_price[valid]) ** 2) / np.sum(
        (ref_price[valid] - np.mean(ref_price[valid])) ** 2)

    print(f"    Valid points: {valid.sum():,}")
    print(f"    MAE:  {mae:.2f} 元/MWh")
    print(f"    Max:  {max_d:.2f} 元/MWh")
    print(f"    R²:   {r2:.4f}")

    for season, smask in [('dry', ref['dry_mask']), ('wet', ref['wet_mask'])]:
        sv = valid & smask.astype(bool)
        if sv.sum() > 0:
            s_mae = np.mean(np.abs(ref_price[sv] - our_price[sv]))
            s_r2 = 1 - np.sum((ref_price[sv] - our_price[sv]) ** 2) / max(
                np.sum((ref_price[sv] - np.mean(ref_price[sv])) ** 2), 1e-10)
            print(f"    {season}: MAE={s_mae:.2f}, R²={s_r2:.4f}, n={sv.sum():,}")

    ok = mae < 50.0
    print(f"\n  {'[OK]' if ok else '[WARN]'} Verification {'passed' if ok else '— significant deviation'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    result = run_inference()

    if args.verify:
        verify(result)

    # Print last-day sample
    pp = result['price_pred']
    valid = ~np.isnan(pp) & (pp > 0)
    if valid.sum() >= 96:
        start = len(pp) - 96
        print(f"\n  Last day predictions:")
        for p in range(0, 96, 8):
            h = p // 4; m = (p % 4) * 15
            print(f"    period {p:2d} ({h:02d}:{m:02d}): {pp[start + p]:8.1f} 元/MWh")

    print(f"\n{'=' * 70}\n  Inference complete!\n{'=' * 70}")


if __name__ == "__main__":
    main()
