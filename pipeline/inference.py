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
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

MODEL_DIR = Path(settings.model_dir)
FEAT_DIR = PROJECT_ROOT / "pipeline" / "output"
STAGE1_VARIABLES = ('solar', 'hydro', 'wind', 'load')
SEASONS = ('dry', 'wet')
STAGE2_SEGMENTS = ('valley', 'peak', 'base')

_STAGE2_OOF = FEAT_DIR / "price_oof.npz"


# ====================================================================
# Model loading
# ====================================================================

def _load_s1_for_lag(grid_lag=0):
    """Load Stage1 models for a specific grid_lag value."""
    lag_suffix = f"_lag{grid_lag}" if grid_lag > 0 else ""
    models = {}
    for var in STAGE1_VARIABLES:
        for season in SEASONS:
            key = f"{var}_{season}"
            path = MODEL_DIR / f"stage1_{var}_{season}{lag_suffix}.pkl"
            if path.exists():
                with open(path, 'rb') as f:
                    info = pickle.load(f)
                    models[key] = info
                    strategy = info.get('strategy', '?') if isinstance(info, dict) else 'raw'
                    n_feat = len(info.get('feat_indices', [])) if isinstance(info, dict) else '?'
                    logger.debug(f"  S1 {key}{lag_suffix}: strategy={strategy}, "
                                f"features={n_feat}, path={path.name}")
            else:
                logger.warning(f"  S1 {key}{lag_suffix}: model file not found — {path}")
    return models


def _load_s2_for_lag(grid_lag=0):
    """Load Stage2 models for a specific grid_lag value."""
    lag_suffix = f"_lag{grid_lag}" if grid_lag > 0 else ""
    models = {}
    for season in SEASONS:
        for seg in STAGE2_SEGMENTS:
            key = f"{seg}_{season}"
            path = MODEL_DIR / f"price_{seg}_{season}{lag_suffix}.pkl"
            if path.exists():
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    models[key] = obj
                    has_model = 'model' in obj
                    has_safe = 'safe_indices' in obj
                    logger.debug(f"  S2 {key}{lag_suffix}: has_model={has_model}, "
                                f"has_safe_indices={has_safe}, path={path.name}")
                else:
                    models[key] = {'model': obj, 'feat_names': None, 'safe_indices': None}
                    logger.debug(f"  S2 {key}{lag_suffix}: raw model (no metadata), path={path.name}")
            else:
                logger.warning(f"  S2 {key}{lag_suffix}: model file not found — {path}")
    return models


def load_stage1_models():
    """Load all Stage1 models (normal + lag variants). Returns {var_season: model_info}."""
    return _load_s1_for_lag(grid_lag=0)


def load_stage2_models():
    """Load 6 Stage2 period models (normal variant)."""
    return _load_s2_for_lag(grid_lag=0)


def load_model_set(grid_lag=0):
    """Load the Stage1 and Stage2 models for one grid lag variant."""
    return _load_s1_for_lag(grid_lag), _load_s2_for_lag(grid_lag)


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
    for var in STAGE1_VARIABLES:
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
    for var in STAGE1_VARIABLES:
        a = preds[var]
        valid = ~np.isnan(a)
        if valid.any():
            logger.debug(f"  S1 {var}/{season}: mean={a[valid].mean():.0f} "
                        f"range=[{a[valid].min():.0f}, {a[valid].max():.0f}] "
                        f"nan={np.isnan(a).sum()}")
        else:
            logger.warning(f"  S1 {var}/{season}: ALL NaN — model missing or failed")

    logger.info(f"  Stage1 {season}: done ({time.time()-t0:.1f}s)")
    return tuple(preds[var] for var in STAGE1_VARIABLES)


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


def season_masks_for_inference(dry_mask, wet_mask, dt_arr):
    """Map production inference rows to available seasonal model families.

    Training currently has dry (Jan-Apr) and wet (May-Jun) models only. For
    Jul-Sep rows we use the wet model as the closest fallback so forward-
    extended rows can still produce prices instead of all-zero output.
    """
    dry = np.asarray(dry_mask, dtype=bool).copy()
    wet = np.asarray(wet_mask, dtype=bool).copy()
    months = pd.DatetimeIndex(pd.to_datetime(dt_arr)).month.values
    wet |= np.isin(months, [7, 8, 9])
    dry &= ~wet
    return dry, wet


def _price_lag(price, periods):
    lagged = np.roll(price, periods)
    lagged[:periods] = np.nan
    return lagged


def price_anchor_from_lags(price, lags=(96, 672)):
    """Build the Stage2 price anchor from selected historical lag prices."""
    lag_arrays = [_price_lag(price, p) for p in lags]
    stacked = np.vstack(lag_arrays)
    valid = np.isfinite(stacked)
    count = valid.sum(axis=0)
    total = np.where(valid, stacked, 0.0).sum(axis=0)
    anchor = np.full(len(price), np.nan)
    np.divide(total, count, out=anchor, where=count > 0)
    lag96 = _price_lag(price, 96)
    lag672 = _price_lag(price, 672)
    return anchor, lag96, lag672


def stage2_guard_policy(models, season):
    """Read serving fallback/blend policy from Stage2 model metadata."""
    for seg in STAGE2_SEGMENTS:
        info = models.get(f"{seg}_{season}")
        if isinstance(info, dict) and isinstance(info.get('guard_policy'), dict):
            return info['guard_policy']
    return {'enabled': False, 'mode': 'model', 'reason': 'no_guard_policy'}


def apply_stage2_guard(price_model, anchor, lag96, policy):
    """Apply recent-performance guard to a season prediction slice."""
    mode = policy.get('mode', 'model') if policy else 'model'
    if not policy or not policy.get('enabled') or mode == 'model':
        return price_model.copy()

    if mode == 'lag96':
        baseline = lag96
        model_weight = 0.0
    elif mode == 'blend':
        baseline_name = policy.get('baseline', 'lag96')
        baseline = lag96 if baseline_name == 'lag96' else anchor
        model_weight = float(policy.get('model_weight', 0.30))
    else:
        return price_model.copy()

    finite_model = np.isfinite(price_model)
    finite_base = np.isfinite(baseline)
    guarded = price_model.copy()
    guarded = np.where(
        finite_model & finite_base,
        model_weight * price_model + (1.0 - model_weight) * baseline,
        guarded,
    )
    guarded = np.where(~finite_model & finite_base, baseline, guarded)
    return guarded


def predict_from_features(models_s1, models_s2, X_full, feat_names, period,
                          price, dry_mask, wet_mask, dt_arr,
                          anchor_lags=(96, 672), log_label=""):
    """Run the shared Stage1 -> Stage2 inference path for a feature matrix."""
    if not models_s2:
        raise RuntimeError("No Stage2 models loaded")

    n = X_full.shape[0]
    dry_mask, wet_mask = season_masks_for_inference(dry_mask, wet_mask, dt_arr)
    season_masks = {'dry': dry_mask, 'wet': wet_mask}
    stage1 = {var: np.full(n, np.nan) for var in STAGE1_VARIABLES}

    for season, mask in season_masks.items():
        predictions = predict_stage1(models_s1, X_full, feat_names, season)
        for var, values in zip(STAGE1_VARIABLES, predictions):
            stage1[var][mask] = values[mask]

    for var, values in stage1.items():
        nan_count = np.isnan(values).sum()
        if nan_count:
            logger.warning(f"  Stage1 {log_label}{var}: {nan_count} NaN -> filled with 0")
            values[np.isnan(values)] = 0.0

    first_stage2 = next(iter(models_s2.values()))
    safe_indices = (first_stage2.get('safe_indices')
                    if isinstance(first_stage2, dict) else None)
    X_stage2 = build_stage2_features(
        X_full, feat_names,
        stage1['solar'], stage1['hydro'], stage1['wind'], stage1['load'],
        period, safe_indices=safe_indices,
    )
    anchor, lag96, lag672 = price_anchor_from_lags(price, lags=anchor_lags)
    price_pred = np.full(n, np.nan)

    for season, mask in season_masks.items():
        idx = np.where(mask)[0]
        if not len(idx):
            continue

        segment_predictions = {}
        for segment in STAGE2_SEGMENTS:
            key = f"{segment}_{season}"
            model_info = models_s2.get(key)
            if model_info is None:
                logger.warning(f"  S2 {log_label}{key}: model missing -> using zeros")
                segment_predictions[segment] = np.zeros(len(idx))
                continue
            model = model_info['model'] if isinstance(model_info, dict) else model_info
            segment_predictions[segment] = model.predict(X_stage2[idx])

        weights = blend_weights(period[idx])
        blended = (
            weights[:, 0] * segment_predictions['valley']
            + weights[:, 1] * segment_predictions['peak']
            + weights[:, 2] * segment_predictions['base']
        )
        raw_price = anchor[idx] + blended
        guard_policy = stage2_guard_policy(models_s2, season)
        price_pred[idx] = apply_stage2_guard(
            raw_price, anchor[idx], lag96[idx], guard_policy)
        if guard_policy.get('enabled'):
            logger.info(f"  S2 {log_label}{season}: guard={guard_policy.get('mode')} "
                        f"reason={guard_policy.get('reason')}")

    resid_pred = price_pred - anchor
    nan_count = np.isnan(price_pred).sum()
    if nan_count:
        logger.warning(f"  {log_label}price_pred: {nan_count} NaN -> filled with 0")
        price_pred[np.isnan(price_pred)] = 0.0

    return {
        'price_pred': price_pred,
        'resid_pred': resid_pred,
        'anchor': anchor,
        'price_lag96': lag96,
        'price_lag672': lag672,
        'dry_mask': dry_mask,
        'wet_mask': wet_mask,
        'stage1': stage1,
    }


# ====================================================================
# Main prediction
# ====================================================================

def run_inference(grid_lag=0):
    """Full inference chain. Returns dict with all predictions.

    Parameters
    ----------
    grid_lag : int — if > 0, load _lag{N} feature files (gap-fill mode).
    """
    t_total = time.time()
    lag_suffix = f"_lag{grid_lag}" if grid_lag > 0 else ""
    logger.info("=" * 50)
    logger.info(f"Inference Pipeline (Stage1 + Stage2){lag_suffix}")

    # 1. Load features
    t0 = time.time()
    feat_path = FEAT_DIR / f"features_15min_dry{lag_suffix}.npz"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}")

    data = np.load(feat_path, allow_pickle=True)
    X_full = data['X']
    feat_names = data['feat_names']
    period = data['period']
    price = data['price']
    dt_arr = data['dt']

    n = X_full.shape[0]
    logger.info(f"  Features: {n} rows × {X_full.shape[1]} dims  "
                f"({time.time()-t0:.1f}s)")

    # 2. Load models
    t0 = time.time()
    m1, m2 = load_model_set(grid_lag)
    logger.info(f"  Models: S1={len(m1)}, S2={len(m2)}  ({time.time()-t0:.1f}s)")

    anchor_lags = (grid_lag, 672) if grid_lag > 0 else (96, 672)
    result = predict_from_features(
        m1, m2, X_full, feat_names, period, price,
        data['dry_mask'], data['wet_mask'], dt_arr,
        anchor_lags=anchor_lags,
        log_label=f"lag{grid_lag} " if grid_lag else "",
    )
    logger.info(f"  Inference done: {time.time()-t_total:.1f}s total")

    result.update({
        'price': price,
        'dt': dt_arr,
        'period': period,
        'oof_s': result['stage1']['solar'],
        'oof_h': result['stage1']['hydro'],
        'oof_w': result['stage1']['wind'],
        'oof_l': result['stage1']['load'],
    })
    return result


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
    print(f"    R2:   {r2:.4f}")

    for season, smask in [('dry', ref['dry_mask']), ('wet', ref['wet_mask'])]:
        sv = valid & smask.astype(bool)
        if sv.sum() > 0:
            s_mae = np.mean(np.abs(ref_price[sv] - our_price[sv]))
            s_r2 = 1 - np.sum((ref_price[sv] - our_price[sv]) ** 2) / max(
                np.sum((ref_price[sv] - np.mean(ref_price[sv])) ** 2), 1e-10)
            print(f"    {season}: MAE={s_mae:.2f}, R2={s_r2:.4f}, n={sv.sum():,}")

    ok = mae < 50.0
    status = 'passed' if ok else '- significant deviation'
    print(f"\n  {'[OK]' if ok else '[WARN]'} Verification {status}")
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
