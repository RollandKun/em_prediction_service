# -*- coding: utf-8 -*-
"""
pipeline/output.py — 特征矩阵保存 + 验证
========================================
从 build_features() 的结果生成枯水/丰水 npz 文件,
并与 EM_Pre3 参考输出做差异对比。

Usage:
    from pipeline.output import save_outputs, verify
    fp_dry, fp_wet = save_outputs(result)
    verify()
"""
import sys
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ──
OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

EM_PRE3_OUTPUT = (
    PROJECT_ROOT.parent / "EM_Pre3" / "Stage1" / "feature" / "output"
)

# Split dates (identical to features_15min.py)
_DRY_TRAIN_START = np.datetime64("2026-01-02")
_DRY_TRAIN_END   = np.datetime64("2026-03-15")
_DRY_VAL_START   = np.datetime64("2026-03-16")
_DRY_VAL_END     = np.datetime64("2026-04-07")
_DRY_TEST_START  = np.datetime64("2026-04-08")
_DRY_TEST_END    = np.datetime64("2026-04-30")

_WET_TRAIN_START = np.datetime64("2026-05-01")
_WET_TRAIN_END   = np.datetime64("2026-05-20")
_WET_VAL_START   = np.datetime64("2026-05-21")
_WET_VAL_END     = np.datetime64("2026-05-28")
_WET_TEST_START  = np.datetime64("2026-05-29")
_WET_TEST_END    = np.datetime64("2026-06-06")
# Expanded training to 6/27 was attempted but degraded wet-season Stage2
# (-0.93 vs +0.26 R²). Late June has fundamentally different pricing dynamics
# from early June. Will revisit when 7-8月 main flood season data accumulates.


def _build_split_indices(dt_arr, mask, train_start, train_end,
                         val_start, val_end, test_start, test_end,
                         y_price_resid):
    """Build Train/Val/Test indices for one season."""
    n = len(dt_arr)
    train_mask = mask & (dt_arr >= train_start) & (dt_arr <= train_end)
    val_mask   = mask & (dt_arr >= val_start)   & (dt_arr <= val_end)
    test_mask  = mask & (dt_arr >= test_start)  & (dt_arr <= test_end)
    valid = ~np.isnan(y_price_resid) if mask.sum() > 0 else np.zeros(n, dtype=bool)
    idx_train = np.where(train_mask & valid)[0]
    idx_val   = np.where(val_mask & valid)[0]
    idx_test  = np.where(test_mask & valid)[0]
    return idx_train, idx_val, idx_test


def save_outputs(result, grid_lag=0):
    """Save dry and wet season npz files to pipeline/output/.

    Parameters
    ----------
    result : dict — from build_features()
    grid_lag : int — if > 0, suffix appended to filename (e.g. _lag192)
    """
    dt_arr = result['dt']
    y_price_resid = result['y_price_resid']
    mask_dry = result['dry_mask']
    mask_wet = result['wet_mask']

    idx_dry_train, idx_dry_val, idx_dry_test = _build_split_indices(
        dt_arr, mask_dry, _DRY_TRAIN_START, _DRY_TRAIN_END,
        _DRY_VAL_START, _DRY_VAL_END, _DRY_TEST_START, _DRY_TEST_END,
        y_price_resid)

    idx_wet_train, idx_wet_val, idx_wet_test = _build_split_indices(
        dt_arr, mask_wet, _WET_TRAIN_START, _WET_TRAIN_END,
        _WET_VAL_START, _WET_VAL_END, _WET_TEST_START, _WET_TEST_END,
        y_price_resid)

    print(f"\n  时期切分:")
    print(f"    枯水: Train={len(idx_dry_train)}, Val={len(idx_dry_val)}, "
          f"Test={len(idx_dry_test)}")
    print(f"    丰水: Train={len(idx_wet_train)}, Val={len(idx_wet_val)}, "
          f"Test={len(idx_wet_test)}")

    # Filename suffix for gap-fill variants
    lag_suffix = f"_lag{grid_lag}" if grid_lag > 0 else ""

    def _save_one(suffix, idx_train, idx_val, idx_test):
        out = {
            'X': result['X'],
            'feat_names': result['feat_names'],
            'y_solar': result['y_solar'],
            'y_hydro': result['y_hydro'],
            'y_wind': result['y_wind'],
            'y_price_resid': result['y_price_resid'],
            'y_price_raw': result['y_price_raw'],
            'price_lag96': result['price_lag96'],
            'price': result['price'],
            'idx_train': idx_train,
            'idx_val': idx_val,
            'idx_test': idx_test,
            'dt': result['dt'],
            'period': result['period'],
            'month': result['month'],
            'n_samples': result['n_samples'],
            'dry_mask': result['dry_mask'],
            'wet_mask': result['wet_mask'],
            'solar_feat_cols': result['solar_feat_cols'],
            'hydro_feat_cols': result['hydro_feat_cols'],
            'wind_feat_cols': result['wind_feat_cols'],
            'load_feat_cols': result['load_feat_cols'],
        }
        fp = OUTPUT_DIR / f'features_15min_{suffix}{lag_suffix}.npz'
        np.savez_compressed(fp, **out)
        print(f"  保存: {fp}")
        return fp

    fp_dry = _save_one('dry', idx_dry_train, idx_dry_val, idx_dry_test)
    fp_wet = _save_one('wet', idx_wet_train, idx_wet_val, idx_wet_test)
    return fp_dry, fp_wet


def verify():
    """Compare DB-backed feature output with original Excel-backed npz files."""
    print("\n" + "=" * 60)
    print("  Verification: DB vs Excel-backed features")
    print("=" * 60)

    all_ok = True

    for suffix in ['dry', 'wet']:
        ref_path = EM_PRE3_OUTPUT / f'features_15min_{suffix}.npz'
        new_path = OUTPUT_DIR / f'features_15min_{suffix}.npz'

        if not ref_path.exists():
            print(f"\n  [WARN] Reference file not found: {ref_path}")
            print(f"    Skipping {suffix} verification.")
            all_ok = False
            continue

        if not new_path.exists():
            print(f"\n  [WARN] New file not found: {new_path}")
            print(f"    Run without --verify-only first.")
            all_ok = False
            continue

        print(f"\n  ── {suffix} season ──")

        ref = np.load(ref_path, allow_pickle=True)
        new = np.load(new_path, allow_pickle=True)

        # Compare feature matrices
        X_ref = ref['X']
        X_new = new['X']

        print(f"    Reference X shape: {X_ref.shape}")
        print(f"    New X shape:       {X_new.shape}")

        if X_ref.shape != X_new.shape:
            print(f"    [FAIL] Shape mismatch!")
            all_ok = False
            continue

        # Element-wise comparison
        abs_diff = np.abs(X_ref - X_new)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        n_diff = np.sum(~np.isclose(X_ref, X_new, rtol=1e-10, atol=1e-12))

        print(f"    Max absolute diff:  {max_diff:.2e}")
        print(f"    Mean absolute diff: {mean_diff:.2e}")
        print(f"    Elements with diff > 1e-10: {n_diff} / {X_ref.size} "
              f"({100*n_diff/X_ref.size:.4f}%)")

        if max_diff < 1e-10:
            print(f"    [OK] Feature matrix is identical (max_diff < 1e-10)")
        elif max_diff < 1e-6:
            print(f"    [OK] Feature matrix matches within 1e-6 (float32-level agreement)")
            all_ok = False
        else:
            print(f"    [FAIL] Significant differences detected!")
            worst_idx = np.unravel_index(np.argmax(abs_diff), X_ref.shape)
            print(f"    Worst at row={worst_idx[0]}, col={worst_idx[1]}: "
                  f"ref={X_ref[worst_idx]:.6f} new={X_new[worst_idx]:.6f}")

            col_diffs = np.max(np.abs(X_ref - X_new), axis=0)
            worst_cols = np.argsort(-col_diffs)[:5]
            if 'feat_names' in ref:
                fn = ref['feat_names']
                print(f"    Worst 5 features:")
                for ci in worst_cols:
                    print(f"      {fn[ci]}: max_diff={col_diffs[ci]:.6f}")

            all_ok = False

        # Compare targets
        for target in ['y_solar', 'y_hydro', 'y_wind', 'y_price_resid']:
            if target in ref and target in new:
                t_ref = ref[target]
                t_new = new[target]
                t_nan_ref = np.isnan(t_ref)
                t_nan_new = np.isnan(t_new)
                nan_match = np.array_equal(t_nan_ref, t_nan_new)
                t_valid = ~t_nan_ref & ~t_nan_new
                if t_valid.sum() > 0:
                    t_diff = np.max(np.abs(t_ref[t_valid] - t_new[t_valid]))
                else:
                    t_diff = 0.0
                status = '[OK]' if (t_diff < 1e-10 and nan_match) else '[WARN]'
                print(f"    {target}: max_diff={t_diff:.2e} nan_match={nan_match} {status}")

        # Compare split indices
        for idx_name in ['idx_train', 'idx_val', 'idx_test']:
            if idx_name in ref and idx_name in new:
                i_ref = ref[idx_name]
                i_new = new[idx_name]
                if np.array_equal(i_ref, i_new):
                    print(f"    {idx_name}: [OK] ({len(i_ref)} indices match)")
                else:
                    print(f"    {idx_name}: [WARN] indices differ "
                          f"(ref={len(i_ref)}, new={len(i_new)}, "
                          f"intersection={len(np.intersect1d(i_ref, i_new))})")
                    all_ok = False

    if all_ok:
        print(f"\n  [OK] All verifications passed!")
    else:
        print(f"\n  [WARN] Some verifications failed — see details above.")

    return all_ok
