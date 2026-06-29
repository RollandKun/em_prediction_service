# -*- coding: utf-8 -*-
"""
pipeline/train_stage2.py — Stage2 电价预测训练（Phase 5）
==========================================================
策略（v13 最终版）：
  枯水期 (dry):  RandomForest 预测 price[t+96] - anchor（残差），anchor = (lag96 + lag672)/2
  丰水期 (wet):  XGBoost 直接预测 price[t+96] 绝对值

三时段分别建模：valley / peak / base，推理时用余弦软融合。

产出：
  models/price_{valley,peak,base}_{dry,wet}.pkl  (6 files)
  pipeline/output/price_oof.npz                    (OOF predictions)
  pipeline/output/reports/price_stage2_report.md   (评估报告)

Usage:
  python -m pipeline.train_stage2                      # Full training + save
  python -m pipeline.train_stage2 --no-versioning      # Skip model_versions write
"""
import sys
import io
import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，不显示图形窗口

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

# ── 路径 ──
FEAT_DIR = PROJECT_ROOT / "pipeline" / "output"
OOF_DIR = FEAT_DIR  # Stage1 OOF 与特征在同一目录
MODEL_DIR = Path(settings.model_dir)
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = FEAT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
FEAT_DRY = FEAT_DIR / "features_15min_dry.npz"
FEAT_WET = FEAT_DIR / "features_15min_wet.npz"


# ====================================================================
# 1. 特征加载与 Stage2 输入构建
# ====================================================================

def load_all_data(grid_lag=0):
    """加载特征 + Stage1 OOF → 构建 Stage2 80维输入。

    完全移植自 EM_Pre3/Stage2/train_price_stage2.py 的 load_all_data()。

    Parameters
    ----------
    grid_lag : int — if > 0, load _lag{N} feature and OOF files.
    """
    lag_suffix = f"_lag{grid_lag}" if grid_lag > 0 else ""
    feat_dry_path = FEAT_DIR / f"features_15min_dry{lag_suffix}.npz"
    feat_wet_path = FEAT_DIR / f"features_15min_wet{lag_suffix}.npz"

    print("=" * 60)
    print(f"  Stage 2 — 混合策略 (枯水 RF 残差 + 丰水 XGB 绝对值){lag_suffix}")
    print("=" * 60)

    if not feat_dry_path.exists():
        raise FileNotFoundError(f"特征文件不存在: {feat_dry_path}。请先运行 pipeline/feature_engine.py --grid-lag {grid_lag}")

    data_dry = np.load(feat_dry_path, allow_pickle=True)
    data_wet = np.load(feat_wet_path, allow_pickle=True)
    all_names = data_dry['feat_names'].tolist()

    # ── 安全特征筛选（与 EM_Pre3 完全一致） ──
    allowed_a = {'A_price_lag_96', 'A_price_lag_192', 'A_price_lag_288', 'A_price_lag_672',
                 'A_price_ma_24h', 'A_price_ma_7d', 'A_price_vol_24h', 'A_price_max_30d'}
    allowed_b_prefixes = ('B_solar_lag_96', 'B_solar_lag_672', 'B_wind_lag_96', 'B_wind_lag_672',
                          'B_hydro_lag_96', 'B_hydro_lag_672', 'B_load_lag_96', 'B_load_lag_672')
    allowed_c = {'C_net_load_lag_96', 'C_surplus_lag_96'}
    allowed_j = {'J_rad_pp_x_period_cos', 'J_rain72h_x_flood'}

    safe_indices, safe_names = [], []
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
        if include:
            safe_indices.append(i)
            safe_names.append(n)

    print(f"  安全特征: {len(safe_indices)} dims")

    # ── 加载 Stage1 OOF ──
    # 如果本地 OOF 文件存在（刚跑完 train_stage1），优先使用
    # 否则回退到 EM_Pre3 的参考 OOF
    oof_solar = _load_oof('solar', grid_lag)
    oof_hydro = _load_oof('hydro', grid_lag)
    oof_wind  = _load_oof('wind', grid_lag)
    oof_load  = _load_oof('load', grid_lag)
    print(f"  Stage1 OOF loaded: solar={oof_solar.shape}, hydro={oof_hydro.shape}, "
          f"wind={oof_wind.shape}, load={oof_load.shape}")

    # ── 构建 Stage2 80维输入 ──
    def build_X(data, oof_s, oof_h, oof_w, oof_l):
        """4 OOF + 70 safe + 6 interaction = 80 dims."""
        cols = [oof_s.reshape(-1, 1), oof_h.reshape(-1, 1),
                oof_w.reshape(-1, 1), oof_l.reshape(-1, 1),
                data['X'][:, safe_indices]]
        period_col = data['period']
        period_cos = np.cos(2 * np.pi * period_col / 96)
        slot_valley = ((period_col >= 36) & (period_col <= 67)).astype(np.float32)
        slot_peak   = ((period_col >= 68) & (period_col <= 87)).astype(np.float32)
        cols.extend([
            (oof_s * period_cos).reshape(-1, 1),
            (oof_s * slot_valley).reshape(-1, 1),
            (oof_l * slot_peak).reshape(-1, 1),
            (oof_s * oof_l).reshape(-1, 1),
            ((oof_s + oof_w) * slot_valley).reshape(-1, 1),
            (oof_l * period_cos).reshape(-1, 1),
        ])
        return np.concatenate(cols, axis=1)

    X_dry = build_X(data_dry, oof_solar, oof_hydro, oof_wind, oof_load)
    X_wet = build_X(data_wet, oof_solar, oof_hydro, oof_wind, oof_load)
    feat_names = (['oof_solar', 'oof_hydro', 'oof_wind', 'oof_load'] + safe_names +
                  ['int_solar_period', 'int_solar_valley', 'int_load_peak',
                   'int_solar_load', 'int_renew_valley', 'int_load_period'])
    print(f"  X: {X_dry.shape[1]} dims (4 OOF + {len(safe_indices)} safe + 6 interactions)")

    # ── 目标构建 ──
    y_resid = data_dry['y_price_resid']  # price[t+96] - price[t]
    price_t = data_dry['price']
    price_t96 = price_t + y_resid         # price[t+96]
    lag96 = data_dry['price_lag96']
    lag672 = np.roll(price_t, 672); lag672[:672] = np.nan
    anchor = (lag96 + lag672) / 2.0       # 昨周均值基线

    period = data_dry['period']
    idx_dry_train = data_dry['idx_train']; idx_dry_val = data_dry['idx_val']
    idx_dry_test  = data_dry['idx_test']
    idx_wet_train = data_wet['idx_train']; idx_wet_val = data_wet['idx_val']
    idx_wet_test  = data_wet['idx_test']

    # 过滤 NaN (anchor 前 672 行为 NaN)
    valid_dry = ~np.isnan(X_dry).any(axis=1) & ~np.isnan(anchor)
    valid_wet = ~np.isnan(X_wet).any(axis=1) & ~np.isnan(anchor)
    idx_dry_train = np.intersect1d(idx_dry_train, np.where(valid_dry)[0])
    idx_dry_val   = np.intersect1d(idx_dry_val,   np.where(valid_dry)[0])
    idx_dry_test  = np.intersect1d(idx_dry_test,  np.where(valid_dry)[0])
    idx_wet_train = np.intersect1d(idx_wet_train, np.where(valid_wet)[0])
    idx_wet_val   = np.intersect1d(idx_wet_val,   np.where(valid_wet)[0])
    idx_wet_test  = np.intersect1d(idx_wet_test,  np.where(valid_wet)[0])

    print(f"  枯水 Train/Val/Test: {len(idx_dry_train)}/{len(idx_dry_val)}/{len(idx_dry_test)}")
    print(f"  丰水 Train/Val/Test: {len(idx_wet_train)}/{len(idx_wet_val)}/{len(idx_wet_test)}")

    return {
        'X_dry': X_dry, 'X_wet': X_wet,
        'y_dry': price_t96, 'y_wet': price_t96,
        'anchor': anchor, 'lag96': lag96,
        'idx_dry_train': idx_dry_train, 'idx_dry_val': idx_dry_val,
        'idx_dry_test': idx_dry_test,
        'idx_wet_train': idx_wet_train, 'idx_wet_val': idx_wet_val,
        'idx_wet_test': idx_wet_test,
        'period': period, 'feat_names': feat_names, 'price_t': price_t,
        'safe_indices': safe_indices, 'safe_names': safe_names,
    }


def _load_oof(var_name: str, grid_lag: int = 0) -> np.ndarray:
    """加载 Stage1 OOF 预测。

    优先使用 pipeline/output/{var}_oof{lag_suffix}.npz（Phase 5 本地训练产出），
    回退到 EM_Pre3 Stage1/prediction/output/{var}_oof.npz（参考 OOF）。
    """
    lag_suffix = f"_lag{grid_lag}" if grid_lag > 0 else ""
    # 本地 OOF
    local_path = OOF_DIR / f"{var_name}_oof{lag_suffix}.npz"
    if local_path.exists():
        data = np.load(local_path, allow_pickle=True)
        key = f"oof_{var_name}"
        if key in data:
            print(f"    {var_name}: from pipeline/output/{var_name}_oof{lag_suffix}.npz")
            return data[key]

    # 回退：EM_Pre3 参考 OOF
    ref_path = (
        PROJECT_ROOT.parent / "EM_Pre3" / "Stage1" / "prediction" / "output"
        / f"{var_name}_oof.npz"
    )
    if ref_path.exists():
        data = np.load(ref_path, allow_pickle=True)
        key = f"oof_{var_name}"
        if key in data:
            print(f"    {var_name}: from EM_Pre3 (reference OOF)")
            return data[key]

    raise FileNotFoundError(
        f"Stage1 OOF not found for {var_name}. Tried:\n"
        f"  {local_path}\n  {ref_path}\n"
        f"Run pipeline/train_stage1.py first."
    )


# ====================================================================
# 2. 时段掩码 + 软融合权重
# ====================================================================

def build_period_masks(period):
    """构建三时段训练/推理掩码。

    训练范围比推理宽 4 个 period（±1h），确保边界样本充足。
    valley_train: 32-71  → valley_infer: 36-67（午谷 09:00-16:45）
    peak_train:   64-91  → peak_infer:   68-87（晚峰 17:00-21:45）
    base_train:   0-31 or 92-95 → base_infer: 0-35 or 88-95（基荷）
    """
    p = period
    return {
        'valley_train': (p >= 32) & (p <= 71),
        'valley_infer': (p >= 36) & (p <= 67),
        'peak_train':   (p >= 64) & (p <= 91),
        'peak_infer':   (p >= 68) & (p <= 87),
        'base_train':   (p <= 31) | (p >= 92),
        'base_infer':   (p <= 35) | (p >= 88),
    }


def blend_weights(period):
    """余弦软融合权重：在 valley/peak/base 交界处平滑过渡。

    过渡区间宽度 = 4 个 period (1h)，使用线性 ramp:
      period 0-31:  base=1
      period 32-35: base→valley 过渡 (32→0, 35→1)
      period 36-63: valley=1
      period 64-67: valley→peak 过渡
      period 68-83: peak=1
      period 84-87: peak→base 过渡
      period 88-95: base=1
    """
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
# 3. 单季训练（dry: RF残差 / wet: XGB绝对值）
# ====================================================================

def train_one_season(X, y_abs, anchor, price_lag96,
                     idx_train, idx_val, idx_test, period, season_name):
    """训练一个季节的 3 个时段模型。

    枯水 (dry):  RandomForest 预测 price[t+96] - anchor
    丰水 (wet):  XGBoost 直接预测 price[t+96]

    Returns
    -------
    dict : {price_pred, resid_pred, segments, masks, train, val, test, baseline}
    """
    masks = build_period_masks(period)
    idx_tv = np.concatenate([idx_train, idx_val])
    n_tv, n_test, n_total = len(idx_tv), len(idx_test), X.shape[0]
    is_dry = (season_name == 'dry')

    if is_dry:
        y_target = y_abs - anchor  # 残差
        print(f"  枯水策略: RF预测残差 → price = anchor + resid")
    else:
        y_target = y_abs            # 绝对值
        print(f"  丰水策略: XGBoost 直接预测 price[t+96]")

    X_tv, y_tv = X[idx_tv], y_target[idx_tv]

    # ── 模型参数 ──
    if not is_dry:
        # 丰水期：加权 XGBoost（抑制极端值影响）
        abs_r = np.abs(y_tv - np.median(y_tv))
        w_tv = 1.0 / (1.0 + abs_r / (np.percentile(abs_r, 80) + 1e-6))
        xgb_params = {
            'valley': dict(max_depth=3, learning_rate=0.03, n_estimators=300,
                           reg_lambda=6.0, reg_alpha=2.0, min_child_weight=10,
                           subsample=0.80, colsample_bytree=0.75),
            'peak':   dict(max_depth=3, learning_rate=0.03, n_estimators=300,
                           reg_lambda=6.0, reg_alpha=2.0, min_child_weight=10,
                           subsample=0.80, colsample_bytree=0.75),
            'base':   dict(max_depth=4, learning_rate=0.03, n_estimators=300,
                           reg_lambda=5.0, reg_alpha=1.5, min_child_weight=8,
                           subsample=0.80, colsample_bytree=0.80),
        }
        n_folds = 3
    else:
        n_folds = 5

    # ── 逐时段训练 ──
    segments = {}
    for seg in ['valley', 'peak', 'base']:
        print(f"\n  --- {seg} ---")
        train_mask = masks[f'{seg}_train']
        tv_sel = np.where(train_mask[idx_tv])[0]  # 在 idx_tv 中的位置
        n_sel = len(tv_sel)
        print(f"    样本: {n_sel}/{n_tv}")

        if n_sel < 50:
            # 样本不足，跳过
            segments[seg] = {'model': None, 'oof_tv': np.zeros(n_tv),
                             'oof_test': np.zeros(n_test)}
            continue

        X_sel, y_sel = X_tv[tv_sel], y_tv[tv_sel]
        tscv = TimeSeriesSplit(n_splits=n_folds)

        if is_dry:
            # ── Random Forest（5-fold OOF） ──
            oof_sel = np.full(n_sel, np.nan)
            for fold, (fi_tr, fi_vl) in enumerate(tscv.split(X_sel)):
                m = RandomForestRegressor(
                    n_estimators=100, max_depth=5, min_samples_leaf=10,
                    random_state=RANDOM_SEED, n_jobs=-1,
                )
                m.fit(X_sel[fi_tr], y_sel[fi_tr])
                oof_sel[fi_vl] = m.predict(X_sel[fi_vl])

            # 全量模型（用于推理）
            fm = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
            fm.fit(X_sel, y_sel)
        else:
            # ── XGBoost（加权，3-fold OOF + early stopping） ──
            abs_r_sel = np.abs(y_sel)
            w_sel = 1.0 / (1.0 + abs_r_sel / (np.percentile(abs_r_sel, 80) + 1e-6))

            oof_sel = np.full(n_sel, np.nan)
            for fold, (fi_tr, fi_vl) in enumerate(tscv.split(X_sel)):
                m = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1,
                                 **xgb_params[seg])
                m.fit(X_sel[fi_tr], y_sel[fi_tr], sample_weight=w_sel[fi_tr],
                      eval_set=[(X_sel[fi_vl], y_sel[fi_vl])], verbose=False)
                oof_sel[fi_vl] = m.predict(X_sel[fi_vl])

            # 全量模型：85% early stopping
            sp = int(n_sel * 0.85)
            fm = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1,
                              early_stopping_rounds=30, **xgb_params[seg])
            fm.fit(X_sel[:sp], y_sel[:sp], sample_weight=w_sel[:sp],
                   eval_set=[(X_sel[sp:], y_sel[sp:])], verbose=False)

        # 全量预测（用于 OOF 拼接）
        oof_tv_full = fm.predict(X_tv)
        oof_test_full = fm.predict(X[idx_test])
        oof_tv_final = oof_tv_full.copy()
        # 用 fold OOF 替换对应位置
        for i, loc in enumerate(tv_sel):
            if not np.isnan(oof_sel[i]):
                oof_tv_final[loc] = oof_sel[i]

        segments[seg] = {'model': fm, 'oof_tv': oof_tv_final, 'oof_test': oof_test_full}
        print(f"    OOF覆盖: {np.sum(~np.isnan(oof_tv_final))}/{n_tv}")

    # ── 软融合 ──
    weights = blend_weights(period)
    oof_full = np.full(n_total, np.nan)
    for i, gi in enumerate(idx_tv):
        w = weights[gi]
        oof_full[gi] = (w[0] * segments['valley']['oof_tv'][i] +
                        w[1] * segments['peak']['oof_tv'][i] +
                        w[2] * segments['base']['oof_tv'][i])
    for i, gi in enumerate(idx_test):
        w = weights[gi]
        oof_full[gi] = (w[0] * segments['valley']['oof_test'][i] +
                        w[1] * segments['peak']['oof_test'][i] +
                        w[2] * segments['base']['oof_test'][i])
    oof_full[np.isnan(oof_full)] = 0.0

    # 最终价格
    if is_dry:
        price_pred = anchor + oof_full   # 残差 + anchor = 价格
        resid_pred = oof_full
    else:
        price_pred = oof_full             # 直接 = 价格
        resid_pred = oof_full - anchor

    # ── 评估 ──
    def evaluate(idx, label):
        valid = np.isin(idx, np.where(~np.isnan(price_pred))[0])
        ei = idx[valid]
        if len(ei) == 0:
            return {'R2': np.nan, 'MAE': np.nan, 'n': 0, 'seg_mae': {}}
        yt = y_abs[ei]; yp = price_pred[ei]
        r2 = r2_score(yt, yp); mae = mean_absolute_error(yt, yp)
        seg_mae = {}
        for sg in ['valley', 'peak', 'base']:
            sm = masks[f'{sg}_infer'][ei]
            seg_mae[sg] = mean_absolute_error(yt[sm], yp[sm]) if sm.sum() > 0 else np.nan
        print(f"  {label}: R2={r2:.4f}, MAE_price={mae:.2f}, n={len(ei)}")
        print(f"    午谷={seg_mae['valley']:.2f}  晚峰={seg_mae['peak']:.2f}  基荷={seg_mae['base']:.2f}")
        return {'R2': r2, 'MAE': mae, 'n': len(ei), 'seg_mae': seg_mae}

    tr_ev = evaluate(idx_train, 'Train')
    va_ev = evaluate(idx_val,   'Val')
    te_ev = evaluate(idx_test,  'Test')

    # 基线
    bl_v = ~np.isnan(y_abs[idx_test]) & ~np.isnan(price_pred[idx_test])
    bl_t = y_abs[idx_test][bl_v]
    bl_mae_anchor = mean_absolute_error(bl_t, anchor[idx_test][bl_v])
    bl_mae_lag96  = mean_absolute_error(bl_t, price_lag96[idx_test][bl_v])
    print(f"  基线 anchor(昨周均值): MAE={bl_mae_anchor:.2f}")
    print(f"  基线 lag96:            MAE={bl_mae_lag96:.2f}")

    return {
        'price_pred': price_pred, 'resid_pred': resid_pred,
        'segments': segments, 'masks': masks,
        'train': tr_ev, 'val': va_ev, 'test': te_ev,
        'baseline': {'MAE_anchor': bl_mae_anchor, 'MAE_lag96': bl_mae_lag96},
    }


# ====================================================================
# 4. 主流程
# ====================================================================

def train_and_save(skip_versioning: bool = False, grid_lag: int = 0):
    """完整 Stage2 训练 + 保存 + model versioning。

    Parameters
    ----------
    skip_versioning : bool
        True = 跳过 model_versions 表写入（测试时使用）
    grid_lag : int
        If > 0, use lagged feature/OOF files and save with _lag{N} suffix.
    """
    lag_suffix = f"_lag{grid_lag}" if grid_lag > 0 else ""
    print("=" * 60)
    print(f"  Stage 2 — 枯水 RF 残差 + 丰水 XGB 绝对值{lag_suffix}")
    print("=" * 60)

    d = load_all_data(grid_lag=grid_lag)

    # ── 逐季训练 ──
    results = {}
    for sn, Xk, yk, ak, lk, trk, vlk, tek in [
        ('dry', 'X_dry', 'y_dry', 'anchor', 'lag96',
         'idx_dry_train', 'idx_dry_val', 'idx_dry_test'),
        ('wet', 'X_wet', 'y_wet', 'anchor', 'lag96',
         'idx_wet_train', 'idx_wet_val', 'idx_wet_test'),
    ]:
        print(f"\n{'#' * 60}")
        print(f"# {sn.upper()} SEASON")
        print(f"{'#' * 60}")
        results[sn] = train_one_season(
            d[Xk], d[yk], d[ak], d[lk], d[trk], d[vlk], d[tek], d['period'], sn,
        )

    # ── 合并枯/丰水 OOF → 全量价格预测 ──
    print(f"\n{'=' * 60}")
    print("  保存结果")
    print(f"{'=' * 60}")

    # Load masks from the same feature file used for training
    _feat_dry_path = FEAT_DIR / f"features_15min_dry{lag_suffix}.npz"
    dmask = np.load(_feat_dry_path, allow_pickle=True)['dry_mask'].astype(bool)
    wmask = np.load(_feat_dry_path, allow_pickle=True)['wet_mask'].astype(bool)
    n_total = len(dmask)

    price_pred_all = np.full(n_total, np.nan)
    resid_pred_all = np.full(n_total, np.nan)
    price_pred_all[dmask] = results['dry']['price_pred'][dmask]
    price_pred_all[wmask] = results['wet']['price_pred'][wmask]
    resid_pred_all[dmask] = results['dry']['resid_pred'][dmask]
    resid_pred_all[wmask] = results['wet']['resid_pred'][wmask]

    np.savez(
        FEAT_DIR / f'price_oof{lag_suffix}.npz',
        oof_price=price_pred_all, resid_pred=resid_pred_all,
        price=d['price_t'], price_lag96=d['lag96'],
        dry_mask=dmask, wet_mask=wmask,
    )
    print(f"  [OK] price_oof{lag_suffix}.npz")

    # ── 保存 6 个 Stage2 模型 ──
    models_saved = []
    for sn in ['dry', 'wet']:
        for seg in ['valley', 'peak', 'base']:
            m = results[sn]['segments'][seg]['model']
            if m is not None:
                path = MODEL_DIR / f'price_{seg}_{sn}{lag_suffix}.pkl'
                with open(path, 'wb') as f:
                    pickle.dump({
                        'model': m,
                        'season': sn, 'segment': seg,
                        'grid_lag': grid_lag,
                        'feat_names': d['feat_names'],
                        'safe_indices': d['safe_indices'],
                        'safe_names': d['safe_names'],
                    }, f)
                models_saved.append(str(path))
                print(f"  [OK] {path.name}")

    # ── 生成评估报告 ──
    _write_report(results, FEAT_DIR)
    print(f"  报告: {REPORT_DIR / 'price_stage2_report.md'}")

    # ── Model versioning ──
    if not skip_versioning:
        _write_model_versions(results)
        print("  [OK] model_versions updated")

    # ── 输出关键指标 ──
    print(f"\n{'=' * 60}")
    print("  Stage 2 训练完成")
    print(f"{'=' * 60}")
    for sn, label in [('dry', '枯水期'), ('wet', '丰水期')]:
        r = results[sn]
        print(f"\n  {label}:")
        print(f"    Test R2  = {r['test']['R2']:.4f}")
        print(f"    Test MAE = {r['test']['MAE']:.2f} 元/MWh")
        print(f"    基线 anchor = {r['baseline']['MAE_anchor']:.2f}")
        print(f"    基线 lag96  = {r['baseline']['MAE_lag96']:.2f}")

    return results


# ====================================================================
# 辅助：报告 + Model Versioning
# ====================================================================

def _write_report(results: dict, output_dir: Path):
    """生成 Markdown 评估报告（与 EM_Pre3 格式一致）。"""
    lines = [
        "# Stage2 电价预测 — v13 混合策略\n",
        "> 枯水: RF 预测 anchor 残差 | 丰水: XGBoost 直接预测绝对值\n",
        "---\n",
    ]
    for sn, st in [('dry', '枯水期'), ('wet', '丰水期')]:
        r = results[sn]
        lines.append(f"## {st}\n")
        lines.append("| 指标 | Train | Val | Test | anchor基线 | lag96基线 |")
        lines.append("|---|---|---|---|---|---|")
        tr = r['train']; va = r['val']; te = r['test']; bl = r['baseline']
        lines.append(f"| R² | {tr['R2']:.4f} | {va['R2']:.4f} | {te['R2']:.4f} | — | — |")
        lines.append(
            f"| MAE_price | {tr['MAE']:.2f} | {va['MAE']:.2f} | {te['MAE']:.2f} "
            f"| {bl['MAE_anchor']:.2f} | {bl['MAE_lag96']:.2f} |"
        )
        lines.append(f"| n | {tr['n']} | {va['n']} | {te['n']} | — | — |\n")
    lines.append(f"\n> 训练时间: {datetime.now().isoformat()}\n")

    with open(REPORT_DIR / 'price_stage2_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _write_model_versions(results: dict):
    """将训练好的模型指标写入 model_versions 表。

    每个模型一行记录，包含 R²/MAE 等指标。
    新训练完成后将旧模型设为 archived。
    """
    from sqlalchemy import create_engine, text

    url = settings.database_url_sync
    engine = create_engine(url, echo=False)
    version_tag = datetime.now().strftime('v%Y%m%d_%H%M')

    with engine.connect() as conn:
        with conn.begin():
            for season in ['dry', 'wet']:
                r = results[season]
                for seg in ['valley', 'peak', 'base']:
                    m = r['segments'][seg]['model']
                    if m is None:
                        continue

                    version_name = f"stage2_{season}_{seg}_{version_tag}"
                    model_type = f"stage2_price_{seg}"
                    metrics = {
                        'R2_test': round(r['test']['R2'], 4),
                        'MAE_test': round(r['test']['MAE'], 2),
                        'seg_mae_test': {k: round(v, 2) if not np.isnan(v) else None
                                        for k, v in r['test'].get('seg_mae', {}).items()},
                        'baseline_MAE_anchor': round(r['baseline']['MAE_anchor'], 2),
                        'baseline_MAE_lag96': round(r['baseline']['MAE_lag96'], 2),
                        'n_test': r['test']['n'],
                    }

                    conn.execute(
                        text("""
                            INSERT INTO model_versions (version_name, model_type, file_path, metrics, status)
                            VALUES (:name, :type, :path, CAST(:metrics AS jsonb), 'active')
                        """),
                        {
                            'name': version_name,
                            'type': model_type,
                            'path': f"models/price_{seg}_{season}.pkl",
                            'metrics': json.dumps(metrics, ensure_ascii=False),
                        },
                    )

        conn.commit()

    engine.dispose()
    print(f"  [OK] {version_tag}: 6 models registered in model_versions")


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage2 电价预测训练 (枯水 RF 残差 + 丰水 XGB 绝对值)"
    )
    parser.add_argument('--no-versioning', action='store_true',
                        help='跳过 model_versions 表写入')
    parser.add_argument('--grid-lag', type=int, default=0,
                        help='Train with lagged features (192 = gap-fill for t-2)')
    args = parser.parse_args()

    train_and_save(skip_versioning=args.no_versioning, grid_lag=args.grid_lag)


if __name__ == "__main__":
    main()
