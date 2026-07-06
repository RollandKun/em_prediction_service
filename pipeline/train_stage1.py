# -*- coding: utf-8 -*-
"""
pipeline/train_stage1.py — Train and save Stage1 models (solar/hydro/wind/load)
==============================================================================
Produces: em_prediction_service/models/stage1_{var}_{season}.pkl (8 files)

Usage: python -m pipeline.train_stage1

Strategy (v12):
  - Solar: direct absolute value prediction
  - Wind:  direct absolute value prediction
  - Hydro: lag_96 residual (hydro[t+96] - hydro_lag_96) + amplified precipitation
  - Load:  lag_96 residual (load[t+96] - load_lag_96)
"""
import sys
import io
import warnings
from pathlib import Path
import pickle

import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

# ── Paths ──
FEAT_DIR = PROJECT_ROOT / "pipeline" / "output"
MODEL_DIR = Path(settings.model_dir)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# ── Feature selections (identical to EM_Pre3 training scripts) ──

SOLAR_FEATURES = [
    'D_rad_avg_pp', 'D_rad_eff_pp', 'D_temp_avg_pp', 'D_cloud_avg_pp',
    'D_cdd_avg_pp', 'D_rain_h_avg_pp', 'D_rain_24h_avg_pp',
    'F_period_sin_pp', 'F_period_cos_pp', 'F_month_sin_pp', 'F_month_cos_pp',
    'F_is_holiday_pp', 'F_is_weekend_pp', 'F_is_dry_season_pp', 'F_is_flood_pp',
    'F_slot_午谷_pp', 'F_slot_上午_pp', 'F_slot_深夜_pp',
    'B_solar_lag_96', 'B_solar_lag_672', 'K_solar_lag_192',
    'L_morning_mean_solar', 'L_morning_last_solar',
    'L_morning_ramp_solar', 'L_morning_std_solar',
    'N_d2_solar_peak', 'N_d2_solar_ramp_up', 'N_d2_solar_day_std',
    'N_d2_solar_duration', 'N_d2_solar_day_mean', 'N_d2_solar_day_total',
    'K_solar_ma_7d', 'J_rad_pp_x_period_cos',
    'D_is_daytime', 'D_is_daytime_pp', 'D_ghi_ramp_15min',
]

HYDRO_FEATURES = [
    'B_hydro_lag_96', 'B_hydro_lag_672', 'B_hydro_ma_7d',
    'D_temp_avg_pp', 'D_rain_24h_avg_pp', 'D_rain_h_avg_pp', 'D_cdd_avg_pp',
    'K_rain_72h_acc_pp', 'D_precip_15d_sum',
    # Basin-level precipitation (amplified hydro signal)
    'D_precip_yaan_pp', 'D_precip_pzh_pp', 'D_precip_ls_pp',
    'D_precip_avg_sq_pp', 'D_precip_avg_sqrt_pp',
    'L_morning_mean_hydro', 'L_morning_last_hydro',
    'L_morning_ramp_hydro', 'L_morning_std_hydro',
    'N_d2_hydro_range', 'N_d2_hydro_day_mean',
    'N_d2_hydro_night_mean', 'N_d2_hydro_daytime_mean',
    'F_period_sin_pp', 'F_period_cos_pp', 'F_month_sin_pp', 'F_month_cos_pp',
    'F_is_dry_season_pp', 'F_is_flood_pp',
    'F_slot_晚峰_pp', 'F_slot_午谷_pp',
    'J_rain72h_x_flood', 'J_rain24h_x_hydro',
    'J_precip_yaan_x_flood', 'J_precip_15d_x_hydro',
    'O_hydro_lag_192', 'O_hydro_lag_193', 'O_hydro_lag_196',
    'O_hydro_lag_288', 'O_hydro_d2_ramp_1h',
    'P_hydro_d2_w2h_mean', 'P_hydro_d2_w2h_std', 'P_hydro_d2_w2h_trend',
    'P_hydro_d2_w2h_range', 'P_hydro_d2_w2h_max_step', 'P_hydro_d2_w2h_accel',
]

WIND_FEATURES = [
    'B_wind_lag_96', 'B_wind_lag_672', 'K_wind_ma_7d',
    'D_wind_cubic_pp', 'D_wind_cubic_now',
    'D_cloud_avg_pp', 'D_temp_avg_pp',
    'D_is_generating_pp', 'D_wind_turbulence_1h',
    'L_morning_mean_wind', 'L_morning_last_wind',
    'L_morning_ramp_wind', 'L_morning_std_wind',
    'N_d2_wind_range', 'N_d2_wind_day_mean', 'N_d2_wind_night_mean',
    'F_period_cos_pp', 'F_period_sin_pp', 'F_month_sin_pp', 'F_month_cos_pp',
    'K_temp_diurnal_range',
    'O_wind_lag_192', 'O_wind_lag_193', 'O_wind_lag_196',
    'O_wind_d2_ramp_1h', 'O_wind_d2_ramp_2h',
    'P_wind_d2_w2h_mean', 'P_wind_d2_w2h_std', 'P_wind_d2_w2h_trend',
    'P_wind_d2_w2h_range', 'P_wind_d2_w2h_max_step', 'P_wind_d2_w2h_accel',
]

LOAD_FEATURES = [
    'B_load_lag_96', 'B_load_lag_672', 'B_hydro_ma_7d',
    'D_temp_avg_pp', 'D_cdd_avg_pp',
    'D_apparent_temp_pp', 'D_apparent_temp_now',
    'L_morning_mean_load', 'L_morning_last_load',
    'L_morning_ramp_load', 'L_morning_std_load',
    'N_d2_load_morning_peak', 'N_d2_load_evening_peak', 'N_d2_load_valley',
    'N_d2_load_peak_ratio', 'N_d2_load_day_mean',
    'M_days_from_sf', 'M_is_sf_window',
    'F_period_sin_pp', 'F_period_cos_pp', 'F_month_sin_pp', 'F_month_cos_pp',
    'F_is_holiday_pp', 'F_is_weekend_pp', 'F_is_dry_season_pp',
    'O_load_lag_192', 'O_load_lag_193', 'O_load_lag_196', 'O_load_lag_200',
    'O_load_d2_ramp_1h', 'O_load_d2_ramp_2h',
    'P_load_d2_w2h_mean', 'P_load_d2_w2h_std', 'P_load_d2_w2h_trend',
    'P_load_d2_w2h_range', 'P_load_d2_w2h_max_step', 'P_load_d2_w2h_accel',
]

# ── XGBoost parameters (matching original training scripts) ──
PARAMS_DRY = dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                  subsample=0.85, colsample_bytree=0.80,
                  reg_lambda=2.0, reg_alpha=0.5, min_child_weight=3,
                  random_state=RANDOM_SEED, n_jobs=-1)
PARAMS_WET = dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                  subsample=0.80, colsample_bytree=0.75,
                  reg_lambda=4.0, reg_alpha=1.0, min_child_weight=5,
                  random_state=RANDOM_SEED, n_jobs=-1)
HYDRO_PARAMS_DRY = dict(n_estimators=300, max_depth=5, learning_rate=0.03,
                        subsample=0.80, colsample_bytree=0.75,
                        reg_lambda=5.0, reg_alpha=2.0, min_child_weight=8,
                        random_state=RANDOM_SEED, n_jobs=-1)
HYDRO_PARAMS_WET = dict(n_estimators=200, max_depth=3, learning_rate=0.03,
                        subsample=0.75, colsample_bytree=0.70,
                        reg_lambda=8.0, reg_alpha=3.0, min_child_weight=12,
                        random_state=RANDOM_SEED, n_jobs=-1)


def _generate_charts(chart_data, data_dry, data_wet):
    """Generate OOF scatter, residual hist, feature importance charts + report."""
    CHART_DIR = FEAT_DIR / "charts"
    REPORT_DIR = FEAT_DIR / "reports"
    CHART_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    VAR_LABELS = {'solar': '光伏 (MW)', 'hydro': '水电 (MW)',
                  'wind': '风电 (MW)', 'load': '负荷 (MW)'}
    SEASON_LABELS = {'dry': '枯水期', 'wet': '丰水期'}
    SEASON_COLORS = {'dry': '#2166AC', 'wet': '#B2182B'}

    # Set up matplotlib Chinese font
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    from datetime import datetime
    report_lines = [
        "# Stage1 训练报告",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## 指标汇总\n",
        "| 变量 | 季节 | 策略 | 固定Test R² | 固定Test MAE (MW) | RollingTest R² | RollingTest MAE (MW) |",
        "|------|------|------|-------------|-------------------|----------------|----------------------|",
    ]

    for var_name in ['solar', 'hydro', 'wind', 'load']:
        print(f"\n  Charting {var_name}...")
        var_label = VAR_LABELS[var_name]

        # ── Collect data from both seasons ──
        all_oof, all_actual, all_colors = [], [], []
        all_resid, all_resid_colors = [], []
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{var_label} — Stage1 OOF 评估', fontsize=14, fontweight='bold')

        for season_idx, season in enumerate(['dry', 'wet']):
            d = chart_data[var_name][season]
            color = SEASON_COLORS[season]
            season_label = SEASON_LABELS[season]
            ax_scatter = axes[season_idx, 0]
            ax_ts = axes[season_idx, 1]
            ax_hist = axes[season_idx, 2]

            # OOF valid points
            mask = d['oof_mask']
            oof_v = d['oof'][mask]
            act_v = d['actual'][mask]

            if len(oof_v) == 0:
                for ax in [ax_scatter, ax_ts, ax_hist]:
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                continue

            # 1. OOF vs Actual scatter
            ax_scatter.scatter(act_v, oof_v, c=color, alpha=0.3, s=2, edgecolors='none')
            lims = [min(act_v.min(), oof_v.min()), max(act_v.max(), oof_v.max())]
            ax_scatter.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
            ax_scatter.set_xlabel('实际值 (MW)')
            ax_scatter.set_ylabel('OOF 预测 (MW)')
            ax_scatter.set_title(f'{season_label} | OOF vs Actual')
            # R² annotation
            ss_res = np.sum((act_v - oof_v) ** 2)
            ss_tot = np.sum((act_v - np.mean(act_v)) ** 2)
            oof_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            ax_scatter.text(0.05, 0.95, f'OOF R²={oof_r2:.4f}\nn={len(oof_v):,}',
                            transform=ax_scatter.transAxes, va='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 2. Test set time series (last 672 points ≈ 7 days)
            te = d['test_idx']
            if len(te) > 0 and len(d['test_pred']) > 0:
                n_show = min(672, len(te))
                te_show = te[-n_show:]
                ax_ts.plot(d['actual'][te_show], 'k-', lw=0.8, alpha=0.7, label='实际')
                ax_ts.plot(d['test_pred'][-n_show:], color=color, lw=0.8, alpha=0.8, label='预测')
                ax_ts.set_xlabel('时序 (15min)')
                ax_ts.set_ylabel('MW')
                ax_ts.set_title(f'{season_label} | Test 时序 (末{n_show//96}天)')
                ax_ts.legend(fontsize=7)
            else:
                ax_ts.text(0.5, 0.5, '无测试集', ha='center', va='center', transform=ax_ts.transAxes)

            # 3. Residual distribution
            if len(d['test_pred']) > 0 and len(te) > 0:
                resid = d['actual'][te] - d['test_pred']
                ax_hist.hist(resid, bins=60, color=color, alpha=0.7, edgecolor='white', lw=0.3)
                ax_hist.axvline(0, color='k', ls='--', lw=0.8)
                ax_hist.axvline(np.mean(resid), color='red', ls='-', lw=1.0, label=f'均值={np.mean(resid):.1f}')
                ax_hist.set_xlabel('残差 (MW)')
                ax_hist.set_ylabel('频次')
                ax_hist.set_title(f'{season_label} | Test 残差分布 (σ={np.std(resid):.1f})')
                ax_hist.legend(fontsize=7)
            else:
                ax_hist.text(0.5, 0.5, '无测试集', ha='center', va='center', transform=ax_hist.transAxes)

            # Accumulate for combined charts
            all_oof.append(oof_v)
            all_actual.append(act_v)
            all_colors.append(color)
            all_resid.extend((d['actual'][te] - d['test_pred']).tolist() if len(te) > 0 and len(d['test_pred']) > 0 else [])
            all_resid_colors.extend([color] * len(te) if len(te) > 0 and len(d['test_pred']) > 0 else [])

            # Report row
            r2_txt = f"{d['r2']:.4f}" if not np.isnan(d['r2']) else "N/A"
            mae_txt = f"{d['mae']:.1f}" if not np.isnan(d['mae']) else "N/A"
            roll_r2_txt = f"{d['rolling_r2']:.4f}" if not np.isnan(d['rolling_r2']) else "N/A"
            roll_mae_txt = f"{d['rolling_mae']:.1f}" if not np.isnan(d['rolling_mae']) else "N/A"
            report_lines.append(
                f"| {var_label} | {season_label} | {d['strategy']} | "
                f"{r2_txt} | {mae_txt} | {roll_r2_txt} | {roll_mae_txt} |"
            )

        plt.tight_layout()
        fp = CHART_DIR / f"stage1_{var_name}_evaluation.png"
        fig.savefig(fp, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    {fp.name}")

        # ── Feature importance (combined dry+wet) ──
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Average importance across seasons
        imp_dry = chart_data[var_name]['dry']['feat_imp']
        imp_wet = chart_data[var_name]['wet']['feat_imp']
        feat_names = chart_data[var_name]['dry']['feat_names']
        imp_avg = (imp_dry + imp_wet) / 2
        top_n = min(20, len(imp_avg))
        top_idx = np.argsort(imp_avg)[-top_n:]
        colors = ['#B2182B' if imp_wet[i] > imp_dry[i] else '#2166AC' for i in top_idx]
        ax2.barh(range(top_n), imp_avg[top_idx], color=colors, edgecolor='white', lw=0.5)
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
        ax2.set_xlabel('平均重要性')
        ax2.set_title(f'{var_label} — 特征重要性 Top {top_n} (蓝=枯水高, 红=丰水高)')
        ax2.invert_yaxis()
        plt.tight_layout()
        fp2 = CHART_DIR / f"stage1_{var_name}_importance.png"
        fig2.savefig(fp2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"    {fp2.name}")

    # ── Combined residual histogram ──
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    for var_idx, var_name in enumerate(['solar', 'hydro', 'wind', 'load']):
        ax = plt.subplot(2, 2, var_idx + 1)
        for season in ['dry', 'wet']:
            d = chart_data[var_name][season]
            te = d['test_idx']
            if len(te) > 0 and len(d['test_pred']) > 0:
                resid = d['actual'][te] - d['test_pred']
                ax.hist(resid, bins=50, alpha=0.5, color=SEASON_COLORS[season],
                        label=SEASON_LABELS[season], edgecolor='white', lw=0.2)
                ax.axvline(0, color='k', ls='--', lw=0.8)
        ax.set_title(VAR_LABELS[var_name])
        ax.set_xlabel('残差 (MW)')
        ax.legend(fontsize=7)
    fig3.suptitle('Stage1 Test 残差分布汇总', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fp3 = CHART_DIR / "stage1_residual_summary.png"
    fig3.savefig(fp3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"    {fp3.name}")

    # ── Save report ──
    report_lines.append(f"\n图表保存至: `{CHART_DIR}`")
    report_path = REPORT_DIR / "stage1_report.md"
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f"\n  Report: {report_path}")


def train_and_save(grid_lag=0):
    """Train all Stage1 models (dry+wet for each variable) and save to models/.

    Parameters
    ----------
    grid_lag : int — if > 0, load features with _lag{N} suffix and save models
               with _lag{N} prefix (gap-fill mode).
    """
    lag_suffix = f"_lag{grid_lag}" if grid_lag > 0 else ""
    print("=" * 70)
    print(f"  Stage1 Model Training — v11 Mixed Anchors{lag_suffix}")
    print("=" * 70)

    feat_dry_path = FEAT_DIR / f"features_15min_dry{lag_suffix}.npz"
    feat_wet_path = FEAT_DIR / f"features_15min_wet{lag_suffix}.npz"

    if not feat_dry_path.exists():
        print("ERROR: Feature files not found. Run pipeline/feature_engine.py first.")
        return

    data_dry = np.load(feat_dry_path, allow_pickle=True)
    data_wet = np.load(feat_wet_path, allow_pickle=True)
    all_names = data_dry['feat_names'].tolist()

    X_full_dry = data_dry['X']
    X_full_wet = data_wet['X']

    idx_dry_train = data_dry['idx_train']
    idx_dry_val   = data_dry['idx_val']
    idx_dry_test  = data_dry['idx_test']
    idx_dry_rolling_test = data_dry['idx_rolling_test'] if 'idx_rolling_test' in data_dry else np.array([], dtype=int)
    idx_wet_train = data_wet['idx_train']
    idx_wet_val   = data_wet['idx_val']
    idx_wet_test  = data_wet['idx_test']
    idx_wet_rolling_test = data_wet['idx_rolling_test'] if 'idx_rolling_test' in data_wet else np.array([], dtype=int)

    TARGET_MAP = {'solar': 'y_solar', 'hydro': 'y_hydro', 'wind': 'y_wind'}

    trained = []

    # Store per-variable results for charting
    chart_data = {}  # var_name → {season: {oof, actual, test_idx, feat_names, feat_imp, r2, mae}}

    # OOF containers for Stage2 (combined dry+wet, length = total rows)
    # Both npz files contain full 16209-row data with dry_mask/wet_mask filters
    n_total = len(data_dry['dry_mask'])
    oof_combined = {var: np.full(n_total, np.nan) for var in ['solar', 'hydro', 'wind', 'load']}

    for var_name, feat_list in [
        ('solar', SOLAR_FEATURES), ('hydro', HYDRO_FEATURES),
        ('wind',  WIND_FEATURES),  ('load',  LOAD_FEATURES),
    ]:
        print(f"\n{'─' * 60}")
        print(f"  {var_name.upper()}")
        print(f"{'─' * 60}")

        fidx = [all_names.index(n) for n in feat_list if n in all_names]
        missing = [n for n in feat_list if n not in all_names]
        if missing:
            print(f"  [WARN] Missing features: {missing}")
        print(f"  Features: {len(fidx)}/{len(feat_list)}")

        X_dry = X_full_dry[:, fidx]
        X_wet = X_full_wet[:, fidx]

        for season, X_season, idx_tr, idx_vl, idx_te, idx_roll, data_season in [
            ('dry', X_dry, idx_dry_train, idx_dry_val, idx_dry_test,
             idx_dry_rolling_test, data_dry),
            ('wet', X_wet, idx_wet_train, idx_wet_val, idx_wet_test,
             idx_wet_rolling_test, data_wet),
        ]:
            # For load, target is y_price_resid from features (we don't have y_load)
            # We need to extract load from features. Let's get B_load[t] from X_full.
            if var_name == 'load':
                # Load target is y_solar of data... no, we need actual load target.
                # The features npz doesn't have y_load directly.
                # We compute it: load[t+96] = sf(load[t], 96)
                # load[t] is feature B_load[t] in all_names
                load_t_idx = all_names.index('B_load[t]')
                load_t = X_full_dry[:, load_t_idx] if season == 'dry' else X_full_wet[:, load_t_idx]
                y_abs = np.roll(load_t, -96)
                y_abs[-96:] = np.nan
            else:
                y_abs = data_season[TARGET_MAP[var_name]]

            # ── Build residual target ──
            if var_name in ('solar', 'wind'):
                y_target = y_abs
                strategy = "direct absolute"
            elif var_name == 'hydro':
                lag_anchor = np.roll(y_abs, 96)
                lag_anchor[:96] = np.nan
                y_target = y_abs - lag_anchor
                strategy = "lag_96 residual"
            else:  # wind, load
                lag_anchor = np.roll(y_abs, 96)
                lag_anchor[:96] = np.nan
                y_target = y_abs - lag_anchor
                strategy = "lag_96 residual"

            valid = ~np.isnan(y_target)
            if var_name not in ('solar', 'wind'):
                valid = valid & ~np.isnan(lag_anchor)
            valid &= ~np.isnan(X_season).any(axis=1)

            tr = np.intersect1d(idx_tr, np.where(valid)[0])
            vl = np.intersect1d(idx_vl, np.where(valid)[0])
            te = np.intersect1d(idx_te, np.where(valid)[0])
            roll = np.intersect1d(idx_roll, np.where(valid)[0])
            idx_tv = np.concatenate([tr, vl])

            X_tv = X_season[idx_tv]
            y_tv = y_target[idx_tv]

            if var_name == 'hydro':
                params = HYDRO_PARAMS_DRY if season == 'dry' else HYDRO_PARAMS_WET
            else:
                params = PARAMS_DRY.copy() if season == 'dry' else PARAMS_WET.copy()

            print(f"  {var_name}/{season}: {strategy} → "
                  f"Train={len(tr)}, Val={len(vl)}, Test={len(te)}, "
                  f"RollingTest={len(roll)}, TV={len(idx_tv)}")

            # ── Final model (train on full train+val for inference) ──
            model = XGBRegressor(**params)
            model.fit(X_tv, y_tv, verbose=False)

            # ── OOF computation via TimeSeriesSplit (for Stage2 training) ──
            # OOF covers ALL valid indices. Uses 5-fold TSCV on train+val.
            # This is the same approach as EM_Pre3 Stage1 training scripts.
            oof_full = np.full(len(y_target), np.nan)  # OOF for all valid rows
            tscv = TimeSeriesSplit(n_splits=5)
            for fold, (fi_tr, fi_vl) in enumerate(tscv.split(idx_tv)):
                fold_tr = idx_tv[fi_tr]
                fold_vl = idx_tv[fi_vl]
                m_fold = XGBRegressor(**params)
                m_fold.fit(X_season[fold_tr], y_target[fold_tr], verbose=False)
                oof_full[fold_vl] = m_fold.predict(X_season[fold_vl])
            # Convert residual OOF back to absolute values
            oof_valid_mask = ~np.isnan(oof_full)  # before 0.0 fill
            if var_name in ('solar', 'wind'):
                oof_abs_full = oof_full
            else:
                oof_abs_full = oof_full + lag_anchor
            oof_abs_full[np.isnan(oof_abs_full)] = 0.0
            print(f"    OOF: {np.sum(oof_valid_mask):,}/{len(idx_tv):,} points covered")

            # Store in combined OOF array (both oof_abs_full and mask are full 16209-length)
            season_mask = data_season['dry_mask'].astype(bool) if season == 'dry' else data_season['wet_mask'].astype(bool)
            oof_combined[var_name][season_mask] = oof_abs_full[season_mask]

            # ── Test evaluation ──
            if len(te) > 0:
                y_pred_resid = model.predict(X_season[te])
                if var_name in ('solar', 'wind'):
                    y_pred_abs = y_pred_resid
                else:
                    y_pred_abs = y_pred_resid + lag_anchor[te]
                r2 = r2_score(y_abs[te], y_pred_abs)
                mae = mean_absolute_error(y_abs[te], y_pred_abs)
                print(f"    Test: R2={r2:.4f}, MAE={mae:.1f} MW")
                test_pred = y_pred_abs
            else:
                r2 = float('nan')
                mae = float('nan')
                test_pred = np.array([])

            if len(roll) > 0:
                y_pred_resid_roll = model.predict(X_season[roll])
                if var_name in ('solar', 'wind'):
                    y_pred_abs_roll = y_pred_resid_roll
                else:
                    y_pred_abs_roll = y_pred_resid_roll + lag_anchor[roll]
                rolling_r2 = r2_score(y_abs[roll], y_pred_abs_roll)
                rolling_mae = mean_absolute_error(y_abs[roll], y_pred_abs_roll)
                print(f"    RollingTest: R2={rolling_r2:.4f}, MAE={rolling_mae:.1f} MW")
            else:
                rolling_r2 = float('nan')
                rolling_mae = float('nan')

            # Store for charting
            if var_name not in chart_data:
                chart_data[var_name] = {}
            chart_data[var_name][season] = {
                'oof': oof_abs_full, 'oof_mask': oof_valid_mask,
                'actual': y_abs, 'test_idx': te, 'test_pred': test_pred,
                'feat_names': [n for n in feat_list if n in all_names],
                'feat_imp': model.feature_importances_,
                'r2': r2 if len(te) > 0 else float('nan'),
                'mae': mae if len(te) > 0 else float('nan'),
                'rolling_r2': rolling_r2,
                'rolling_mae': rolling_mae,
                'strategy': strategy,
            }

            model_path = MODEL_DIR / f"stage1_{var_name}_{season}{lag_suffix}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'var': var_name,
                    'season': season,
                    'strategy': strategy,
                    'horizon': 96,
                    'grid_lag': grid_lag,
                    'feat_names': [n for n in feat_list if n in all_names],
                    'feat_indices': fidx,
                }, f)
            trained.append(str(model_path))
            print(f"    Saved: {model_path.name}")

    # ── Save OOF for Stage2 ──
    print(f"\n{'─' * 60}")
    print(f"  Saving Stage1 OOF → pipeline/output/")
    for var_name in ['solar', 'hydro', 'wind', 'load']:
        oof_path = FEAT_DIR / f"{var_name}_oof{lag_suffix}.npz"
        # 格式与 EM_Pre3 Stage1/prediction/output/*_oof.npz 一致
        np.savez(oof_path, **{f"oof_{var_name}": oof_combined[var_name]})
        oof_n = np.sum(~np.isnan(oof_combined[var_name]))
        print(f"    {var_name}_oof{lag_suffix}.npz: {oof_n:,}/{n_total:,} points ({oof_n/n_total*100:.1f}%)")

    # ── Generate charts and report ──
    _generate_charts(chart_data, data_dry, data_wet)

    print(f"\n{'=' * 70}")
    print(f"  Complete: {len(trained)} models → {MODEL_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage1 model training")
    parser.add_argument("--grid-lag", type=int, default=0,
                        help="Train with lagged grid features (192 = gap-fill for t-2)")
    args = parser.parse_args()
    train_and_save(grid_lag=args.grid_lag)
