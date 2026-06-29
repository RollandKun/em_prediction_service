# -*- coding: utf-8 -*-
"""
pipeline/feature_engine.py — 特征工程核心（纯计算，无 I/O）
=========================================================
从 DB 加载的数据（DataFrame + numpy 数组）→ 177 维特征矩阵。

依赖关系：
  shared/weather_config.py  →  NODES, CLUSTERS, NODE_NAMES, VAR_PREFIXES, REGION_WEIGHTS
  pipeline/data_loader.py   →  load_from_db()
  pipeline/output.py        →  save_outputs(), verify()

用法：
  python -m pipeline.feature_engine              # 完整构建 + 保存 + 验证
  python -m pipeline.feature_engine --verify-only  # 仅对比现有 npz
"""
import sys
import io
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Shared weather config ──
from shared.weather_config import (
    NODE_NAMES, VAR_PREFIXES, ALL_WEATHER_KEYS, CLUSTERS, REGION_WEIGHTS
)

# ── Constants ──
H = 96          # 24h-ahead horizon
RANDOM_SEED = 42


# ====================================================================
# Utility functions
# ====================================================================

def sf(arr, s=H):
    """shift forward: arr[t+s] → position t (NaN封尾)."""
    r = np.roll(np.asarray(arr, dtype=float), -s)
    r[-s:] = np.nan
    return r


def nan0(x):
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)


def roll_mean(x, w):
    return pd.Series(x).rolling(w, min_periods=max(1, w // 2)).mean().values


def roll_std(x, w):
    return pd.Series(x).rolling(w, min_periods=max(1, w // 2)).std().values


def roll_max(x, w):
    """过去 w 步滚动最大值 (仅历史)."""
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).max().shift(1).bfill().values


def roll_sum(x, w):
    """滚动求和 (用于累积降水)."""
    return pd.Series(x).rolling(w, min_periods=1).sum().values


def _hourly_rolling_sum(arr, n_hours):
    """对 ffill 后的 15-min 数组做小时级滚动求和。

    天气数据为小时级，ffill 到 15 分钟。直接做 15 分钟滚动和会高估 4 倍。
    通过每隔 4 步取一次值重建小时级数据，避免重复计数。
    """
    n = len(arr)
    arr_h = arr[::4]                          # 每 4 步取 1，重建小时级
    hourly_sum = roll_sum(arr_h, n_hours)     # 小时级滚动和
    result = np.repeat(hourly_sum, 4)          # 广播回 15-min
    if len(result) < n:
        result = np.pad(result, (0, n - len(result)), constant_values=np.nan)
    return result[:n]


def safe_fillna(arr, val=0.0):
    """填充 NaN 并保持浮点类型."""
    result = np.asarray(arr, dtype=float).copy()
    result[np.isnan(result)] = val
    return result


# ====================================================================
# Main: build_features
# ====================================================================

def build_features(dt_arr, df, solar, wind, hydro, load, price,
                   bidspace, reserve, nonmarket, tieline, load_tie,
                   grid_lag=0):
    """Build complete feature matrix (v3 cluster averaging).

    Parameters
    ----------
    dt_arr : pd.DatetimeIndex
    df : pd.DataFrame — merged grid + weather, ffill'd to 15-min
    solar, wind, hydro, load, price : np.ndarray (n,)
    bidspace, reserve, nonmarket, tieline, load_tie : np.ndarray (n,)
    grid_lag : int — lag in periods to shift grid-dependent arrays.
              0 = use current grid (normal mode).
              192 = use grid from 2 days ago (gap-fill mode for t-2 constraint).

    Returns
    -------
    dict with X, feat_names, targets, splits, metadata, feature selectors.
    """
    n = len(df)

    lag_tag = f" (grid_lag={grid_lag})" if grid_lag > 0 else ""
    print("\n" + "=" * 60)
    print(f"  Building features (A-P groups, v3 cluster avg{lag_tag})")
    print("=" * 60)

    # ── Preserve originals for target computation ──
    solar_orig = solar.copy()
    wind_orig = wind.copy()
    hydro_orig = hydro.copy()
    load_orig = load.copy()
    price_orig = price.copy()

    # ── Shift grid arrays for feature construction (gap-fill mode) ──
    if grid_lag > 0:
        def _lag(arr, k):
            r = np.roll(arr, k); r[:k] = np.nan; return r
        solar  = _lag(solar,  grid_lag)
        wind   = _lag(wind,   grid_lag)
        hydro  = _lag(hydro,  grid_lag)
        load   = _lag(load,   grid_lag)
        price  = _lag(price,  grid_lag)
        bidspace = _lag(bidspace, grid_lag)
        nonmarket = _lag(nonmarket, grid_lag)
        print(f"  Grid arrays shifted by {grid_lag} periods (gap-fill mode)")

    # ── Time features ──
    period_arr = dt_arr.hour * 4 + dt_arr.minute // 15
    hour_arr = dt_arr.hour.values
    month_arr = dt_arr.month.values
    dow_arr = dt_arr.dayofweek.values
    is_holiday = (df['日期类型'] == '节假日').values.astype(float)
    is_weekend = (dow_arr >= 5).astype(float)
    is_dry_season = df.get(
        '枯水期',
        pd.Series((month_arr >= 1) & (month_arr <= 4), index=df.index)
    ).values.astype(float)
    is_flood = df.get(
        '汛期',
        pd.Series(month_arr >= 5, index=df.index)
    ).values.astype(float)
    is_main_flood = df.get(
        '主汛期',
        pd.Series(month_arr >= 6, index=df.index)
    ).values.astype(float)

    # ── Weather columns ──
    wcols = [c for c in df.columns if c in ALL_WEATHER_KEYS]
    wdata = {c: df[c].values.astype(float) for c in wcols}
    print(f"  Weather variables: {len(wcols)} columns")

    # ── Step 1: 集群内等权平均 → 7 个区域级变量 ──
    def _cluster_avg(prefix, cluster_nodes):
        """集群内等权平均。自动跳过缺失子节点。"""
        total = np.zeros(n)
        count = 0
        for node_name in cluster_nodes:
            col = f"{prefix}_{node_name}"
            if col in wdata:
                total += safe_fillna(wdata[col], 0.0)
                count += 1
        if count > 0:
            total /= count
        return total

    cd_temp = _cluster_avg('temp', CLUSTERS['chengdu'])
    cd_rh   = _cluster_avg('rh',   CLUSTERS['chengdu'])
    dz_temp = _cluster_avg('temp', CLUSTERS['dazhou'])
    dz_rh   = _cluster_avg('rh',   CLUSTERS['dazhou'])
    yb_temp = _cluster_avg('temp', CLUSTERS['yibin'])
    yb_rh   = _cluster_avg('rh',   CLUSTERS['yibin'])
    ls_temp    = _cluster_avg('temp',   CLUSTERS['liangshan'])
    ls_rad     = _cluster_avg('rad',    CLUSTERS['liangshan'])
    ls_cloud   = _cluster_avg('cloud',  CLUSTERS['liangshan'])
    ls_wind    = _cluster_avg('wind',   CLUSTERS['liangshan'])
    ls_precip  = _cluster_avg('precip', CLUSTERS['liangshan'])
    gz_rad     = _cluster_avg('rad',    CLUSTERS['ganzi'])
    gz_cloud   = _cluster_avg('cloud',  CLUSTERS['ganzi'])
    gz_wind    = _cluster_avg('wind',   CLUSTERS['ganzi'])
    ya_precip  = _cluster_avg('precip', CLUSTERS['yaan'])
    pzh_precip = _cluster_avg('precip', CLUSTERS['panzhihua'])

    # ── Step 2: 区域间加权融合 → 6 个最终天气变量 ──
    w = REGION_WEIGHTS
    temp_avg   = (w['temp']['chengdu']*cd_temp + w['temp']['dazhou']*dz_temp
                  + w['temp']['yibin']*yb_temp)
    rh_avg     = (w['rh']['chengdu']*cd_rh + w['rh']['dazhou']*dz_rh
                  + w['rh']['yibin']*yb_rh)
    rad_avg    = w['rad']['liangshan']*ls_rad + w['rad']['ganzi']*gz_rad
    cloud_avg  = w['cloud']['liangshan']*ls_cloud + w['cloud']['ganzi']*gz_cloud
    wind_avg   = w['wind']['liangshan']*ls_wind + w['wind']['ganzi']*gz_wind
    precip_avg = (w['precip']['yaan']*ya_precip + w['precip']['panzhihua']*pzh_precip
                  + w['precip']['liangshan']*ls_precip)

    # ── 衍生天气特征 ──
    cdd_avg = np.maximum(temp_avg - 18.0, 0)
    hdd_avg = np.maximum(18.0 - temp_avg, 0)
    temp_chg_24h_avg = temp_avg - np.roll(temp_avg, 96)

    # 体感温度（用户指定线性简化公式，替换旧 humidex）：1.07*T + 0.2*RH - 2.7
    apparent_temp = 1.07 * temp_avg + 0.2 * rh_avg - 2.7

    # 风能 ~ V^3（去掉 rho/2 常数系数，由模型学习）
    wind_cubic = wind_avg ** 3

    # 降水累积（hourly-aware 避免 ffill 4x 高估）
    rain_24h_mean = _hourly_rolling_sum(precip_avg, 24)     # 24h 累积降水
    rain_72h_acc = _hourly_rolling_sum(precip_avg, 72)      # 72h = 3天短期洪峰
    precip_15d_sum = _hourly_rolling_sum(precip_avg, 360)   # 360h = 15天水库基础水位

    # 有效辐射
    rad_eff = rad_avg * np.maximum(1.0 - cloud_avg / 100.0, 0)

    # 关键单点引用（v3: 集群平均替代单点，消除异常值风险）
    cloud_liangshan_raw = ls_cloud
    temp_liangshan_raw  = ls_temp
    cloud_chg_24h = cloud_liangshan_raw - np.roll(cloud_liangshan_raw, 96)
    temp_chg_24h_raw = temp_liangshan_raw - np.roll(temp_liangshan_raw, 96)

    # 成都集群平均气温（用于 EDA 标志 + 日较差）
    temp_chengdu_raw = cd_temp
    temp_diurnal_range = np.zeros(n)
    for day in range(n // 96):
        ds = day * 96
        de = min(ds + 96, n)
        temp_diurnal_range[ds:de] = (
            np.nanmax(temp_chengdu_raw[ds:de]) - np.nanmin(temp_chengdu_raw[ds:de])
        )

    # ── Season masks ──
    mask_dry = (month_arr >= 1) & (month_arr <= 4)
    mask_wet = (month_arr >= 5) & (month_arr <= 6)
    print(f"  枯水期: {mask_dry.sum()} 行, 丰水期: {mask_wet.sum()} 行")

    # ================================================================
    # Model-specific features (v10 decoupling - per physics mechanism)
    # ================================================================

    # ── Solar: 辐射断崖 + 云层移动 ──
    is_daytime = (rad_avg > 10.0).astype(float)          # 日间掩码
    is_daytime_pp = sf(is_daytime, H)                     # t+96 日间预报
    ghi_ramp_15min = rad_avg - np.roll(rad_avg, 1)       # 15min 辐射爬坡率
    ghi_ramp_15min[0] = 0.0

    # ── Wind: 启停逻辑 + 湍流 ──
    wind_avg_pp = sf(wind_avg, H)
    is_generating_pp = ((wind_avg_pp > 3.0) & (wind_avg_pp < 25.0)).astype(float)
    wind_turbulence_1h = roll_std(wind_avg, 4)           # 4 步 = 1h 湍流强度

    # ================================================================
    # Time encoding (t+96)
    # ================================================================
    print("  构建特征...")

    period_pp = sf(period_arr, H)
    month_pp = sf(month_arr, H)
    holiday_pp = sf(is_holiday, H)
    weekend_pp = sf(is_weekend, H)
    dry_season_pp = sf(is_dry_season, H)
    flood_pp = sf(is_flood, H)

    period_sin_pp = np.sin(2 * np.pi * period_pp / 96)
    period_cos_pp = np.cos(2 * np.pi * period_pp / 96)
    month_sin_pp = np.sin(2 * np.pi * month_pp / 12)
    month_cos_pp = np.cos(2 * np.pi * month_pp / 12)

    # 6 time slots
    slots_pp = np.zeros((n, 6))
    slots_pp[:, 0] = (period_pp >= 0) & (period_pp < 20)    # 深夜 0:00-4:45
    slots_pp[:, 1] = (period_pp >= 20) & (period_pp < 36)   # 早峰 5:00-8:45
    slots_pp[:, 2] = (period_pp >= 36) & (period_pp < 48)   # 上午 9:00-11:45
    slots_pp[:, 3] = (period_pp >= 48) & (period_pp < 68)   # 午谷 12:00-16:45
    slots_pp[:, 4] = (period_pp >= 68) & (period_pp < 88)   # 晚峰 17:00-21:45
    slots_pp[:, 5] = (period_pp >= 88) & (period_pp < 96)   # 夜间 22:00-23:45

    # ================================================================
    # Weather features (t + t+96 PP)
    # ================================================================
    w_fut = {
        'temp_avg_pp':      sf(temp_avg, H),
        'rad_avg_pp':       sf(rad_avg, H),
        'rad_eff_pp':       sf(rad_eff, H),
        'cloud_avg_pp':     sf(cloud_avg, H),
        'rain_h_avg_pp':    sf(precip_avg, H),
        'rain_24h_avg_pp':  sf(rain_24h_mean, H),
        'cdd_avg_pp':       sf(cdd_avg, H),
        'hdd_avg_pp':       sf(hdd_avg, H),
        'temp_chg_avg_pp':  sf(temp_chg_24h_avg, H),
        'apparent_temp_pp': sf(apparent_temp, H),
        'wind_cubic_pp':    sf(wind_cubic, H),
    }

    w_now = {
        'temp_avg_now':     temp_avg,
        'rad_avg_now':      rad_avg,
        'cloud_avg_now':    cloud_avg,
        'apparent_temp_now': apparent_temp,
        'wind_cubic_now':   wind_cubic,
    }

    rain_72h_acc_pp = sf(rain_72h_acc, H)

    # ================================================================
    # D-1 morning snapshot (L group, 16 dims)
    # ================================================================
    def d1_morning_op(arr, op):
        n_days = len(arr) // 96
        arr_2d = arr[:n_days * 96].reshape(n_days, 96)
        morning_slice = arr_2d[:, :35]
        if op == 'mean':
            val = np.mean(morning_slice, axis=1)
        elif op == 'last':
            val = morning_slice[:, -1]
        elif op == 'ramp':
            val = morning_slice[:, -1] - morning_slice[:, 24]
        elif op == 'std':
            val = np.std(morning_slice, axis=1)
        else:
            raise ValueError(f"Unknown op: {op}")
        result = np.repeat(val, 96)
        if len(result) < len(arr):
            result = np.pad(result, (0, len(arr) - len(result)), constant_values=np.nan)
        return result

    morning_mean_solar = d1_morning_op(solar, 'mean')
    morning_mean_hydro = d1_morning_op(hydro, 'mean')
    morning_mean_wind  = d1_morning_op(wind, 'mean')
    morning_mean_load  = d1_morning_op(load, 'mean')
    morning_last_solar = d1_morning_op(solar, 'last')
    morning_last_hydro = d1_morning_op(hydro, 'last')
    morning_last_wind  = d1_morning_op(wind, 'last')
    morning_last_load  = d1_morning_op(load, 'last')
    morning_ramp_solar = d1_morning_op(solar, 'ramp')
    morning_ramp_hydro = d1_morning_op(hydro, 'ramp')
    morning_ramp_wind  = d1_morning_op(wind, 'ramp')
    morning_ramp_load  = d1_morning_op(load, 'ramp')
    morning_std_solar  = d1_morning_op(solar, 'std')
    morning_std_hydro  = d1_morning_op(hydro, 'std')
    morning_std_wind   = d1_morning_op(wind, 'std')
    morning_std_load   = d1_morning_op(load, 'std')

    # ================================================================
    # D-2 daily curve (N group, 18 dims)
    # ================================================================
    def d2_daily(arr, op):
        n_days = len(arr) // 96
        arr_2d = arr[:n_days * 96].reshape(n_days, 96)
        if op == 'peak':
            val = np.max(arr_2d, axis=1)
        elif op == 'mean':
            val = np.mean(arr_2d, axis=1)
        elif op == 'range':
            val = np.max(arr_2d, axis=1) - np.min(arr_2d, axis=1)
        elif op == 'std':
            val = np.std(arr_2d, axis=1)
        elif op == 'total':
            val = np.sum(arr_2d, axis=1)
        elif op == 'night_mean':
            night = np.concatenate([arr_2d[:, :20], arr_2d[:, 88:]], axis=1)
            val = np.mean(night, axis=1)
        elif op == 'daytime_mean':
            val = np.mean(arr_2d[:, 28:76], axis=1)
        elif op == 'ramp_up':
            daytime = arr_2d[:, 28:76]
            val = np.max(daytime, axis=1) - arr_2d[:, 28]
        elif op == 'duration':
            val = np.sum(arr_2d > 100, axis=1).astype(float)
        elif op == 'morning_peak':
            val = np.max(arr_2d[:, 20:36], axis=1)
        elif op == 'evening_peak':
            val = np.max(arr_2d[:, 68:88], axis=1)
        elif op == 'valley':
            val = np.min(arr_2d[:, 40:68], axis=1)
        elif op == 'peak_ratio':
            eve = np.max(arr_2d[:, 68:88], axis=1)
            mor = np.max(arr_2d[:, 20:36], axis=1)
            val = np.where(mor > 0, eve / (mor + 1), 0.0)
        else:
            raise ValueError(f"Unknown op: {op}")
        result = np.full(len(arr), np.nan)
        for d in range(1, n_days):
            result[d*96:(d+1)*96] = val[d-1]
        return result

    d2_solar_peak       = d2_daily(solar, 'peak')
    d2_solar_ramp_up    = d2_daily(solar, 'ramp_up')
    d2_solar_day_std    = d2_daily(solar, 'std')
    d2_solar_duration   = d2_daily(solar, 'duration')
    d2_solar_day_mean   = d2_daily(solar, 'mean')
    d2_solar_day_total  = d2_daily(solar, 'total')
    d2_hydro_range        = d2_daily(hydro, 'range')
    d2_hydro_day_mean     = d2_daily(hydro, 'mean')
    d2_hydro_night_mean   = d2_daily(hydro, 'night_mean')
    d2_hydro_daytime_mean = d2_daily(hydro, 'daytime_mean')
    d2_wind_range       = d2_daily(wind, 'range')
    d2_wind_day_mean    = d2_daily(wind, 'mean')
    d2_wind_night_mean  = d2_daily(wind, 'night_mean')
    d2_load_morning_peak = d2_daily(load, 'morning_peak')
    d2_load_evening_peak = d2_daily(load, 'evening_peak')
    d2_load_valley       = d2_daily(load, 'valley')
    d2_load_peak_ratio   = d2_daily(load, 'peak_ratio')
    d2_load_day_mean     = d2_daily(load, 'mean')

    # Spring Festival 2026
    spring_festival = np.datetime64('2026-02-17')
    days_from_sf = (dt_arr.values - spring_festival).astype('timedelta64[D]').astype(float)
    is_sf_window = (np.abs(days_from_sf) <= 10).astype(float)

    # ================================================================
    # Feature matrix assembly (A-P groups, 177 dims total)
    # ================================================================
    feature_list = []

    def add_feat(group, name, arr):
        feature_list.append((group, name, safe_fillna(arr, 0.0)))

    # ── A. Price momentum (12) ──
    add_feat('A', 'price_lag_96', np.roll(price, 96))
    add_feat('A', 'price_lag_192', np.roll(price, 192))
    add_feat('A', 'price_lag_288', np.roll(price, 288))
    add_feat('A', 'price_lag_672', np.roll(price, 672))
    add_feat('A', 'price_chg_3d', price - np.roll(price, 288))
    add_feat('A', 'price_chg_7d', price - np.roll(price, 672))
    add_feat('A', 'price_vol_24h', roll_std(np.roll(price, 1), 96))
    add_feat('A', 'price_ma_24h', roll_mean(np.roll(price, 1), 96))
    add_feat('A', 'price_ma_7d', roll_mean(np.roll(price, 1), 672))
    add_feat('A', 'price_accel', price - 2 * np.roll(price, 96) + np.roll(price, 192))
    add_feat('A', 'price_diff_1d', price - np.roll(price, 96))
    add_feat('A', 'price_max_30d', roll_max(price, 720))

    # ── B. Generation (14) ──
    add_feat('B', 'solar[t]', solar)
    add_feat('B', 'wind[t]', wind)
    add_feat('B', 'hydro[t]', hydro)
    add_feat('B', 'load[t]', load)
    add_feat('B', 'solar_lag_96', np.roll(solar, 96))
    add_feat('B', 'solar_lag_672', np.roll(solar, 672))
    add_feat('B', 'wind_lag_4', np.roll(wind, 4))
    add_feat('B', 'wind_lag_96', np.roll(wind, 96))
    add_feat('B', 'wind_lag_672', np.roll(wind, 672))
    add_feat('B', 'hydro_lag_96', np.roll(hydro, 96))
    add_feat('B', 'hydro_lag_672', np.roll(hydro, 672))
    add_feat('B', 'load_lag_96', np.roll(load, 96))
    add_feat('B', 'load_lag_672', np.roll(load, 672))
    add_feat('B', 'hydro_ma_7d', roll_mean(np.roll(hydro, 1), 672))

    # ── C. Supply-demand (10) ──
    net_load = load - solar - wind - hydro
    surplus = solar + wind + hydro * 0.3 - load + 6000
    renew_total = solar + wind + hydro
    add_feat('C', 'net_load[t]', net_load)
    add_feat('C', 'surplus[t]', surplus)
    add_feat('C', 'solar_pen[t]', nan0(solar / (load + 1)))
    add_feat('C', 'hydro_pen[t]', nan0(hydro / (load + 1)))
    add_feat('C', 'wind_pen[t]', nan0(wind / (load + 1)))
    add_feat('C', 'penetration[t]', nan0(renew_total / (load + 1)))
    add_feat('C', 'net_load_lag_96', np.roll(net_load, 96))
    add_feat('C', 'surplus_lag_96', np.roll(surplus, 96))
    add_feat('C', 'surplus_scaled', surplus / (load + 1))
    add_feat('C', 'renew_total[t]', renew_total)

    # ── D. Weather (20: 14 fused + 6 model-specific) ──
    for k in ['temp_avg_pp', 'rad_avg_pp', 'rad_eff_pp', 'cloud_avg_pp',
              'rain_24h_avg_pp', 'rain_h_avg_pp', 'cdd_avg_pp',
              'apparent_temp_pp', 'wind_cubic_pp']:
        if k in w_fut:
            add_feat('D', k, w_fut[k])
    for k in ['temp_avg_now', 'rad_avg_now', 'cloud_avg_now',
              'apparent_temp_now', 'wind_cubic_now']:
        if k in w_now:
            add_feat('D', k, w_now[k])
    add_feat('D', 'is_daytime', is_daytime)
    add_feat('D', 'is_daytime_pp', is_daytime_pp)
    add_feat('D', 'ghi_ramp_15min', ghi_ramp_15min)
    add_feat('D', 'is_generating_pp', is_generating_pp)
    add_feat('D', 'wind_turbulence_1h', wind_turbulence_1h)
    add_feat('D', 'precip_15d_sum', precip_15d_sum)
    # Basin-level precipitation at t+96 (amplify hydro signal)
    add_feat('D', 'precip_yaan_pp', sf(ya_precip, H))
    add_feat('D', 'precip_pzh_pp', sf(pzh_precip, H))
    add_feat('D', 'precip_ls_pp', sf(ls_precip, H))
    # Nonlinear precipitation response (flood → nonlinear runoff)
    add_feat('D', 'precip_avg_sq_pp', sf(precip_avg ** 2, H))       # 强降水非线性放大
    add_feat('D', 'precip_avg_sqrt_pp', sf(np.sqrt(precip_avg + 0.01), H))  # 弱降水敏感性

    # ── E. Grid market (4) ──
    add_feat('E', 'bidspace[t]', bidspace)
    add_feat('E', 'bidspace_ratio', nan0(bidspace / (load + 1)))
    add_feat('E', 'nonmarket[t]', nonmarket)
    add_feat('E', 'nonmarket_ratio', nan0(nonmarket / (load + 1)))

    # ── F. Time encoding (14) ──
    add_feat('F', 'period_sin_pp', period_sin_pp)
    add_feat('F', 'period_cos_pp', period_cos_pp)
    add_feat('F', 'month_sin_pp', month_sin_pp)
    add_feat('F', 'month_cos_pp', month_cos_pp)
    add_feat('F', 'is_holiday_pp', holiday_pp)
    add_feat('F', 'is_weekend_pp', weekend_pp)
    add_feat('F', 'is_dry_season_pp', dry_season_pp)
    add_feat('F', 'is_flood_pp', flood_pp)
    slot_names = ['深夜', '早峰', '上午', '午谷', '晚峰', '夜间']
    for i, sn in enumerate(slot_names):
        add_feat('F', f'slot_{sn}_pp', slots_pp[:, i])

    # ── G. EDA flags (6) ──
    def monthly_quantile_flag(arr, upper_q=0.90, lower_q=0.10):
        flags = np.zeros(n)
        for m in range(1, 13):
            mm = month_arr == m
            if mm.sum() > 100:
                lo = np.quantile(arr[mm], lower_q)
                hi = np.quantile(arr[mm], upper_q)
                flags[mm & (arr > hi)] = 1
                flags[mm & (arr < lo)] = -1
        return flags

    add_feat('G', 'extreme_load', monthly_quantile_flag(load))
    add_feat('G', 'extreme_price', monthly_quantile_flag(price, 0.95, 0.05))
    add_feat('G', 'extreme_solar', monthly_quantile_flag(solar))
    add_feat('G', 'extreme_wind', monthly_quantile_flag(wind))
    add_feat('G', 'high_load_low_solar',
             ((load > np.roll(np.quantile(load, 0.7), 1)) &
              (solar < np.quantile(solar, 0.3))).astype(float))
    add_feat('G', 'high_temp_low_wind',
             ((temp_chengdu_raw > 25) &
              (wind < np.quantile(wind, 0.3))).astype(float))

    # ── H. Rolling stats (4) ──
    add_feat('H', 'load_max_30d', roll_max(load, 720))
    add_feat('H', 'solar_ma_30d', roll_mean(np.roll(solar, 1), 720))
    add_feat('H', 'wind_ma_30d', roll_mean(np.roll(wind, 1), 720))
    add_feat('H', 'hydro_ma_30d', roll_mean(np.roll(hydro, 1), 720))

    # ── I. Baseline (1) ──
    add_feat('I', 'sim7d', np.roll(price, 672))

    # ── J. Interaction (6: 3 original + 3 precip×hydro) ──
    add_feat('J', 'rad_pp_x_period_cos',
             w_fut.get('rad_avg_pp', np.zeros(n)) * period_cos_pp)
    add_feat('J', 'rain72h_x_flood', rain_72h_acc_pp * flood_pp)
    add_feat('J', 'cloud_chg_x_temp_chg', cloud_chg_24h * temp_chg_24h_raw)
    # Precipitation × hydro interactions (amplify precip signal for hydro residual)
    hydro_now = np.roll(hydro, 0)
    add_feat('J', 'rain24h_x_hydro', w_fut.get('rain_24h_avg_pp', np.zeros(n)) * hydro_now)
    add_feat('J', 'precip_yaan_x_flood', sf(ya_precip, H) * flood_pp)
    add_feat('J', 'precip_15d_x_hydro', precip_15d_sum * hydro_now)

    # ── K. Stage1-specific (19) ──
    add_feat('K', 'solar_lag_4', np.roll(solar, 4))
    add_feat('K', 'solar_lag_192', np.roll(solar, 192))
    add_feat('K', 'solar_ma_24h', roll_mean(np.roll(solar, 1), 96))
    add_feat('K', 'solar_ma_7d', roll_mean(np.roll(solar, 1), 672))
    add_feat('K', 'solar_diff_1d', solar - np.roll(solar, 96))
    add_feat('K', 'hydro_lag_4', np.roll(hydro, 4))
    add_feat('K', 'hydro_ma_24h', roll_mean(np.roll(hydro, 1), 96))
    add_feat('K', 'hydro_diff_1d', hydro - np.roll(hydro, 96))
    add_feat('K', 'hydro_diff_7d', hydro - np.roll(hydro, 672))
    add_feat('K', 'hydro_vol_24h', roll_std(np.roll(hydro, 1), 96))
    add_feat('K', 'wind_ma_24h', roll_mean(np.roll(wind, 1), 96))
    add_feat('K', 'wind_ma_7d', roll_mean(np.roll(wind, 1), 672))
    add_feat('K', 'wind_vol_24h', roll_std(np.roll(wind, 1), 96))
    add_feat('K', 'wind_diff_1d', wind - np.roll(wind, 96))
    add_feat('K', 'cloud_chg_24h', cloud_chg_24h)
    add_feat('K', 'temp_chg_24h', temp_chg_24h_raw)
    add_feat('K', 'temp_diurnal_range', temp_diurnal_range)
    add_feat('K', 'rain_72h_acc[t]', rain_72h_acc)
    add_feat('K', 'rain_72h_acc_pp', rain_72h_acc_pp)

    # ── L. D-1 morning snapshot (16) ──
    add_feat('L', 'morning_mean_solar', morning_mean_solar)
    add_feat('L', 'morning_mean_hydro', morning_mean_hydro)
    add_feat('L', 'morning_mean_wind',  morning_mean_wind)
    add_feat('L', 'morning_mean_load',  morning_mean_load)
    add_feat('L', 'morning_last_solar', morning_last_solar)
    add_feat('L', 'morning_last_hydro', morning_last_hydro)
    add_feat('L', 'morning_last_wind',  morning_last_wind)
    add_feat('L', 'morning_last_load',  morning_last_load)
    add_feat('L', 'morning_ramp_solar', morning_ramp_solar)
    add_feat('L', 'morning_ramp_hydro', morning_ramp_hydro)
    add_feat('L', 'morning_ramp_wind',  morning_ramp_wind)
    add_feat('L', 'morning_ramp_load',  morning_ramp_load)
    add_feat('L', 'morning_std_solar',  morning_std_solar)
    add_feat('L', 'morning_std_hydro',  morning_std_hydro)
    add_feat('L', 'morning_std_wind',   morning_std_wind)
    add_feat('L', 'morning_std_load',   morning_std_load)

    # ── M. Spring Festival (2) ──
    add_feat('M', 'days_from_sf', days_from_sf)
    add_feat('M', 'is_sf_window', is_sf_window)

    # ── N. D-2 daily curve (18) ──
    add_feat('N', 'd2_solar_peak',       d2_solar_peak)
    add_feat('N', 'd2_solar_ramp_up',    d2_solar_ramp_up)
    add_feat('N', 'd2_solar_day_std',    d2_solar_day_std)
    add_feat('N', 'd2_solar_duration',   d2_solar_duration)
    add_feat('N', 'd2_solar_day_mean',   d2_solar_day_mean)
    add_feat('N', 'd2_solar_day_total',  d2_solar_day_total)
    add_feat('N', 'd2_hydro_range',      d2_hydro_range)
    add_feat('N', 'd2_hydro_day_mean',   d2_hydro_day_mean)
    add_feat('N', 'd2_hydro_night_mean', d2_hydro_night_mean)
    add_feat('N', 'd2_hydro_daytime_mean', d2_hydro_daytime_mean)
    add_feat('N', 'd2_wind_range',       d2_wind_range)
    add_feat('N', 'd2_wind_day_mean',    d2_wind_day_mean)
    add_feat('N', 'd2_wind_night_mean',  d2_wind_night_mean)
    add_feat('N', 'd2_load_morning_peak', d2_load_morning_peak)
    add_feat('N', 'd2_load_evening_peak', d2_load_evening_peak)
    add_feat('N', 'd2_load_valley',      d2_load_valley)
    add_feat('N', 'd2_load_peak_ratio',  d2_load_peak_ratio)
    add_feat('N', 'd2_load_day_mean',    d2_load_day_mean)

    # ── O. D-2 period-level dynamics (16) ──
    add_feat('O', 'wind_lag_192', np.roll(wind, 192))
    add_feat('O', 'wind_lag_193', np.roll(wind, 193))
    add_feat('O', 'wind_lag_196', np.roll(wind, 196))
    add_feat('O', 'wind_d2_ramp_1h', np.roll(wind, 192) - np.roll(wind, 196))
    add_feat('O', 'wind_d2_ramp_2h', np.roll(wind, 192) - np.roll(wind, 200))
    add_feat('O', 'hydro_lag_192', np.roll(hydro, 192))
    add_feat('O', 'hydro_lag_193', np.roll(hydro, 193))
    add_feat('O', 'hydro_lag_196', np.roll(hydro, 196))
    add_feat('O', 'hydro_lag_288', np.roll(hydro, 288))
    add_feat('O', 'hydro_d2_ramp_1h', np.roll(hydro, 192) - np.roll(hydro, 196))
    add_feat('O', 'load_lag_192', np.roll(load, 192))
    add_feat('O', 'load_lag_193', np.roll(load, 193))
    add_feat('O', 'load_lag_196', np.roll(load, 196))
    add_feat('O', 'load_lag_200', np.roll(load, 200))
    add_feat('O', 'load_d2_ramp_1h', np.roll(load, 192) - np.roll(load, 196))
    add_feat('O', 'load_d2_ramp_2h', np.roll(load, 192) - np.roll(load, 200))

    # ── P. D-2 2h window trajectory (18) ──
    def d2_window_stats(arr, s=192, e=200):
        cols = [np.roll(arr, lag) for lag in range(s, e + 1)]
        w = np.column_stack(cols)
        r = {}
        r['mean'] = np.mean(w, axis=1)
        r['std'] = np.std(w, axis=1, ddof=0)
        r['trend'] = w[:, 0] - w[:, -1]
        r['range'] = np.max(w, axis=1) - np.min(w, axis=1)
        r['max_step'] = np.max(np.abs(np.diff(w, axis=1)), axis=1)
        r['accel'] = (w[:, 0] - w[:, 4]) - (w[:, 4] - w[:, -1])
        return r

    for var, label in [(wind, 'wind'), (hydro, 'hydro'), (load, 'load')]:
        ws = d2_window_stats(var)
        for stat in ['mean', 'std', 'trend', 'range', 'max_step', 'accel']:
            add_feat('P', f'{label}_d2_w2h_{stat}', ws[stat])

    # ── Assemble ──
    feat_names = [f'{g}_{n}' for g, n, _ in feature_list]
    X = np.column_stack([arr for _, _, arr in feature_list])

    # Targets always use ORIGINAL (unshifted) arrays — we still predict real t+96 values
    y_solar = sf(solar_orig, H)
    y_hydro = sf(hydro_orig, H)
    y_wind = sf(wind_orig, H)
    y_price_resid = sf(price_orig, H) - price_orig
    y_price_raw = sf(price_orig, H)
    price_lag96 = np.roll(price_orig, 96)

    print(f"  总特征维度: {len(feat_names)}")
    for grp in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']:
        grp_feats = [n for g, n, _ in feature_list if g == grp]
        if grp_feats:
            print(f"    {grp}组 ({len(grp_feats)}): {', '.join(grp_feats[:6])}"
                  f"{'...' if len(grp_feats) > 6 else ''}")

    # Feature selectors (v10 decoupling)
    solar_feat_cols = [i for i, (g, n, _) in enumerate(feature_list)
                       if 'solar' in n.lower() or 'rad' in n or 'period_cos' in n
                       or 'is_daytime' in n or 'ghi_ramp' in n
                       or n == 'morning_mean_solar']
    hydro_feat_cols = [i for i, (g, n, _) in enumerate(feature_list)
                       if 'hydro' in n.lower() or 'rain' in n or 'flood' in n
                       or 'precip_15d' in n
                       or n == 'morning_mean_hydro']
    wind_feat_cols = [i for i, (g, n, _) in enumerate(feature_list)
                      if 'wind' in n.lower() or 'cloud_chg' in n or 'temp_chg' in n
                      or 'diurnal' in n or 'is_generating' in n or 'turbulence' in n
                      or n == 'morning_mean_wind']
    load_feat_cols = [i for i, (g, n, _) in enumerate(feature_list)
                      if 'load' in n.lower() or 'morning_mean_load' in n.lower()
                      or 'sf' in n.lower() or 'temp_avg_pp' in n or 'cdd' in n
                      or 'apparent_temp' in n]

    return {
        'X': X,
        'feat_names': np.array(feat_names),
        'feature_list': feature_list,
        'y_solar': y_solar,
        'y_hydro': y_hydro,
        'y_wind': y_wind,
        'y_price_resid': y_price_resid,
        'y_price_raw': y_price_raw,
        'price_lag96': price_lag96,
        'price': price_orig,  # always original — used for anchor computation in Stage2
        'dt': dt_arr.values,
        'period': period_arr,
        'month': month_arr,
        'n_samples': n,
        'dry_mask': mask_dry,
        'wet_mask': mask_wet,
        'solar_feat_cols': solar_feat_cols,
        'hydro_feat_cols': hydro_feat_cols,
        'wind_feat_cols': wind_feat_cols,
        'load_feat_cols': load_feat_cols,
    }


# ====================================================================
# CLI
# ====================================================================

def main():
    """Orchestrate: load -> build -> save -> verify."""
    from pipeline.data_loader import load_from_db
    from pipeline.output import save_outputs, verify

    parser = argparse.ArgumentParser(description="DB-backed feature engineering (v3)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only compare with existing npz (skip build)")
    parser.add_argument("--grid-lag", type=int, default=0,
                        help="Lag grid arrays by N periods (192 = gap-fill for t-2). "
                             "0 = normal mode.")
    parser.add_argument("--forward-extend", type=int, default=0,
                        help="Extend datetime index by N periods beyond last grid row. "
                             "Use with --grid-lag 192 to extend forecast horizon.")
    args = parser.parse_args()

    if args.verify_only:
        verify()
        return

    lag_tag = f" [grid_lag={args.grid_lag}]" if args.grid_lag > 0 else ""
    ext_tag = f" [forward_extend={args.forward_extend}]" if args.forward_extend > 0 else ""
    print("=" * 70)
    print(f"  feature_engine.py - DB-backed feature engineering{lag_tag}{ext_tag}")
    print("=" * 70)

    if args.forward_extend > 0:
        from pipeline.data_loader import load_for_inference
        (dt_arr, df, solar, wind, hydro, load, price,
         bidspace, reserve, nonmarket, tieline, load_tie) = load_for_inference(
            forward_extend=args.forward_extend)
    else:
        (dt_arr, df, solar, wind, hydro, load, price,
         bidspace, reserve, nonmarket, tieline, load_tie) = load_from_db()

    result = build_features(dt_arr, df, solar, wind, hydro, load, price,
                            bidspace, reserve, nonmarket, tieline, load_tie,
                            grid_lag=args.grid_lag)

    fp_dry, fp_wet = save_outputs(result, grid_lag=args.grid_lag)

    if not args.grid_lag:
        verify()

    print(f"\n{'=' * 70}")
    print(f"  Complete!")
    print(f"    {fp_dry}")
    print(f"    {fp_wet}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
