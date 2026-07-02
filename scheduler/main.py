# -*- coding: utf-8 -*-
"""
scheduler/main.py — 定时任务调度器（Phase 4）
================================================
职责：管理所有定时任务的注册和执行。设计为独立进程运行（Docker scheduler 容器），
通过 APScheduler (BackgroundScheduler) 管理任务生命周期。

定时任务清单（北京时间 CST）：
  refresh_token            55 0 * * *    每日 00:55 刷新 API Token
  fetch_grid               0 1 * * *     每日 01:00 拉取昨日电网数据
  fetch_weather            30 1 * * *    每日 01:30 拉取今日 NWP 气象预报
  daily_inference          0 2 * * *     每日 02:00 执行推理 → predictions 表（9点前出第二天预测）
  validate_data            30 2 * * *    每日 02:30 校验昨日数据质量
  refresh_token_and_fetch  0 12 * * *    每日 12:00 刷新 Token + 补拉电网数据（备份）
  weekly_retrain           0 3 * * 0     每周日 03:00 全量重训练
  hourly_health            0 * * * *     每小时健康心跳

注意：
  - cron 使用 Asia/Shanghai 时区 (CST = UTC+8)
  - 调度器在独立 Docker 容器中运行（不影响 API 推理性能）
  - 训练任务内存需求 4G，见 docker-compose.yml

启动方式：
  python -m scheduler.main                      # 前台持续运行
  python -m scheduler.main --once               # 执行一次所有任务后退出
  python -m scheduler.main --job fetch_grid     # 只执行指定任务
  python -m scheduler.main --list               # 列出所有任务
"""
import sys
import io
import time
import signal
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, date, timedelta
from apscheduler.triggers.cron import CronTrigger
warnings.filterwarnings("ignore")
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

# ====================================================================
# 日志配置
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("scheduler")


# ====================================================================
# 任务函数定义（每个任务 = 一个独立可调用的函数，返回 dict 结果）
# ====================================================================

def job_fetch_weather() -> dict:
    """拉取 NWP 气象预报 → weather_forecast 表。

    通过 Open-Meteo ECMWF forecast API 获取未来 4 天预报（19 子节点 × 6 变量）。
    """
    logger.info("=" * 50)
    logger.info("JOB: fetch_weather — 拉取气象预报")
    start = time.time()
    try:
        from ingestion.weather_fetcher import fetch_weather_forecast
        n = fetch_weather_forecast()
        elapsed = time.time() - start
        logger.info(f"JOB: fetch_weather 完成 — {n} 条预报 ({elapsed:.1f}s)")
        return {'ok': True, 'records': n, 'elapsed': round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"JOB: fetch_weather 失败 — {e}", exc_info=True)
        return {'ok': False, 'error': str(e), 'elapsed': round(elapsed, 1)}


def job_fetch_grid() -> dict:
    """拉取昨日电网数据 → grid_data 表。

    每日 09:05 执行：此时前一日 96 条 15 分钟数据应已发布到电网平台。
    Token 从 grid_token.txt 读取，过期时自动重新登录。
    """
    logger.info("=" * 50)
    logger.info("JOB: fetch_grid — 拉取昨日电网数据")
    start = time.time()
    try:
        from ingestion.grid_fetcher import fetch_grid_data
        yesterday = date.today() - timedelta(days=1)
        n = fetch_grid_data(target_date=yesterday)
        elapsed = time.time() - start
        logger.info(f"JOB: fetch_grid 完成 — {n} 条 ({elapsed:.1f}s)")
        return {'ok': True, 'records': n, 'date': yesterday.isoformat(), 'elapsed': round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"JOB: fetch_grid 失败 — {e}", exc_info=True)
        return {'ok': False, 'error': str(e), 'elapsed': round(elapsed, 1)}


def job_refresh_token() -> dict:
    """每日 00:55：仅刷新 API Token，写入 grid_token.txt。

    凌晨 01:00 拉取电网数据前确保 token 有效。
    """
    logger.info("=" * 50)
    logger.info("JOB: refresh_token — 刷新 API Token")
    start = time.time()
    try:
        from ingestion.auth_login import login
        token = login()
        elapsed = time.time() - start
        logger.info(f"JOB: refresh_token 完成 — {len(token)} chars ({elapsed:.1f}s)")
        return {'ok': True, 'token_len': len(token), 'elapsed': round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"JOB: refresh_token 失败 — {e}", exc_info=True)
        return {'ok': False, 'error': str(e), 'elapsed': round(elapsed, 1)}


def job_refresh_token_and_fetch() -> dict:
    """每日 12:00：重新登录获取新 token → 拉取昨日电网数据（备份）。

    即使凌晨 01:00 那次 fetch_grid 失败，中午也能补上。
    """
    logger.info("=" * 50)
    logger.info("JOB: refresh_token_and_fetch — 刷新 Token + 拉取电网数据")
    start = time.time()
    try:
        # Step 1: 登录刷新 token
        from ingestion.auth_login import login
        logger.info("  [token] 登录获取新 token...")
        token = login()
        logger.info(f"  [token] 完成 — {len(token)} chars")

        # Step 2: 拉取数据
        from ingestion.grid_fetcher import fetch_grid_data
        yesterday = date.today() - timedelta(days=1)
        n = fetch_grid_data(target_date=yesterday)
        elapsed = time.time() - start
        logger.info(f"JOB: refresh_token_and_fetch 完成 — Token已刷新, 数据{n}条 ({elapsed:.1f}s)")
        return {'ok': True, 'records': n, 'date': yesterday.isoformat(), 'elapsed': round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"JOB: refresh_token_and_fetch 失败 — {e}", exc_info=True)
        return {'ok': False, 'error': str(e), 'elapsed': round(elapsed, 1)}


def job_daily_inference() -> dict:
    """执行每日推理：特征工程 → Stage1 → Stage2 → 存入 predictions 表。

    每日 09:10 执行：此时电网数据和气象预报应已就绪。
    预测目标 t+96（24 小时后），即预测次日的 96 个时段价格。

    流程：
      1. 从 DB 重新构建全量特征矩阵（feature_engine）
      2. 加载 14 个模型（Stage1 × 8 + Stage2 × 6）
      3. 运行完整推理链路
      4. 取最后 96 行预测写入 predictions 表
    """
    logger.info("=" * 50)
    logger.info("JOB: daily_inference — 执行每日推理")
    start = time.time()
    try:
        import numpy as np
        import pandas as pd
        from sqlalchemy import create_engine, text

        # 1. 从 DB 重新构建特征（确保使用最新数据）
        logger.info("  [inference] 构建特征矩阵...")
        from pipeline.data_loader import load_for_inference
        from pipeline.feature_engine import build_features
        from pipeline.output import save_outputs

        (dt_arr, df, solar, wind, hydro, load, price,
         bidspace, reserve, nonmarket, tieline, load_tie) = load_for_inference()

        result = build_features(dt_arr, df, solar, wind, hydro, load, price,
                                bidspace, reserve, nonmarket, tieline, load_tie)

        # 保存 npz（供 API 下次启动使用）
        save_outputs(result)

        X = result['X']
        feat_names_arr = result['feat_names']
        period = result['period']
        price = result['price']
        dry_mask = result['dry_mask']
        wet_mask = result['wet_mask']

        n = X.shape[0]
        logger.info(f"  [inference] 特征: {n} 行 × {X.shape[1]} 维")

        # 2. 加载模型
        logger.info("  [inference] 加载模型...")
        from pipeline.inference import (
            load_stage1_models, load_stage2_models,
            predict_stage1, build_stage2_features, blend_weights,
        )

        m1 = load_stage1_models()
        m2 = load_stage2_models()
        fnames = list(feat_names_arr)

        # 3. Stage1 预测（4 变量 × 2 季节）
        oof_s = np.full(n, np.nan); oof_h = np.full(n, np.nan)
        oof_w = np.full(n, np.nan); oof_l = np.full(n, np.nan)

        for season, mask in [('dry', dry_mask), ('wet', wet_mask)]:
            s, h, w, l = predict_stage1(m1, X, fnames, season)
            oof_s[mask] = s[mask]; oof_h[mask] = h[mask]
            oof_w[mask] = w[mask]; oof_l[mask] = l[mask]
        for name, a in [('solar', oof_s), ('hydro', oof_h), ('wind', oof_w), ('load', oof_l)]:
            nan_count = np.isnan(a).sum()
            if nan_count > 0:
                logger.warning(f"  Stage1 {name}: {nan_count} NaN OOF → filled with 0")
            a[np.isnan(a)] = 0.0

        # 4. Stage2 预测（使用训练时保存的特征索引）
        if not m2:
            logger.error("No Stage2 models loaded — cannot run inference")
            return {'ok': False, 'error': 'No Stage2 models loaded'}
        first_m2 = next(iter(m2.values()))
        safe_idx = first_m2.get('safe_indices') if isinstance(first_m2, dict) else None
        X_s2 = build_stage2_features(X, fnames, oof_s, oof_h, oof_w, oof_l,
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
            price_pred[idx] = anchor[idx] + blended  # residual + anchor

        nan_price = np.isnan(price_pred).sum()
        if nan_price > 0:
            logger.warning(f"  price_pred: {nan_price} NaN values → filled with 0")
        price_pred[np.isnan(price_pred)] = 0.0
        logger.info(f"  [inference] 价格预测: mean={np.mean(price_pred):.1f} 元/MWh")

        # 5. 写入 predictions 表（最后 96 行 = 最新完整天）
        url = settings.database_url_sync
        engine = create_engine(url, echo=False)

        if n >= 96:
            last_preds = price_pred[-96:]
            last_dt = dt_arr[-96:]

            records = []
            for i in range(96):
                data_time = pd.Timestamp(last_dt[i])
                target_time = data_time + pd.Timedelta(hours=24)  # t+96
                season_label = 'dry' if dry_mask[n - 96 + i] else 'wet'
                records.append({
                    'target_time': target_time.to_pydatetime(),
                    'predicted_price': float(last_preds[i]),
                    'model_version': settings.feature_version,
                    'season': season_label,
                    'period': int(period[n - 96 + i]),
                })

            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(
                        text("""
                            INSERT INTO predictions
                                (target_time, predicted_price, model_version, season, period)
                            SELECT :target_time, :predicted_price, :model_version, :season, :period
                            WHERE NOT EXISTS (
                                SELECT 1 FROM predictions
                                WHERE target_time = :target_time
                                  AND model_version = :model_version
                            )
                        """),
                        records,
                    )
                    conn.commit()
            logger.info(f"  [inference] 写入 {len(records)} 条预测")

        engine.dispose()

        # ── 6. Gap-fill pass: forward_extend=192 + grid_lag=192 ──
        # Extend datetime 192 periods beyond grid data. New rows have NaN
        # grid but real weather forecast. grid_lag=192 fills grid[t] with
        # grid[t-192] which IS available → predicts 2 extra days.
        logger.info("  [inference] Gap-fill pass (forward_extend=192, grid_lag=192)...")
        t_lag = time.time()
        try:
            from pipeline.inference import _load_s1_for_lag, _load_s2_for_lag

            m1_lag = _load_s1_for_lag(grid_lag=192)
            m2_lag = _load_s2_for_lag(grid_lag=192)

            if len(m1_lag) >= 4 and len(m2_lag) >= 3:
                # Re-load with forward extension
                (dt_arr_lag, df_lag, solar_lag, wind_lag, hydro_lag, load_lag,
                 price_lag, bidspace_lag, reserve_lag, nonmarket_lag,
                 tieline_lag, load_tie_lag) = load_for_inference(forward_extend=192)

                result_lag = build_features(dt_arr_lag, df_lag,
                                            solar_lag, wind_lag, hydro_lag, load_lag,
                                            price_lag, bidspace_lag, reserve_lag,
                                            nonmarket_lag, tieline_lag, load_tie_lag,
                                            grid_lag=192)
                save_outputs(result_lag, grid_lag=192)

                X_lag = result_lag['X']
                n_lag = X_lag.shape[0]  # may be larger than n due to forward_extend
                fnames_lag = list(result_lag['feat_names'])
                period_lag = result_lag['period']
                price_lag = result_lag['price']
                dry_lag = result_lag['dry_mask']
                wet_lag = result_lag['wet_mask']

                oof_s_lag = np.full(n_lag, np.nan); oof_h_lag = np.full(n_lag, np.nan)
                oof_w_lag = np.full(n_lag, np.nan); oof_l_lag = np.full(n_lag, np.nan)

                for season, mask in [('dry', dry_lag), ('wet', wet_lag)]:
                    s, h, w, l_ = predict_stage1(m1_lag, X_lag, fnames_lag, season)
                    oof_s_lag[mask] = s[mask]; oof_h_lag[mask] = h[mask]
                    oof_w_lag[mask] = w[mask]; oof_l_lag[mask] = l_[mask]
                for name, a in [('solar', oof_s_lag), ('hydro', oof_h_lag),
                               ('wind', oof_w_lag), ('load', oof_l_lag)]:
                    nan_count = np.isnan(a).sum()
                    if nan_count > 0:
                        logger.warning(f"  Stage1 lag {name}: {nan_count} NaN OOF → filled with 0")
                    a[np.isnan(a)] = 0.0

                first_m2_lag = next(iter(m2_lag.values()))
                safe_idx_lag = first_m2_lag.get('safe_indices') if isinstance(first_m2_lag, dict) else None
                X_s2_lag = build_stage2_features(X_lag, fnames_lag, oof_s_lag, oof_h_lag,
                                                 oof_w_lag, oof_l_lag, period_lag,
                                                 safe_indices=safe_idx_lag)

                lag96_l = np.roll(price_lag, 96); lag96_l[:96] = np.nan
                lag672_l = np.roll(price_lag, 672); lag672_l[:672] = np.nan
                anchor_l = (lag96_l + lag672_l) / 2.0

                price_pred_lag = np.full(n_lag, np.nan)
                for season, mask in [('dry', dry_lag), ('wet', wet_lag)]:
                    idx = np.where(mask)[0]
                    if len(idx) == 0: continue
                    seg_preds_l = {}
                    for seg in ['valley', 'peak', 'base']:
                        key = f"{seg}_{season}"
                        if key in m2_lag:
                            m_l = m2_lag[key]
                            model_l = m_l['model'] if isinstance(m_l, dict) else m_l
                            seg_preds_l[seg] = model_l.predict(X_s2_lag[idx])
                        else:
                            seg_preds_l[seg] = np.zeros(len(idx))
                    w_l = blend_weights(period_lag[idx])
                    blended_l = (w_l[:, 0] * seg_preds_l['valley'] +
                                w_l[:, 1] * seg_preds_l['peak'] +
                                w_l[:, 2] * seg_preds_l['base'])
                    price_pred_lag[idx] = anchor_l[idx] + blended_l  # residual + anchor

                nan_price_l = np.isnan(price_pred_lag).sum()
                if nan_price_l > 0:
                    price_pred_lag[np.isnan(price_pred_lag)] = 0.0
                logger.info(f"  [inference] Lag192 价格预测: mean={np.mean(price_pred_lag):.1f}")

                # Write gap-fill predictions for ALL dates beyond normal coverage.
                # forward_extend=192 adds 2 extra days of predictions.
                if n_lag >= 96:
                    # Normal covers up to dt_arr[-1] + 24h. Lag extends further.
                    last_normal_date = (pd.Timestamp(dt_arr[-1]) +
                                       pd.Timedelta(hours=24)).strftime('%Y-%m-%d')

                    # Collect all dates in lag predictions beyond normal horizon
                    engine2 = create_engine(url, echo=False)
                    existing_dates = set()
                    with engine2.connect() as conn:
                        rows = conn.execute(
                            text("SELECT DISTINCT target_time::date FROM predictions")
                        ).fetchall()
                        existing_dates = {str(r[0]) for r in rows}
                    engine2.dispose()

                    # Extract lag predictions as 96-period days
                    lag_target_dates = {}
                    for i in range(n_lag):
                        ts = pd.Timestamp(dt_arr_lag[i])
                        td = (ts + pd.Timedelta(hours=24)).strftime('%Y-%m-%d')
                        if td not in lag_target_dates:
                            lag_target_dates[td] = np.full(96, np.nan)
                        p = int(period_lag[i])
                        if 0 <= p < 96:
                            lag_target_dates[td][p] = price_pred_lag[i]

                    records_written = 0
                    for td, preds in sorted(lag_target_dates.items()):
                        if td in existing_dates or td <= last_normal_date:
                            continue  # already covered by normal pass
                        # Fill NaN periods
                        preds[np.isnan(preds)] = 0.0
                        records_lag = []
                        for p in range(96):
                            records_lag.append({
                                'target_time': pd.Timestamp(f'{td} {p//4:02d}:{(p%4)*15:02d}:00'),
                                'predicted_price': float(preds[p]),
                                'model_version': settings.feature_version + '_lag192',
                                'season': 'dry' if pd.Timestamp(td).month <= 4 else 'wet',
                                'period': p,
                            })
                        engine3 = create_engine(url, echo=False)
                        with engine3.connect() as conn:
                            with conn.begin():
                                conn.execute(
                                    text("""INSERT INTO predictions
                                        (target_time, predicted_price, model_version, season, period)
                                        SELECT :target_time, :predicted_price, :model_version, :season, :period
                                        WHERE NOT EXISTS (
                                            SELECT 1 FROM predictions
                                            WHERE target_time = :target_time
                                              AND model_version = :model_version
                                        )"""),
                                    records_lag,
                                )
                                conn.commit()
                        engine3.dispose()
                        records_written += len(records_lag)
                        logger.info(f"  [inference] Lag192 gap-fill: {td} ({len(records_lag)} periods)")

                    if records_written > 0:
                        logger.info(f"  [inference] Lag192 total: {records_written} gap-fill predictions")
                logger.info(f"  [inference] Gap-fill pass done ({time.time()-t_lag:.1f}s)")
            else:
                logger.warning(f"  [inference] Lag192 models insufficient (S1={len(m1_lag)}, S2={len(m2_lag)})")
        except Exception as e:
            logger.warning(f"  [inference] Gap-fill pass failed (non-fatal): {e}")

        elapsed = time.time() - start
        logger.info(f"JOB: daily_inference 完成 ({elapsed:.1f}s)")
        return {'ok': True, 'elapsed': round(elapsed, 1),
                'avg_price': round(float(np.mean(price_pred[-96:])), 1)}

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"JOB: daily_inference 失败 — {e}", exc_info=True)
        return {'ok': False, 'error': str(e), 'elapsed': round(elapsed, 1)}


def job_validate_data() -> dict:
    """校验昨日数据质量 → data_quality_log 表。

    每日 09:30 执行：检查 96 时段完整性、值域合理性、气象覆盖率。
    """
    logger.info("=" * 50)
    logger.info("JOB: validate_data — 校验昨日数据质量")
    start = time.time()
    try:
        from ingestion.validator import validate_date
        yesterday = date.today() - timedelta(days=1)
        result = validate_date(target_date=yesterday)
        elapsed = time.time() - start
        logger.info(f"JOB: validate_data 完成 — 等级={result['status'].upper()} ({elapsed:.1f}s)")
        return {
            'ok': True, 'status': result['status'],
            'completeness': result['completeness']['completeness_pct'],
            'elapsed': round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"JOB: validate_data 失败 — {e}", exc_info=True)
        return {'ok': False, 'error': str(e), 'elapsed': round(elapsed, 1)}


def job_weekly_retrain() -> dict:
    """全量重训练（Stage1 + Stage2）→ 更新模型文件。

    每周日 02:00 执行：数据积累一周后，在系统负载最低时重训练。
    内存需求 ~4GB（加载全量特征 + 训练 14 个模型）。

    当前对接 Phase 3 完成的 train_stage1.py / train_stage2.py。
    """
    logger.info("=" * 50)
    logger.info("JOB: weekly_retrain — 全量重训练")
    start = time.time()
    try:
        # Stage1（4 变量 × 枯/丰 = 8 模型）
        logger.info("  [retrain] Stage1: solar/hydro/wind/load × dry/wet")
        from pipeline.train_stage1 import train_and_save
        train_and_save()

        # Stage2（3 时段 × 枯/丰 = 6 模型）
        logger.info("  [retrain] Stage2: valley/peak/base × dry/wet")
        from pipeline.train_stage2 import train_and_save as train_stage2
        train_stage2()

        elapsed = time.time() - start
        logger.info(f"JOB: weekly_retrain 完成 ({elapsed:.1f}s)")
        return {'ok': True, 'elapsed': round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"JOB: weekly_retrain 失败 — {e}", exc_info=True)
        return {'ok': False, 'error': str(e), 'elapsed': round(elapsed, 1)}


def job_hourly_health() -> dict:
    """每小时健康心跳：确认调度器进程存活。"""
    db_host = settings.database_url.split('@')[1].split('/')[0] if '@' in settings.database_url else '?'
    logger.info(f"HEARTBEAT: alive | db={db_host} | fv={settings.feature_version}")
    return {'ok': True, 'timestamp': datetime.now().isoformat()}


# ====================================================================
# 任务注册表
# ====================================================================

# 名称 → 函数
JOB_REGISTRY = {
    'refresh_token':              job_refresh_token,
    'fetch_weather':              job_fetch_weather,
    'fetch_grid':                 job_fetch_grid,
    'daily_inference':            job_daily_inference,
    'validate_data':              job_validate_data,
    'weekly_retrain':             job_weekly_retrain,
    'hourly_health':              job_hourly_health,
    'refresh_token_and_fetch':    job_refresh_token_and_fetch,
}

# 名称 → APScheduler cron 配置（Asia/Shanghai = 北京时间）
JOB_SCHEDULES = {
    'refresh_token':              {'cron': '55 0 * * *',    'timezone': 'Asia/Shanghai'},
    'fetch_grid':                 {'cron': '0 1 * * *',     'timezone': 'Asia/Shanghai'},
    'fetch_weather':              {'cron': '30 1 * * *',    'timezone': 'Asia/Shanghai'},
    'daily_inference':            {'cron': '0 2 * * *',     'timezone': 'Asia/Shanghai'},
    'validate_data':              {'cron': '30 2 * * *',    'timezone': 'Asia/Shanghai'},
    'weekly_retrain':             {'cron': '0 3 * * 0',     'timezone': 'Asia/Shanghai'},
    'hourly_health':              {'cron': '0 * * * *',     'timezone': 'Asia/Shanghai'},
    'refresh_token_and_fetch':    {'cron': '0 12 * * *',    'timezone': 'Asia/Shanghai'},
}


# ====================================================================
# 调度器创建与任务注册
# ====================================================================

def create_scheduler():
    """创建 BackgroundScheduler 实例。

    使用 BackgroundScheduler（非 AsyncIOScheduler），因为：
      - 调度器在独立容器中运行（不与 FastAPI 共享事件循环）
      - pandas/numpy/XGBoost 操作是同步的，不需要 async
      - 线程池可防止训练/推理任务互相阻塞
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.jobstores.memory import MemoryJobStore

    jobstores = {'default': MemoryJobStore()}
    executors = {
        'default': ThreadPoolExecutor(max_workers=2),  # 最多 2 个任务并发
    }
    job_defaults = {
        'coalesce': True,            # 合并错过的触发
        'max_instances': 1,          # 同一任务最多 1 个运行实例
        'misfire_grace_time': 300,   # 5 分钟内错过的仍执行
    }

    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone='Asia/Shanghai',
    )
    return scheduler


def register_all_jobs(scheduler) -> int:
    """将所有任务注册到调度器。

    Returns
    -------
    int : 注册的任务数
    """
    from apscheduler.triggers.cron import CronTrigger
    registered = 0
    for job_name, job_func in JOB_REGISTRY.items():
        schedule = JOB_SCHEDULES.get(job_name, {})
        cron_expr = schedule.pop('cron', None)
        scheduler.add_job(
            job_func,
            trigger=CronTrigger.from_crontab(cron_expr, **schedule),
            id=job_name,
            name=job_name,
            replace_existing=True,
        )
        registered += 1
        logger.info(f"  注册: {job_name:20s} → {cron_expr}")
    return registered


# ====================================================================
# 运行模式
# ====================================================================

def run_scheduler():
    """启动调度器并阻塞运行（生产模式）。

    注册全部定时任务后进入无限循环，直到收到 SIGTERM/SIGINT。
    """
    scheduler = create_scheduler()

    logger.info("=" * 60)
    logger.info("  EM Prediction Scheduler — 启动中...")
    logger.info(f"  时区: Asia/Shanghai (CST = UTC+8)")
    logger.info(f"  特征版本: {settings.feature_version}")
    logger.info("=" * 60)

    n = register_all_jobs(scheduler)
    logger.info(f"  共注册 {n} 个定时任务")
    logger.info("=" * 60)

    scheduler.start()

    # 打印首次触发时间
    for job in scheduler.get_jobs():
        logger.info(f"  {job.name:20s} → 下次: {job.next_run_time}")

    # 注册优雅关闭
    def shutdown(signum, frame):
        logger.info(f"\n  收到信号 {signum}，关闭调度器...")
        scheduler.shutdown(wait=False)
        logger.info("  调度器已停止。")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logger.info("  调度器运行中... (Ctrl+C 停止)")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        shutdown(None, None)


def run_once(job_name: str = None):
    """执行一次任务后退出（测试/手动触发模式）。

    Parameters
    ----------
    job_name : str or None
        任务名，None = 按依赖顺序执行全部（天气→电网→推理→校验）
    """
    logger.info("=" * 60)
    logger.info("  EM Prediction Scheduler — 单次执行模式")
    logger.info("=" * 60)

    if job_name:
        if job_name not in JOB_REGISTRY:
            logger.error(f"未知任务: '{job_name}'")
            logger.info(f"可用: {', '.join(JOB_REGISTRY.keys())}")
            sys.exit(1)

        logger.info(f"  执行: {job_name}")
        result = JOB_REGISTRY[job_name]()
        logger.info(f"  结果: {result}")
    else:
        # 按依赖顺序执行：天气预报 → 电网数据 → 推理 → 校验
        ordered = ['refresh_token', 'fetch_grid', 'fetch_weather', 'daily_inference', 'validate_data']
        results = {}
        for name in ordered:
            logger.info(f"\n{'─' * 40}")
            logger.info(f"  执行: {name}")
            results[name] = JOB_REGISTRY[name]()
            summary = 'OK' if results[name].get('ok') else f"FAIL: {results[name].get('error', '?')}"
            logger.info(f"  结果: {summary}")

        # 汇总
        logger.info(f"\n{'=' * 60}")
        logger.info("  单次执行完成:")
        for name, r in results.items():
            icon = "✅" if r.get('ok') else "❌"
            logger.info(f"    {icon} {name}: {r}")


def main():
    import pandas as pd  # 确保 pandas 可用

    parser = argparse.ArgumentParser(
        description="EM Prediction Service — 定时任务调度器"
    )
    parser.add_argument('--once', action='store_true',
                        help='执行一次所有日常任务后退出')
    parser.add_argument('--job', type=str, default=None,
                        help=f'执行指定任务: {", ".join(JOB_REGISTRY.keys())}')
    parser.add_argument('--list', action='store_true',
                        help='列出所有注册的任务')
    args = parser.parse_args()

    if args.list:
        print("注册的定时任务 (Asia/Shanghai):")
        for name in JOB_REGISTRY:
            sched = JOB_SCHEDULES.get(name, {})
            print(f"  {name:20s} → {sched.get('cron', 'N/A')}")
        return

    if args.once or args.job:
        run_once(args.job)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
