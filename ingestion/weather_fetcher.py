# -*- coding: utf-8 -*-
"""
ingestion/weather_fetcher.py — 气象数据采集（v3：Open-Meteo 19 子节点集群平均）
===========================================================================
职责：通过 Open-Meteo API 拉取 ECMWF 气象数据（实况+预报），写入 weather_obs / weather_forecast 表。

架构升级（v3 — 集群平均）：
  - 19 个子节点覆盖 7 个区域集群，每区域 2-3 个子节点
  - 集群内等权平均消除单点噪音 → 特征引擎做两步融合
  - 19 子节点 × 6 变量（temp/rh/precip/cloud/wind/rad）→ 114 个 JSONB key
  - Open-Meteo 单次请求支持任意坐标数，API 调用次数不变
  - 小时级数据存储到 weather_obs，15 分钟插值 + 集群平均在 feature_engine 中完成
  - 带 SQLite 缓存（requests_cache）+ 自动重试（retry_requests）

7 区域集群（19 子节点）：
  【负荷】成都平原(cd_center/cd_west/cd_east) / 川东(dz_main/dz_south/dz_north) / 川南(yb_main/yb_north/yb_east)
  【光伏】甘孜(gz_main/gz_north) + 凉山(ls_south)
  【风电】凉山(ls_main/ls_west)
  【水电】雅安(ya_main/ya_west) / 攀枝花(pzh_main/pzh_north/pzh_east) + 凉山(ls_west)

API 端点：
  - 实况/历史：archive-api.open-meteo.com/v1/archive (models=ecmwf_ifs)
  - 预报：api.open-meteo.com/v1/ecmwf

使用示例：
  # 历史数据回填
  python -m ingestion.weather_fetcher --backfill 2026-01-02 2026-06-25
  # 拉取预报
  python -m ingestion.weather_fetcher --forecast
  # 拉取单日实况
  python -m ingestion.weather_fetcher --date 2026-06-15
  # 预览模式
  python -m ingestion.weather_fetcher --forecast --dry-run
"""
import sys
import io
import json
import argparse
import warnings
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Open-Meteo 专用客户端（带缓存+重试）
import requests_cache
from retry_requests import retry
import openmeteo_requests

warnings.filterwarnings("ignore")
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

# ── 缓存目录 ──
CACHE_DIR = PROJECT_ROOT / ".openmeteo_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── 天气节点/变量配置（从共享模块导入，唯一真相源）──
from shared.weather_config import NODES, VAR_PREFIXES

# ── 拉取的全部变量（顺序必须与 VAR_PREFIXES 一致！）──
HOURLY_VARS = [
    "temperature_2m",        # index 0 → temp_*
    "relative_humidity_2m",  # index 1 → rh_*
    "precipitation",         # index 2 → precip_*
    "cloud_cover",           # index 3 → cloud_*
    "wind_speed_100m",       # index 4 → wind_*
    "shortwave_radiation",   # index 5 → rad_*
]

# ── API 端点 ──
FORECAST_URL = "https://api.open-meteo.com/v1/ecmwf"           # ECMWF IFS 预报
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive" # ECMWF IFS 历史


# ====================================================================
# Open-Meteo 客户端（单例，全局复用缓存）
# ====================================================================

def _build_client() -> openmeteo_requests.Client:
    """构造带缓存和重试的 Open-Meteo 客户端。

    缓存策略：SQLite 缓存永不过期（expire_after=-1），避免重复拉取。
    重试策略：最多 5 次，指数退避 0.2s 基数。
    """
    cache_path = CACHE_DIR / "openmeteo_cache.sqlite"
    cache_session = requests_cache.CachedSession(
        str(cache_path),
        expire_after=-1,       # 永不过期，手动清缓存来控制刷新
        allowable_methods=("GET", "POST"),
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# ====================================================================
# 辅助函数
# ====================================================================

def _extract_lats_lons() -> tuple:
    """从 NODES 配置中提取经纬度列表。"""
    lats = [node["lat"] for node in NODES]
    lons = [node["lon"] for node in NODES]
    return lats, lons


def _build_time_index(hourly) -> pd.DatetimeIndex:
    """从 openmeteo Hourly 对象构造带北京时区的时间索引。

    注意：API 返回的 Time() 是 Unix 秒（UTC），需要手动转为北京时间。
    虽然请求中设了 timezone=Asia/Shanghai，但 openmeteo_requests 库返回的
    底层时间戳始终是 UTC epoch 秒，timezone 参数主要影响日聚合边界。

    Parameters
    ----------
    hourly : openmeteo_requests.Hourly
        API 响应的逐小时数据对象

    Returns
    -------
    pd.DatetimeIndex : 带 Asia/Shanghai 时区的时间索引
    """
    start_utc = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end_utc   = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    interval  = pd.Timedelta(seconds=hourly.Interval())

    time_index = pd.date_range(
        start=start_utc.tz_convert("Asia/Shanghai"),
        end=end_utc.tz_convert("Asia/Shanghai"),
        freq=interval,
        inclusive="left",
    )
    return time_index


def _build_records(responses, time_index: pd.DatetimeIndex) -> list:
    """将 Open-Meteo 响应解析为 weather_obs 记录格式。

    对每个时间步，遍历 19 子节点 × 6 个变量 = 114 个 JSONB key，
    拼装为一个 variables JSON 字典。

    Parameters
    ----------
    responses : list
        openmeteo_requests 返回的响应列表，responses[i] 对应 NODES[i]
    time_index : pd.DatetimeIndex
        时间轴（北京时间）

    Returns
    -------
    list[dict] : 每条 = {datetime, variables_json_string}
    """
    n_timesteps = len(time_index)
    records = []

    for t_idx in range(n_timesteps):
        variables = {}
        dt = time_index[t_idx]
        # 去除时区信息（存 naive datetime，PostgreSQL 自动处理）
        dt_naive = dt.tz_localize(None) if dt.tzinfo else dt

        for i, node in enumerate(NODES):
            node_name = node["name"]
            hourly = responses[i].Hourly()

            for j, prefix in enumerate(VAR_PREFIXES):
                values = hourly.Variables(j).ValuesAsNumpy()
                if t_idx < len(values):
                    val = float(values[t_idx])
                    # 跳过 NaN（如夜间辐射=0 是正常的，但真正的 NaN 不应该写）
                    if not np.isnan(val):
                        key = f"{prefix}_{node_name}"
                        variables[key] = val

        if variables:
            records.append({
                "datetime": dt_naive,
                "variables": json.dumps(variables, ensure_ascii=False),
            })

    return records


# ====================================================================
# 数据库写入
# ====================================================================

def _upsert_obs(records: list, dry_run: bool = False) -> int:
    """UPSERT 气象实况到 weather_obs 表。

    ON CONFLICT (datetime) → 整体替换 variables JSONB。
    """
    if not records:
        print("  [DB] 无气象实况需写入")
        return 0

    if dry_run:
        print(f"\n  [DB] DRY-RUN 模式 — 预览前 3 条:")
        for rec in records[:3]:
            dt = rec["datetime"]
            vars_dict = json.loads(rec["variables"])
            # 只显示前 5 个 key 作为预览
            vars_preview = dict(list(vars_dict.items())[:5])
            print(f"    {dt} | keys={len(vars_dict)} | {vars_preview}...")
        print(f"  [DB] 共 {len(records)} 条（未写入）")
        return 0

    url = settings.database_url_sync
    engine = create_engine(url, echo=False)

    sql = """
        INSERT INTO weather_obs (datetime, variables)
        VALUES (:datetime, CAST(:variables AS jsonb))
        ON CONFLICT (datetime)
        DO UPDATE SET variables = CAST(EXCLUDED.variables AS jsonb)
    """

    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text(sql), records)
        conn.commit()

    engine.dispose()
    print(f"  [DB] 写入 {len(records)} 条气象实况 (UPSERT)")
    return len(records)


def _upsert_forecast(records: list, dry_run: bool = False) -> int:
    """UPSERT 气象预报到 weather_forecast 表。

    每条 record 需同时携带 fetch_time（预报获取时间）和 target_time（预报有效时间）。
    """
    if not records:
        print("  [DB] 无预报数据需写入")
        return 0

    if dry_run:
        print(f"\n  [DB] DRY-RUN — 共 {len(records)} 条预报（未写入）")
        return 0

    url = settings.database_url_sync
    engine = create_engine(url, echo=False)

    sql = """
        INSERT INTO weather_forecast (fetch_time, target_time, variables)
        VALUES (:fetch_time, :target_time, CAST(:variables AS jsonb))
        ON CONFLICT (fetch_time, target_time)
        DO UPDATE SET variables = CAST(EXCLUDED.variables AS jsonb)
    """

    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text(sql), records)
        conn.commit()

    engine.dispose()
    print(f"  [DB] 写入 {len(records)} 条预报 (UPSERT)")
    return len(records)


# ====================================================================
# 公开接口（保持与 scheduler 调用方兼容）
# ====================================================================

def fetch_weather_obs(
    target_date: Optional[date] = None,
    dry_run: bool = False,
) -> int:
    """拉取指定日期的气象实况并写入 weather_obs。

    使用 Open-Meteo archive API (models=ecmwf_ifs) 拉取单日历史再分析数据。

    Parameters
    ----------
    target_date : date or None
        目标日期，None = 昨日
    dry_run : bool
        True 仅预览，不写入数据库

    Returns
    -------
    int : 写入的记录数
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    date_str = target_date.isoformat()
    print(f"  [WeatherFetcher] 拉取气象实况: {date_str}")

    lats, lons = _extract_lats_lons()
    client = _build_client()

    # 构造 archive API 参数
    params = {
        "latitude":   lats,
        "longitude":  lons,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly":     HOURLY_VARS,
        "models":     "ecmwf_ifs",
        "timezone":   "Asia/Shanghai",
    }

    print(f"  [Open-Meteo] 请求 archive API ({len(NODES)} 节点, {date_str})")
    responses = client.weather_api(ARCHIVE_URL, params=params)
    print(f"  [Open-Meteo] 成功 — 返回 {len(responses)} 个节点数据")

    # 构造时间轴 + 解析数据
    time_index = _build_time_index(responses[0].Hourly())
    records = _build_records(responses, time_index)

    return _upsert_obs(records, dry_run=dry_run)


def fetch_weather_forecast(
    fetch_date: Optional[date] = None,
    dry_run: bool = False,
) -> int:
    """拉取 NWP 气象预报并写入 weather_forecast。

    使用 Open-Meteo forecast API (ECMWF IFS) 拉取未来 4 天预报。

    Parameters
    ----------
    fetch_date : date or None
        预报获取时间（API 调用时间），None = 今天
    dry_run : bool
        True 仅预览，不写入数据库

    Returns
    -------
    int : 写入的记录数
    """
    if fetch_date is None:
        fetch_date = date.today()

    print(f"  [WeatherFetcher] 拉取气象预报 (fetch_date={fetch_date})")

    lats, lons = _extract_lats_lons()
    client = _build_client()

    # 构造 forecast API 参数
    params = {
        "latitude":      lats,
        "longitude":     lons,
        "hourly":        HOURLY_VARS,
        "forecast_days": 4,          # 未来 4 天，覆盖 t+96 窗口
        "timezone":      "Asia/Shanghai",
    }

    print(f"  [Open-Meteo] 请求 forecast API ({len(NODES)} 节点)")
    responses = client.weather_api(FORECAST_URL, params=params)
    print(f"  [Open-Meteo] 成功 — 返回 {len(responses)} 个节点数据")

    # 构造时间轴 + 解析数据
    time_index = _build_time_index(responses[0].Hourly())
    records = _build_records(responses, time_index)

    # 转换为 forecast 记录格式（需要 fetch_time + target_time）
    # fetch_time 设为当日 01:30（与调度时间一致）
    fetch_dt = datetime(fetch_date.year, fetch_date.month, fetch_date.day, 1, 30, 0)
    forecast_records = []
    for rec in records:
        forecast_records.append({
            "fetch_time": fetch_dt,
            "target_time": rec["datetime"],
            "variables": rec["variables"],
        })

    return _upsert_forecast(forecast_records, dry_run=dry_run)


def backfill_obs(
    start_date: date,
    end_date: date,
    dry_run: bool = False,
) -> int:
    """从 Open-Meteo archive API 回填历史气象实况。

    逐日（按周分批）拉取 start_date 到 end_date（含）的历史再分析数据，
    UPSERT 到 weather_obs 表。使用 requests_cache 避免重复请求。

    Parameters
    ----------
    start_date : date
        回填起始日期（含）
    end_date : date
        回填结束日期（含）
    dry_run : bool
        True 仅预览，不写入

    Returns
    -------
    int : 写入的总记录数
    """
    print(f"\n  [Backfill] 历史数据回填: {start_date} → {end_date}")
    delta = (end_date - start_date).days + 1
    print(f"  [Backfill] 共 {delta} 天")

    lats, lons = _extract_lats_lons()
    client = _build_client()

    total_written = 0
    current = start_date
    chunk_size = 7  # 每次拉取 7 天，避免单次请求太大

    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_size - 1), end_date)
        start_str = current.isoformat()
        end_str   = chunk_end.isoformat()

        print(f"\n  [Backfill] 拉取 {start_str} ~ {end_str}...")

        params = {
            "latitude":   lats,
            "longitude":  lons,
            "start_date": start_str,
            "end_date":   end_str,
            "hourly":     HOURLY_VARS,
            "models":     "ecmwf_ifs",
            "timezone":   "Asia/Shanghai",
        }

        responses = client.weather_api(ARCHIVE_URL, params=params)
        print(f"  [Backfill] 返回 {len(responses)} 个节点数据")

        time_index = _build_time_index(responses[0].Hourly())
        records = _build_records(responses, time_index)

        n = _upsert_obs(records, dry_run=dry_run)
        total_written += n

        current = chunk_end + timedelta(days=1)

    print(f"\n  [Backfill] 完成 — 共写入 {total_written} 条")
    return total_written


# ====================================================================
# CLI 入口
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="气象数据采集 — Open-Meteo ECMWF API（v3 19 子节点集群平均）"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="目标日期 YYYY-MM-DD，实况默认昨日"
    )
    parser.add_argument(
        "--forecast", action="store_true",
        help="拉取 NWP 预报（默认拉取实况）"
    )
    parser.add_argument(
        "--backfill", nargs=2, metavar=("START", "END"),
        help="历史数据回填: --backfill 2026-01-02 2026-06-22"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览，不写入数据库"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Weather Fetcher v3 — Open-Meteo ECMWF 19 子节点 / 7 集群平均")
    print("=" * 60)

    try:
        if args.backfill:
            # ── 历史回填模式 ──
            start = datetime.strptime(args.backfill[0], "%Y-%m-%d").date()
            end   = datetime.strptime(args.backfill[1], "%Y-%m-%d").date()
            n = backfill_obs(start, end, dry_run=args.dry_run)

        elif args.forecast:
            # ── 预报模式 ──
            target = None
            if args.date:
                target = datetime.strptime(args.date, "%Y-%m-%d").date()
            n = fetch_weather_forecast(fetch_date=target, dry_run=args.dry_run)

        else:
            # ── 实况模式 ──
            target = None
            if args.date:
                target = datetime.strptime(args.date, "%Y-%m-%d").date()
            n = fetch_weather_obs(target_date=target, dry_run=args.dry_run)

        print(f"\n  完成：{n} 条记录")

    except Exception as e:
        print(f"\n  [ERROR] 气象数据采集失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
