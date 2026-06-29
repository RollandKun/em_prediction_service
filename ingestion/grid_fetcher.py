# -*- coding: utf-8 -*-
"""
ingestion/grid_fetcher.py — 电网数据采集（Phase 4 → 真实 API 版）
=================================================================
职责：从电网数据平台 API 获取15分钟粒度的电网实况数据，写入 grid_data 表。

数据源（优先级）：
  1. 电网 API（lingfeng-saas.tradingthink.cn）— 需配置 GRID_API_TOKEN
  2. Excel 文件（运行数据披露-.xlsx）— 降级方案 / 历史导入

使用示例：
  python -m ingestion.grid_fetcher                              # 拉取昨日数据
  python -m ingestion.grid_fetcher --date 2026-06-15            # 拉取指定日期
  python -m ingestion.grid_fetcher --date 2026-06-15 --dry-run  # 预览，不入库
  python -m ingestion.grid_fetcher --source excel               # 强制使用 Excel
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

warnings.filterwarnings("ignore")
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

# ====================================================================
# 常量：API legendCode → 数据库英文字段映射
# ====================================================================
# API 返回两组数据：0 后缀 = 实时（我们用），1 后缀 = D-1 预测（跳过）
API_CODE_MAP = {
    'generateRealtimeSpotPriceU':  'price',            # 实时价格 元/MWh
    'provincialLoad0':             'load',             # 省内负荷 MW
    'tieLine0':                    'tieline',          # 联络线 MW
    'newEnergy0':                  'renewable_total',  # 新能源总出力 MW
    'windEnergy0':                 'wind',             # 风电出力 MW
    'solarEnergy0':                'solar',            # 光伏出力 MW
    'hydroPower0':                 'hydro',            # 水电出力 MW
    'biddingSpace0':               'bidspace',         # 竞价空间 MW
    'realtimeSystemPower':         'reserve',          # 系统备用 MW
    'noMarket0':                   'nonmarket',        # 非市场机组 MW
    'provincialLoadAndTieLine0':   'load_tie',        # 省内负荷&联络线 MW
}

# 每个自然日应有 96 个 15 分钟时段
PERIODS_PER_DAY = 96


# ====================================================================
# 时间工具
# ====================================================================

def _parse_api_datetime(label: str) -> pd.Timestamp:
    """解析 API 返回的时间标签 "YYMMDD HHMM" → Timestamp。

    Examples
    --------
    "260622 0015" → 2026-06-22 00:15:00+08:00
    "260622 2400" → 2026-06-23 00:00:00+08:00
    """
    parts = label.strip().split()
    if len(parts) != 2:
        raise ValueError(f"无法解析时间标签: {label}")
    yymmdd, hhmm = parts
    # 解析日期: YYMMDD → 20YY-MM-DD
    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = 2000 + yy
    # 解析时间: HHMM
    h = int(hhmm[:2])
    m = int(hhmm[2:4])
    if h == 24:
        # 24:00 → 次日 00:00
        base = pd.Timestamp(f"{year:04d}-{mm:02d}-{dd:02d}")
        return base + pd.Timedelta(days=1)
    return pd.Timestamp(f"{year:04d}-{mm:02d}-{dd:02d} {h:02d}:{m:02d}:00")


def _determine_day_type(dt: pd.Timestamp) -> str:
    """根据日期判断类型：工作日 / 周末 / 节假日。

    简化版节假日（与 EM_Pre3 一致，覆盖 2026 年）：
    - 元旦 1/1-1/3, 春节 2/15-2/22, 清明 4/4-4/6, 劳动节 5/1-5/5
    """
    m, d = dt.month, dt.day
    dow = dt.dayofweek  # 0=Mon, 6=Sun

    is_holiday = False
    if m == 1 and d <= 3:
        is_holiday = True
    elif m == 2 and 15 <= d <= 22:
        is_holiday = True
    elif m == 4 and 4 <= d <= 6:
        is_holiday = True
    elif m == 5 and 1 <= d <= 5:
        is_holiday = True

    if is_holiday:
        return '节假日'
    elif dow >= 5:
        return '周末'
    else:
        return '工作日'


# ====================================================================
# API 数据获取
# ====================================================================

def _fetch_from_api(target_date: date) -> list[dict]:
    """从电网数据平台 API 获取指定日期的电网实况数据。

    API: POST intraProvincialSpotMarketData/PXNSC/marketSupport
    请求体: {timeOrderType: 96, tradeUnitId: 5101000, startDate, endDate}
    响应:  {code: 200, data: {spotPriceEchartsDataList, biddingSpaceEchartsDataList,
                              provincialLoadEchartsDataList, tieLineEchartsDataList,
                              newEnergyEchartsDataList, windEnergyEchartsDataList,
                              solarEnergyEchartsDataList, hydroPowerEchartsDataList,
                              noMarketEchartsDataList, provincialLoadAndTieLineEchartsDataList,
                              realtimeSystemPowerEchartsDataList, ...}}

    每个 *EchartsDataList 包含多条 series (legendCode 0=实时, 1=D-1预测)。
    我们遍历所有 *EchartsDataList，按 legendCode 汇总后再按时间对齐。
    """
    url = settings.grid_api_url
    trade_unit_id = settings.grid_api_trade_unit_id
    token_file = PROJECT_ROOT / "grid_token.txt"

    def _load_token():
        """加载 token: grid_token.txt > .env > 自动登录"""
        if token_file.exists():
            t = token_file.read_text(encoding='utf-8').strip()
            if t:
                print(f"  [API] Token 来源: {token_file.name}")
                return t
        t = settings.grid_api_token
        if t:
            print(f"  [API] Token 来源: .env GRID_API_TOKEN")
            return t
        # 尝试自动登录
        if settings.grid_api_username and settings.grid_api_password:
            print("  [API] 无本地 Token，尝试自动登录...")
            from ingestion.auth_login import login
            return login()
        raise ValueError(
            "未找到电网 API Token。请以下任一方式配置：\n"
            f"  1. 创建 {token_file}，写入 JWT token\n"
            "  2. 在 .env 中设置 GRID_API_TOKEN + GRID_API_USERNAME + GRID_API_PASSWORD\n"
            "降级方案：python -m ingestion.grid_fetcher --source excel"
        )

    token = _load_token()

    def _try_relogin():
        """Token 过期时尝试重新登录，返回新 token 或抛异常"""
        if not (settings.grid_api_username and settings.grid_api_password):
            return None
        print("  [API] Token 已过期，自动重新登录...")
        from ingestion.auth_login import login
        return login()

    date_str = target_date.strftime('%Y-%m-%d')
    payload = {
        'timeOrderType': 96,
        'tradeUnitId': trade_unit_id,
        'startDate': date_str,
        'endDate': date_str,
    }
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }

    print(f"  [API] POST {url}")
    print(f"  [API] target_date={date_str}, tradeUnitId={trade_unit_id}")

    try:
        import urllib.request, urllib.error
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            new_token = _try_relogin()
            if new_token:
                # 重试一次
                headers['Authorization'] = f'Bearer {new_token}'
                req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                             headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = json.loads(resp.read().decode('utf-8'))
            else:
                raise RuntimeError(
                    f"API Token 已过期或无效 (HTTP {e.code})。\n"
                    f"请更新 {token_file} 文件中的 token，或配置登录凭证。\n"
                    "获取方式: 登录 https://lingfeng-saas.tradingthink.cn -> F12 -> 复制 Authorization"
                )
        else:
            raise RuntimeError(f"API HTTP 错误 {e.code}: {e.reason}")
    except Exception as e:
        raise RuntimeError(f"API 请求失败: {e}")

    if raw.get('code') != 200:
        if raw.get('code') in (401, 403):
            new_token = _try_relogin()
            if new_token:
                headers['Authorization'] = f'Bearer {new_token}'
                req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                             headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = json.loads(resp.read().decode('utf-8'))
            else:
                raise RuntimeError(
                    f"API 认证失败 (code={raw.get('code')})。\n"
                    f"请更新 {token_file} 文件中的 token，或配置登录凭证。"
                )
        else:
            raise RuntimeError(f"API 返回错误: code={raw.get('code')}, msg={raw.get('msg', 'N/A')}")
        # 二次检查重试后的结果
        if raw.get('code') != 200:
            raise RuntimeError(f"API 返回错误: code={raw.get('code')}, msg={raw.get('msg', 'N/A')}")

    data = raw['data']

    # ── 遍历所有 *EchartsDataList，汇总 series ──
    # API 响应结构: data 中每个 key 以 "EchartsDataList" 结尾
    echarts_lists = [k for k in data.keys() if k.endswith('EchartsDataList')]
    print(f"  [API] 发现 {len(echarts_lists)} 个数据组: {echarts_lists}")

    series_by_code = {}
    time_labels = None

    for list_key in echarts_lists:
        series_list = data[list_key]
        if not isinstance(series_list, list):
            continue

        for s in series_list:
            code = s.get('legendCode', '')
            name = s.get('legendName', '')
            y_vals = s.get('yAxisList', [])
            x_labels = s.get('xAxisYYList', [])

            # 跳过 D-1 预测数据（legendCode 含 "1" 后缀或 legendName 含 "D-1"）
            if 'D-1' in name or '预测' in name:
                continue

            series_by_code[code] = s

            # 取时间轴（第一个有效的 xAxisYYList）
            if time_labels is None and x_labels:
                time_labels = x_labels

            # 打印每个序列信息
            if y_vals:
                print(f"    [{list_key.replace('EchartsDataList',''):20s}] {code:35s} | "
                      f"{name:10s} | {len(y_vals):3d} pts | "
                      f"range=[{min(y_vals):.1f}, {max(y_vals):.1f}]")

    if time_labels is None:
        raise RuntimeError("API 响应中没有找到时间轴 (xAxisYYList)")

    print(f"  [API] 共收集 {len(series_by_code)} 个实时序列, {len(time_labels)} 个时间点")

    # ── 构建记录：按时间索引对齐所有 series ──
    records = []
    for i, label in enumerate(time_labels):
        dt = _parse_api_datetime(label)

        # 只保留目标日期的数据（24:00 会变成次日 00:00）
        if dt.date() != target_date:
            continue

        rec = {'datetime': dt.to_pydatetime()}

        # 填充各字段（按 API_CODE_MAP 映射）
        for code, db_field in API_CODE_MAP.items():
            s = series_by_code.get(code)
            if s is not None and i < len(s.get('yAxisList', [])):
                val = s['yAxisList'][i]
                rec[db_field] = float(val) if val is not None else None
            else:
                rec[db_field] = None

        # 日期类型
        rec['day_type'] = _determine_day_type(dt)

        records.append(rec)

    # 统计匹配到的字段
    matched_fields = [f for f in API_CODE_MAP.values() if records and records[0].get(f) is not None]
    print(f"  [API] 提取 {len(records)} 条记录, 匹配字段 ({len(matched_fields)}): {matched_fields}")

    # ── 过滤无效行 (load <= 0, 全部字段为 None) ──
    before = len(records)
    records = [r for r in records if r.get('load') is None or r['load'] > 0]
    if len(records) < before:
        print(f"  [API] 过滤 {before - len(records)} 行无效数据 (load <= 0)")

    return records


# ====================================================================
# Excel 降级数据获取（保留原逻辑）
# ====================================================================

def _fetch_from_excel(xlsx_path: str, target_date: date) -> list[dict]:
    """从运行数据披露-.xlsx 中提取指定日期的电网数据（降级方案）。"""
    # 原始 Excel 列名 → 数据库英文字段映射
    GRID_COL_MAP = {
        '四川全省现货价格（元/MWh）-实时价格': 'price',
        '省内负荷（MW）-实时':               'load',
        '联络线-实时':                       'tieline',
        '新能源（总出力）-实时':             'renewable_total',
        '新能源（风电）-实时':               'wind',
        '新能源（光伏）-实时':               'solar',
        '水电（含抽蓄）总出力-实时':         'hydro',
        '竞价空间（MW）-实时':               'bidspace',
        '备用信息（MW）-系统备用':           'reserve',
        '非市场机组出力-实时':               'nonmarket',
        '省内负荷&联络线-实时':              'load_tie',
    }
    COL_DATE = '日期'
    COL_TIME_POINT = '时点'

    print(f"  [Excel] 从文件读取: {Path(xlsx_path).name}")
    print(f"  [Excel] 目标日期: {target_date}")

    df_raw = pd.read_excel(xlsx_path, header=0)

    keep_cols = {k: v for k, v in GRID_COL_MAP.items() if k in df_raw.columns}
    rename_map = {k: v for k, v in keep_cols.items()}
    date_time_cols = [c for c in [COL_DATE, COL_TIME_POINT] if c in df_raw.columns]

    if COL_DATE not in df_raw.columns or COL_TIME_POINT not in df_raw.columns:
        raise KeyError(f"Excel 缺少必要列 '{COL_DATE}' 或 '{COL_TIME_POINT}'")

    df = df_raw[date_time_cols + list(keep_cols.keys())].copy()
    df = df.rename(columns=rename_map)

    # 构建 datetime
    def _build_datetime(row):
        date_str = str(row[COL_DATE]).replace('/', '-').strip()
        base_date = pd.Timestamp(date_str)
        tp = int(row[COL_TIME_POINT])
        h = tp // 100
        m = tp % 100
        if h == 24:
            base_date = base_date + pd.Timedelta(days=1)
            h = 0
        return pd.Timestamp(f"{base_date.strftime('%Y-%m-%d')} {h:02d}:{m:02d}:00")

    df['datetime'] = df.apply(_build_datetime, axis=1)
    mask = df['datetime'].dt.date == target_date
    df_day = df[mask].sort_values('datetime').copy()

    if len(df_day) == 0:
        print(f"  [Excel] 警告：日期 {target_date} 无数据")
        return []

    if 'load' in df_day.columns:
        df_day = df_day[df_day['load'] > 0].copy()

    numeric_cols = [v for v in rename_map.values() if v != 'day_type']
    for col in numeric_cols:
        if col in df_day.columns:
            df_day[col] = pd.to_numeric(df_day[col], errors='coerce')

    df_day['day_type'] = df_day['datetime'].apply(_determine_day_type)

    records = []
    insert_fields = list(rename_map.values())
    for _, row in df_day.iterrows():
        rec = {'datetime': row['datetime'].to_pydatetime()}
        for field in insert_fields:
            val = row.get(field)
            rec[field] = None if (pd.isna(val) or val is None) else float(val)
        records.append(rec)

    print(f"  [Excel] 提取 {len(records)} 条记录")
    return records


# ====================================================================
# 数据库写入（UPSERT）
# ====================================================================

def _upsert_records(records: list[dict], dry_run: bool = False) -> int:
    """将电网记录 UPSERT 到 grid_data 表。

    PostgreSQL ON CONFLICT (datetime) DO UPDATE 语义：
    - 新数据 → INSERT
    - 已有数据 → UPDATE（更新值可能已有变化）
    """
    if not records:
        print("  [DB] 无数据需写入")
        return 0

    if dry_run:
        print(f"\n  [DRY-RUN] 预览前 3 条:")
        for rec in records[:3]:
            dt = rec['datetime']
            price_val = rec.get('price', 'N/A')
            load_val = rec.get('load', 'N/A')
            print(f"    {dt} | price={price_val} | load={load_val}")
        print(f"  [DRY-RUN] 共 {len(records)} 条（未写入）")
        return 0

    url = settings.database_url_sync
    engine = create_engine(url, echo=False)

    fields = [f for f in records[0].keys() if f != 'created_at']
    placeholders = ', '.join([f':{f}' for f in fields])
    set_clause = ', '.join([
        f'{f} = EXCLUDED.{f}' for f in fields if f != 'datetime'
    ])

    sql = f"""
        INSERT INTO grid_data ({', '.join(fields)})
        VALUES ({placeholders})
        ON CONFLICT (datetime)
        DO UPDATE SET {set_clause}
    """

    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text(sql), records)
        conn.commit()

    engine.dispose()
    print(f"  [DB] UPSERT {len(records)} 条记录")
    return len(records)


# ====================================================================
# 主入口
# ====================================================================

def fetch_grid_data(
    target_date: Optional[date] = None,
    source: str = "api",
    source_path: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """拉取指定日期的电网数据并写入数据库。

    Parameters
    ----------
    target_date : date or None
        目标日期，None = 昨日
    source : str
        "api" = 电网 API（默认）, "excel" = Excel 降级方案
    source_path : str or None
        Excel 模式下的文件路径，None = 使用 .env 中的 GRID_DATA_SOURCE
    dry_run : bool
        True 仅预览，不写入

    Returns
    -------
    int : 写入/预览的记录数
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    # ── 选择数据源 ──
    if source == "excel":
        # 强制 Excel 模式
        path = source_path or settings.grid_data_source
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"电网数据源文件不存在: {path}")
        records = _fetch_from_excel(path, target_date)
    else:
        # API 模式：有 token 则用 API，否则降级到 Excel
        token_file = PROJECT_ROOT / "grid_token.txt"
        has_token = bool(settings.grid_api_token) or token_file.exists() or \
                    (settings.grid_api_username and settings.grid_api_password)
        if has_token:
            try:
                records = _fetch_from_api(target_date)
            except Exception as e:
                print(f"  [WARN] API 失败: {e}")
                # 降级到 Excel
                path = source_path or settings.grid_data_source
                if path and Path(path).exists():
                    print(f"  [INFO] 降级到 Excel 模式")
                    records = _fetch_from_excel(path, target_date)
                else:
                    raise
        else:
            # 无 token，降级到 Excel
            print("  [INFO] 未配置 Token/凭证，使用 Excel 降级方案")
            path = source_path or settings.grid_data_source
            if not path or not Path(path).exists():
                raise FileNotFoundError(
                    f"电网数据源文件不存在: {path}\n"
                    f"请配置 GRID_API_TOKEN 或 GRID_DATA_SOURCE"
                )
            records = _fetch_from_excel(path, target_date)

    return _upsert_records(records, dry_run=dry_run)


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="电网数据采集（API 优先，Excel 降级）"
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='目标日期 YYYY-MM-DD，默认昨日'
    )
    parser.add_argument(
        '--source', type=str, default='api',
        choices=['api', 'excel'],
        help='数据源: api (默认) 或 excel'
    )
    parser.add_argument(
        '--source-path', type=str, default=None,
        help='Excel 文件路径（仅 --source excel 时有效）'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='仅预览，不写入数据库'
    )
    parser.add_argument(
        '--update-token', type=str, default=None, metavar='TOKEN',
        help='更新 grid_token.txt 中的 API Token，然后退出'
    )
    args = parser.parse_args()

    # ── --update-token: 更新 token 后退出 ──
    if args.update_token:
        token_file = PROJECT_ROOT / "grid_token.txt"
        token_file.write_text(args.update_token.strip(), encoding='utf-8')
        print(f"Token 已更新 → {token_file} ({len(args.update_token)} chars)")
        return

    # 解析日期
    target = None
    if args.date:
        try:
            target = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"错误：日期格式应为 YYYY-MM-DD，得到: {args.date}")
            sys.exit(1)

    print("=" * 60)
    print("  Grid Fetcher — 电网数据采集")
    print(f"  数据源: {args.source.upper()}")
    print("=" * 60)

    try:
        n = fetch_grid_data(
            target_date=target,
            source=args.source,
            source_path=args.source_path,
            dry_run=args.dry_run,
        )
        print(f"\n  完成：{n} 条记录")
    except Exception as e:
        print(f"\n  [ERROR] 电网数据采集失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
