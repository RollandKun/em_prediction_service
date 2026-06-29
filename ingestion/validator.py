# -*- coding: utf-8 -*-
"""
ingestion/validator.py — 数据质量校验（Phase 4）
==================================================
职责：对每日入库的电网和气象数据进行完整性/合理性校验，结果写入 data_quality_log。

校验规则：
  1. 完整性检查：电网数据应有 96 个时段，气象数据按变量类型检查
  2. 值域检查：价格/负荷/出力等是否在合理范围内
  3. 异常值标记：连续跳变、负值异常等
  4. 气象数据检查：核心变量（温度/辐射/降水/云量）的覆盖率

质量等级：
  - ok: 96/96 时段 + 全部核心变量完整 + 无异常
  - warning: 个别时段缺失 (< 5%) 或 气象变量部分缺失
  - critical: 大量缺失 (≥ 5%) 或 价格/负荷严重异常

使用示例：
  python -m ingestion.validator                            # 校验昨日
  python -m ingestion.validator --date 2026-06-15          # 校验指定日期
  python -m ingestion.validator --date 2026-06-15 --json   # JSON 输出
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
# 校验参数（可调）
# ====================================================================

# 每个自然日应有 96 个 15 分钟时段
EXPECTED_PERIODS = 96

# 电网数据合理范围（基于四川 2026 年 1-6 月实际数据）
GRID_RANGES = {
    'price':    (-300.0, 1500.0),   # 出清价格 元/MWh（丰水期可低至 -200）
    'load':     (5000.0, 65000.0),  # 省内负荷 MW（夏峰可达 ~58k, 冬谷 ~8k）
    'solar':    (-10.0, 20000.0),   # 光伏出力 MW（允许小幅负值，传感器噪声）
    'wind':     (0.0, 15000.0),     # 风电出力 MW
    'hydro':    (0.0, 40000.0),     # 水电出力 MW
    'renewable_total': (0.0, 30000.0),  # 新能源总出力 MW
    'bidspace': (5000.0, 65000.0),  # 竞价空间 MW（夏峰可达 ~60k）
    'reserve':  (0.0, 20000.0),     # 系统备用 MW（夏峰 ~16k）
    'nonmarket':(0.0, 25000.0),     # 非市场机组 MW
    'tieline':  (-55000.0, 15000.0), # 联络线 MW（四川外送时为负，可低至 -48k）
    'load_tie': (20000.0, 120000.0), # 负荷联络线 MW（= load + tie，夏峰 ~105k）
}

# 气象核心变量（对预测最重要的变量，缺失率过高应告警）
CRITICAL_WEATHER_VARS = [
    '气温(℃)',       # 温度
    '辐射(W/m²)',    # 太阳辐射
    '降水(mm/h)',    # 降水强度
    '云量(0-1)',     # 云量
]

# 时段级跳变阈值：相邻 15 分钟的绝对变化不应超过
PRICE_JUMP_THRESHOLD = 500.0   # 价格跳变 500 元/MWh
LOAD_JUMP_THRESHOLD = 5000.0   # 负荷跳变 5000 MW
SOLAR_JUMP_THRESHOLD = 3000.0  # 光伏跳变 3000 MW


# ====================================================================
# 校验逻辑
# ====================================================================

def _load_day_data(target_date: date) -> pd.DataFrame:
    """从数据库加载指定日期的电网数据，返回 DataFrame（datetime 为索引）。"""
    url = settings.database_url_sync
    engine = create_engine(url, echo=False)

    start_ts = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
    end_ts = start_ts + timedelta(days=1)

    df = pd.read_sql(
        text("""
            SELECT datetime, price, load, solar, wind, hydro,
                   renewable_total, bidspace, reserve, nonmarket, tieline, load_tie
            FROM grid_data
            WHERE datetime >= :start AND datetime < :end
            ORDER BY datetime
        """),
        engine,
        params={'start': start_ts, 'end': end_ts},
    )
    engine.dispose()

    if len(df) > 0:
        df = df.set_index('datetime')

    return df


def _load_day_weather(target_date: date) -> dict[str, float]:
    """从数据库加载指定日期的气象实况，返回 {变量名: 缺失率} 的统计。"""
    url = settings.database_url_sync
    engine = create_engine(url, echo=False)

    start_ts = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
    end_ts = start_ts + timedelta(days=1)

    df = pd.read_sql(
        text("""
            SELECT datetime, variables
            FROM weather_obs
            WHERE datetime >= :start AND datetime < :end
            ORDER BY datetime
        """),
        engine,
        params={'start': start_ts, 'end': end_ts},
    )
    engine.dispose()

    if len(df) == 0:
        return {}  # 无气象数据

    # 展开 JSONB → 统计每个气象变量的缺失率
    var_values = {}  # {var_name: [values]}
    total_rows = len(df)

    for _, row in df.iterrows():
        vars_dict = row['variables']
        if isinstance(vars_dict, str):
            vars_dict = json.loads(vars_dict)
        if vars_dict:
            for var, val in vars_dict.items():
                if var not in var_values:
                    var_values[var] = [np.nan] * total_rows
                var_values[var].append(val if val is not None else np.nan)

    # 对每个变量补齐长度
    stats = {}
    for var_name, vals in var_values.items():
        # 补齐到 total_rows（有些变量在某些行可能未出现）
        while len(vals) < total_rows:
            vals.append(np.nan)
        vals_arr = np.array(vals[:total_rows])
        # 缺失率 = (总行数 - 有效值个数) / 总行数
        missing_rate = np.sum(np.isnan(vals_arr)) / total_rows
        stats[var_name] = round(float(missing_rate), 4)

    return stats


def _check_completeness(df: pd.DataFrame) -> dict:
    """检查 96 时段完整性。

    Returns
    -------
    dict : {'present': 96, 'missing': 0, 'completeness_pct': 100.0, 'missing_slots': [...]}
    """
    if len(df) == 0:
        return {
            'present': 0, 'missing': EXPECTED_PERIODS,
            'completeness_pct': 0.0,
            'missing_slots': list(range(EXPECTED_PERIODS)),
        }

    # 提取 15 分钟时段索引（0-95）
    times = df.index
    periods = [t.hour * 4 + t.minute // 15 for t in times]
    present = set(periods)

    all_periods = set(range(EXPECTED_PERIODS))
    missing = sorted(all_periods - present)

    return {
        'present': len(present),
        'missing': len(missing),
        'completeness_pct': round(len(present) / EXPECTED_PERIODS * 100, 2),
        'missing_slots': missing[:10],  # 最多显示 10 个缺失时段
    }


def _check_value_ranges(df: pd.DataFrame) -> dict:
    """检查各列的值域是否在合理范围内。

    Returns
    -------
    dict : {'price': {'outliers': 3, 'min': -50.0, 'max': 1200.0}, ...}
    """
    range_issues = {}
    for col, (lo, hi) in GRID_RANGES.items():
        if col not in df.columns:
            continue

        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(vals) == 0:
            range_issues[col] = {'outliers': 0, 'min': None, 'max': None, 'note': '全列为空'}
            continue

        outliers = ((vals < lo) | (vals > hi)).sum()
        range_issues[col] = {
            'outliers': int(outliers),
            'min': round(float(vals.min()), 2),
            'max': round(float(vals.max()), 2),
            'range': f"[{lo}, {hi}]",
        }

    return range_issues


def _check_jumps(df: pd.DataFrame) -> dict:
    """检查相邻时段的跳变是否异常。

    Returns
    -------
    dict : {'price_jumps': 2, 'load_jumps': 0, ...}
    """
    jump_issues = {}

    # 价格跳变
    if 'price' in df.columns:
        price_vals = pd.to_numeric(df['price'], errors='coerce').dropna().values
        if len(price_vals) > 1:
            jumps = np.abs(np.diff(price_vals))
            n_jumps = int(np.sum(jumps > PRICE_JUMP_THRESHOLD))
            jump_issues['price_jumps'] = {'count': n_jumps, 'threshold': PRICE_JUMP_THRESHOLD}
            if n_jumps > 0:
                max_jump_idx = int(np.argmax(jumps))
                jump_issues['price_jumps']['max_jump'] = round(float(jumps[max_jump_idx]), 1)
                jump_issues['price_jumps']['max_jump_at'] = f"period {max_jump_idx}→{max_jump_idx + 1}"

    # 负荷跳变
    if 'load' in df.columns:
        load_vals = pd.to_numeric(df['load'], errors='coerce').dropna().values
        if len(load_vals) > 1:
            jumps = np.abs(np.diff(load_vals))
            n_jumps = int(np.sum(jumps > LOAD_JUMP_THRESHOLD))
            jump_issues['load_jumps'] = {'count': n_jumps, 'threshold': LOAD_JUMP_THRESHOLD}

    # 光伏跳变
    if 'solar' in df.columns:
        solar_vals = pd.to_numeric(df['solar'], errors='coerce').dropna().values
        if len(solar_vals) > 1:
            jumps = np.abs(np.diff(solar_vals))
            n_jumps = int(np.sum(jumps > SOLAR_JUMP_THRESHOLD))
            jump_issues['solar_jumps'] = {'count': n_jumps, 'threshold': SOLAR_JUMP_THRESHOLD}

    return jump_issues


def _determine_status(
    completeness: dict,
    range_issues: dict,
    weather_stats: dict,
    jump_issues: dict,
) -> str:
    """综合判定数据质量等级: ok / warning / critical。

    判定规则（优先级从高到低）：
      1. 价格/负荷列 all-NaN → critical
      2. 完整性 < 90% → critical
      3. 价格/负荷越界 > 3 个时段 → warning
      4. 价格跳变 > 5 次 → warning
      5. 核心气象变量缺失率 > 50% → warning
      6. 全部通过 → ok
    """
    # Rule 1: 关键列全空
    if completeness['completeness_pct'] == 0:
        return 'critical'

    # Rule 2: 完整性严重不足
    if completeness['completeness_pct'] < 90:
        return 'critical'

    # Rule 3: 价格/负荷越界
    for col in ['price', 'load']:
        if col in range_issues:
            if range_issues[col].get('outliers', 0) > 3:
                return 'warning'

    # Rule 4: 价格跳变过多
    if 'price_jumps' in jump_issues:
        if jump_issues['price_jumps'].get('count', 0) > 5:
            return 'warning'

    # Rule 5: 核心气象变量缺失严重
    for critical_var in CRITICAL_WEATHER_VARS:
        for var_name, miss_rate in weather_stats.items():
            if critical_var in var_name and miss_rate > 0.5:
                return 'warning'

    # Rule 6: 轻微不完整但可接受
    if completeness['completeness_pct'] < 95:
        return 'warning'

    return 'ok'


# ====================================================================
# 公开接口
# ====================================================================

def validate_date(
    target_date: Optional[date] = None,
) -> dict:
    """校验指定日期的数据质量，结果写入 data_quality_log。

    Parameters
    ----------
    target_date : date or None
        目标日期，None = 昨日

    Returns
    -------
    dict : 完整校验结果，包含 status/completeness/range/jumps/weather/...
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    print(f"  [Validator] 校验日期: {target_date}")

    # ── 电网数据检查 ──
    df = _load_day_data(target_date)
    completeness = _check_completeness(df)
    range_issues = _check_value_ranges(df)
    jump_issues = _check_jumps(df)

    print(f"  [Validator] 电网: {completeness['present']}/{EXPECTED_PERIODS} 时段 "
          f"({completeness['completeness_pct']:.1f}%)")

    if completeness['missing_slots']:
        print(f"  [Validator] 缺失时段: {completeness['missing_slots']}")

    # 汇总越界统计
    total_outliers = sum(iss.get('outliers', 0) for iss in range_issues.values())
    if total_outliers > 0:
        print(f"  [Validator] 值域越界: {total_outliers} 个")
        for col, iss in sorted(range_issues.items()):
            if iss.get('outliers', 0) > 0:
                print(f"    {col}: {iss['outliers']} 个越界 "
                      f"(实际 {iss['min']}~{iss['max']}, 合理 {iss['range']})")

    # ── 气象数据检查 ──
    weather_stats = _load_day_weather(target_date)
    if weather_stats:
        # 只关注核心变量
        critical_missing = {}
        for critical_var in CRITICAL_WEATHER_VARS:
            for var_name, miss_rate in weather_stats.items():
                if critical_var in var_name:
                    critical_missing[var_name] = miss_rate
        if critical_missing:
            worst = max(critical_missing, key=critical_missing.get)
            print(f"  [Validator] 气象: {len(weather_stats)} 变量, "
                  f"核心最差 {worst} 缺失率 {critical_missing[worst]:.1%}")
    else:
        print("  [Validator] 气象: 无数据")

    # ── 综合判定 ──
    status = _determine_status(completeness, range_issues, weather_stats, jump_issues)

    # ── 写入 data_quality_log ──
    details = {
        'completeness': completeness,
        'range_issues': {k: v for k, v in range_issues.items() if v.get('outliers', 0) > 0},
        'jump_issues': {k: v for k, v in jump_issues.items() if v.get('count', 0) > 0},
        'weather_stats': {k: v for k, v in weather_stats.items()
                         if any(cv in k for cv in CRITICAL_WEATHER_VARS)},
    }

    _write_quality_log(target_date, status, completeness['completeness_pct'],
                       total_outliers, details)

    print(f"  [Validator] 质量等级: {status.upper()}")

    return {
        'date': target_date.isoformat(),
        'status': status,
        'completeness': completeness,
        'range_issues': range_issues,
        'jump_issues': jump_issues,
        'weather_stats': weather_stats,
        'total_outliers': total_outliers,
    }


def _write_quality_log(
    check_date: date, status: str, completeness_pct: float,
    anomaly_count: int, details: dict,
):
    """写入数据质量日志到 data_quality_log 表。"""
    url = settings.database_url_sync
    engine = create_engine(url, echo=False)

    with engine.connect() as conn:
        with conn.begin():
            conn.execute(
                text("""
                    INSERT INTO data_quality_log
                        (check_date, status, completeness_pct, anomaly_count, details)
                    VALUES (:date, :status, :completeness, :anomalies, CAST(:details AS jsonb))
                """),
                {
                    'date': check_date,
                    'status': status,
                    'completeness': round(float(completeness_pct), 2),
                    'anomalies': anomaly_count,
                    'details': json.dumps(details, ensure_ascii=False),
                },
            )
        conn.commit()

    engine.dispose()


def main():
    parser = argparse.ArgumentParser(
        description="数据质量校验（电网 96 时段 + 气象覆盖率 + 值域/跳变）"
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='校验日期 YYYY-MM-DD，默认昨日'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='以 JSON 格式输出结果'
    )
    args = parser.parse_args()

    target = None
    if args.date:
        try:
            target = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"错误：日期格式应为 YYYY-MM-DD，得到: {args.date}")
            sys.exit(1)

    print("=" * 60)
    print("  Data Validator — 数据质量校验")
    print("=" * 60)

    result = validate_date(target)

    if args.json:
        # 清理不可序列化的对象
        output = {k: v for k, v in result.items()}
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"\n  校验完成: {result['status'].upper()}")
        print(f"  完整率:   {result['completeness']['completeness_pct']:.1f}%")
        print(f"  越界数:   {result['total_outliers']}")


if __name__ == "__main__":
    main()
