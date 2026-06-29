# -*- coding: utf-8 -*-
"""
pipeline/data_loader.py — DB → arrays（替代旧 Excel 读取）
=========================================================
从 PostgreSQL 加载 grid_data + weather_obs / weather_forecast，
合并为与 v9_15min_base.xlsx 同构的 DataFrame + numpy 数组。

Usage:
    # 训练模式 (仅实况)
    from pipeline.data_loader import load_from_db
    dt_arr, df, solar, wind, hydro, load, price, \
        bidspace, reserve, nonmarket, tieline, load_tie = load_from_db()

    # 推理模式 (实况 + 预报)
    from pipeline.data_loader import load_for_inference
    dt_arr, df, solar, wind, hydro, load, price, \
        bidspace, reserve, nonmarket, tieline, load_tie = load_for_inference()
"""
import sys
import io
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

logger = logging.getLogger(__name__)


def _expand_weather(df_w: pd.DataFrame) -> pd.DataFrame:
    """Expand JSONB weather records to flat DataFrame (datetime + 114 cols)."""
    weather_expanded = []
    for _, row in df_w.iterrows():
        d = {"datetime": row["datetime"]}
        vars_dict = row["variables"]
        if isinstance(vars_dict, str):
            vars_dict = json.loads(vars_dict)
        d.update(vars_dict)
        weather_expanded.append(d)
    df = pd.DataFrame(weather_expanded)
    return df.fillna(np.nan)


def _upsample_weather(df: pd.DataFrame, weather_col_names: list) -> None:
    """Upsample hourly weather to 15-min in-place via forward-fill.
    Hourly → 15-min: 3 NaN gaps per hour, all filled forward (same weather
    across 4 intra-hour slots — consistent with the Perfect-Prog assumption
    that weather is slowly varying)."""
    weather_col_names_in_df = [c for c in weather_col_names if c in df.columns]
    precip_cols = [c for c in weather_col_names_in_df if c.startswith('precip_')]
    interp_cols = [c for c in weather_col_names_in_df if c not in precip_cols]
    if interp_cols:
        df[interp_cols] = df[interp_cols].ffill(limit=3)
    if precip_cols:
        df[precip_cols] = df[precip_cols].ffill(limit=4)
    df[weather_col_names_in_df] = df[weather_col_names_in_df].ffill()
    n_weather = len(weather_col_names_in_df)
    logger.debug(f"  Upsampled {n_weather} weather cols to 15-min "
                 f"(ffill={len(interp_cols)}, precip={len(precip_cols)})")


def _add_columns(df: pd.DataFrame) -> None:
    """Add datetime_bj and season columns in-place."""
    df["datetime_bj"] = df.index
    month_arr = df.index.month.values
    df["枯水期"] = ((month_arr >= 1) & (month_arr <= 4)).astype(int)  # Jan–Apr
    df["汛期"]   = ((month_arr >= 5) & (month_arr <= 6)).astype(int)   # May–Jun
    df["主汛期"] = (month_arr >= 7).astype(int)                         # Jul+


def _extract_arrays(df: pd.DataFrame):
    """Extract numpy arrays from merged DataFrame."""
    solar = df["光伏(MW)"].values.astype(float)
    wind = df["风电(MW)"].values.astype(float)
    hydro = df["水电(MW)"].values.astype(float)
    load = df["省内负荷(MW)"].values.astype(float)
    price = df["出清价格(元/MWh)"].values.astype(float)
    bidspace = df["竞价空间(MW)"].values.astype(float)
    reserve = df["系统备用(MW)"].values.astype(float)
    nonmarket = df["非市场机组(MW)"].values.astype(float)
    tieline = df["联络线(MW)"].values.astype(float)
    load_tie = df["负荷联络线(MW)"].values.astype(float)
    return (solar, wind, hydro, load, price, bidspace, reserve, nonmarket,
            tieline, load_tie)


def _load_grid(engine):
    """Load grid_data, return DataFrame with Chinese column names."""
    df_grid = pd.read_sql(
        "SELECT datetime, price, load, solar, wind, hydro, "
        "renewable_total, bidspace, reserve, nonmarket, tieline, load_tie, day_type "
        "FROM grid_data ORDER BY datetime",
        engine,
    )
    df_grid["datetime"] = pd.to_datetime(df_grid["datetime"])
    df_grid = df_grid.rename(columns={
        "price": "出清价格(元/MWh)", "load": "省内负荷(MW)",
        "solar": "光伏(MW)", "wind": "风电(MW)", "hydro": "水电(MW)",
        "renewable_total": "新能源总出力(MW)", "bidspace": "竞价空间(MW)",
        "reserve": "系统备用(MW)", "nonmarket": "非市场机组(MW)",
        "tieline": "联络线(MW)", "load_tie": "负荷联络线(MW)",
        "day_type": "日期类型",
    })
    logger.info(f"  grid_data: {len(df_grid)} rows, "
                f"{df_grid['datetime'].min()} → {df_grid['datetime'].max()}")
    return df_grid


def load_from_db():
    """Load grid_data + weather_obs from PostgreSQL (training mode).

    Returns
    -------
    dt_arr : pd.DatetimeIndex
    df : pd.DataFrame  (merged with Chinese column names)
    solar, wind, hydro, load, price, bidspace, reserve, nonmarket,
        tieline, load_tie : np.ndarray (float)
    """
    url = settings.database_url_sync
    engine = create_engine(url, echo=False)
    logger.info("Loading from PostgreSQL (training: weather_obs)...")

    # 1. Grid data
    df_grid = _load_grid(engine)

    # 2. Weather observations
    df_w = pd.read_sql(
        "SELECT datetime, variables FROM weather_obs ORDER BY datetime", engine)
    df_w["datetime"] = pd.to_datetime(df_w["datetime"])
    logger.info(f"  weather_obs: {len(df_w)} rows")
    df_weather = _expand_weather(df_w)
    weather_col_names = [c for c in df_weather.columns if c != "datetime"]

    # 3. Merge + upsample
    df = df_grid.merge(df_weather, on="datetime", how="left")
    df = df.set_index("datetime").sort_index()
    _upsample_weather(df, weather_col_names)
    _add_columns(df)

    n_weather_cols = len(df_weather.columns) - 1
    logger.info(f"  Merged: {len(df)} rows, {len(df.columns)} columns "
                f"(grid + {n_weather_cols} weather)")

    # 4. Extract
    arrays = _extract_arrays(df)
    engine.dispose()
    return (df.index, df) + arrays


def load_for_inference():
    """Load grid_data + weather_obs + weather_forecast (inference mode).

    Merges historical weather_obs with weather_forecast for the forward
    horizon, ensuring t+96 Perfect-Prog window is fully covered.
    Returns same structure as load_from_db().
    """
    url = settings.database_url_sync
    engine = create_engine(url, echo=False)
    logger.info("Loading from PostgreSQL (inference: weather_obs + weather_forecast)...")

    # 1. Grid data
    df_grid = _load_grid(engine)

    # 2. Weather: observations + forecast
    df_obs = pd.read_sql(
        "SELECT datetime, variables FROM weather_obs ORDER BY datetime", engine)
    df_obs["datetime"] = pd.to_datetime(df_obs["datetime"])
    logger.info(f"  weather_obs: {len(df_obs)} rows")

    df_fc = pd.read_sql(
        "SELECT target_time AS datetime, variables FROM weather_forecast "
        "ORDER BY target_time", engine)
    df_fc["datetime"] = pd.to_datetime(df_fc["datetime"])
    logger.info(f"  weather_forecast: {len(df_fc)} rows")

    # Expand both, concatenate (obs takes precedence where overlap)
    df_obs_flat = _expand_weather(df_obs)
    df_fc_flat = _expand_weather(df_fc)
    all_weather = pd.concat([df_obs_flat, df_fc_flat], ignore_index=True)
    all_weather = all_weather.drop_duplicates(subset="datetime", keep="first")
    all_weather = all_weather.sort_values("datetime")
    weather_col_names = [c for c in all_weather.columns if c != "datetime"]

    # 3. Merge + upsample (truncate to grid range — forecast covers t+96 via sf())
    df = df_grid.merge(all_weather, on="datetime", how="left")
    df = df.set_index("datetime").sort_index()
    # Keep only rows with valid grid data (price is the core target)
    df = df[df["出清价格(元/MWh)"].notna()].copy()
    _upsample_weather(df, weather_col_names)
    _add_columns(df)

    n_weather_cols = len(all_weather.columns) - 1
    logger.info(f"  Merged: {len(df)} rows, {len(df.columns)} columns "
                f"(grid + {n_weather_cols} weather)")

    # 4. Extract
    arrays = _extract_arrays(df)
    engine.dispose()
    return (df.index, df) + arrays
