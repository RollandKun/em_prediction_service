# em_prediction_service - Historical data import (one-time: Excel → PostgreSQL)
"""
Reads v9_15min_base.xlsx (merged grid + weather, 14,784 rows x 51 cols)
and imports into grid_data + weather_obs tables.

Usage: python -m ingestion.import_historical
"""
import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import settings

# Add project root for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Column name mappings (Chinese → English) ──
GRID_COL_MAP = {
    "出清价格(元/MWh)": "price",
    "省内负荷(MW)": "load",
    "光伏(MW)": "solar",
    "风电(MW)": "wind",
    "水电(MW)": "hydro",
    "新能源总出力(MW)": "renewable_total",
    "竞价空间(MW)": "bidspace",
    "系统备用(MW)": "reserve",
    "非市场机组(MW)": "nonmarket",
    "联络线(MW)": "tieline",
    "负荷联络线(MW)": "load_tie",
    "日期类型": "day_type",
}

# Weather column regex patterns (identified by suffix in parentheses)
WEATHER_PATTERNS = [
    r"气温\(℃\)",
    r"降水\(mm/h\)",
    r"辐射\(W/m²\)",
    r"云量\(0-1\)",
    r"24h降水\(mm\)",
    r"气温24h变\(℃\)",
    r"HDD",
    r"CDD",
]


def find_weather_cols(all_columns: list[str]) -> list[str]:
    """Identify weather columns by matching known patterns."""
    weather_cols = []
    for col in all_columns:
        col_str = str(col).strip()
        for pat in WEATHER_PATTERNS:
            if re.search(pat, col_str):
                weather_cols.append(col_str)
                break
    return weather_cols


def read_base_table(xlsx_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Read v9_15min_base.xlsx, matching features_15min.py's reading logic.

    Returns (df_with_datetime_index, weather_column_names)
    """
    print(f"Reading: {xlsx_path}")
    df = pd.read_excel(xlsx_path, header=1)  # Row 2 = header
    df.columns = [str(c).strip() for c in df.columns]

    # Parse datetime
    if "datetime_bj" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime_bj"])
    else:
        raise KeyError("Column 'datetime_bj' not found — check Excel structure")

    df = df.set_index("datetime").sort_index()
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")

    weather_cols = find_weather_cols(df.columns.tolist())
    print(f"  Grid columns: {len(GRID_COL_MAP)}, Weather columns: {len(weather_cols)}")

    return df, weather_cols


def build_grid_records(df: pd.DataFrame) -> list[dict]:
    """Extract grid data rows from DataFrame."""
    records = []
    for dt, row in df.iterrows():
        rec = {"datetime": dt.to_pydatetime()}
        for cn_name, en_name in GRID_COL_MAP.items():
            if cn_name in df.columns:
                val = row[cn_name]
                if pd.isna(val):
                    rec[en_name] = None
                elif en_name == "day_type":
                    rec[en_name] = str(val).strip()
                else:
                    rec[en_name] = float(val)
        records.append(rec)
    print(f"  Grid records: {len(records)}")
    return records


def build_weather_records(df: pd.DataFrame, weather_cols: list[str]) -> list[dict]:
    """Pack weather columns into JSONB records."""
    records = []
    for dt, row in df.iterrows():
        variables = {}
        for col in weather_cols:
            val = row[col]
            if pd.isna(val):
                variables[col] = None
            elif isinstance(val, (np.floating, float)):
                variables[col] = float(val)
            elif isinstance(val, (np.integer, int)):
                variables[col] = int(val)
            else:
                variables[col] = str(val)
        records.append({
            "datetime": dt.to_pydatetime(),
            "variables": json.dumps(variables, ensure_ascii=False),
        })
    print(f"  Weather records: {len(records)}")
    return records


def import_to_db(grid_records: list[dict], weather_records: list[dict]):
    """Bulk insert into PostgreSQL (synchronous, for import speed)."""
    url = settings.database_url_sync
    engine = create_engine(url, echo=False)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        # Clear existing data
        print("Clearing existing data...")
        session.execute(text("DELETE FROM grid_data"))
        session.execute(text("DELETE FROM weather_obs"))
        session.commit()

        # Bulk insert grid data (chunked for memory safety)
        print(f"Inserting {len(grid_records)} grid records...")
        chunk_size = 1000
        for i in range(0, len(grid_records), chunk_size):
            chunk = grid_records[i:i + chunk_size]
            session.execute(
                text("""
                    INSERT INTO grid_data (datetime, price, load, solar, wind, hydro,
                        renewable_total, bidspace, reserve, nonmarket, tieline, load_tie, day_type)
                    VALUES (:datetime, :price, :load, :solar, :wind, :hydro,
                        :renewable_total, :bidspace, :reserve, :nonmarket, :tieline, :load_tie, :day_type)
                """),
                chunk,
            )
            if (i + chunk_size) % 5000 == 0:
                print(f"  {min(i + chunk_size, len(grid_records))}/{len(grid_records)}")
        session.commit()
        print("  Grid data committed.")

        # Bulk insert weather data
        print(f"Inserting {len(weather_records)} weather records...")
        for i in range(0, len(weather_records), chunk_size):
            chunk = weather_records[i:i + chunk_size]
            session.execute(
                text("""
                    INSERT INTO weather_obs (datetime, variables)
                    VALUES (:datetime, CAST(:variables AS jsonb))
                """),
                chunk,
            )
        session.commit()
        print("  Weather data committed.")

    engine.dispose()
    print("Import complete.")


def verify():
    """Verify imported data integrity."""
    url = settings.database_url_sync
    engine = create_engine(url, echo=False)
    with engine.connect() as conn:
        grid_count = conn.execute(text("SELECT COUNT(*) FROM grid_data")).scalar()
        weather_count = conn.execute(text("SELECT COUNT(*) FROM weather_obs")).scalar()

        print(f"\nVerification:")
        print(f"  grid_data:     {grid_count} rows (expected 14,784)")
        print(f"  weather_obs:   {weather_count} rows (expected 14,784)")

        if grid_count == 14784 and weather_count == 14784:
            print("  [OK] All data imported correctly.")
        else:
            print(f"  [WARN] Mismatch! Expected 14,784 rows each.")

        # Show sample
        sample = conn.execute(
            text("SELECT datetime, price, load, solar FROM grid_data ORDER BY datetime LIMIT 3")
        ).fetchall()
        print(f"\n  Sample grid data:")
        for row in sample:
            print(f"    {row[0]} | price={row[1]} | load={row[2]} | solar={row[3]}")

    engine.dispose()


def main():
    xlsx_path = settings.base_table_source
    if not xlsx_path or not Path(xlsx_path).exists():
        # Fallback: try relative path from EM_Pre3
        alt_path = Path(__file__).resolve().parent.parent.parent / "EM_Pre3" / "Stage1" / "Data" / "v9_15min_base.xlsx"
        if alt_path.exists():
            xlsx_path = str(alt_path)
        else:
            print(f"ERROR: Base table not found. Tried:")
            print(f"  - {settings.base_table_source}")
            print(f"  - {alt_path}")
            print("Set BASE_TABLE_SOURCE in .env or place v9_15min_base.xlsx in the expected location.")
            sys.exit(1)

    print("=" * 60)
    print("Historical Data Import: v9_15min_base.xlsx → PostgreSQL")
    print("=" * 60)

    df, weather_cols = read_base_table(xlsx_path)
    grid_records = build_grid_records(df)
    weather_records = build_weather_records(df, weather_cols)
    import_to_db(grid_records, weather_records)
    verify()


if __name__ == "__main__":
    main()
