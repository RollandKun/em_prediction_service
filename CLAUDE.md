# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sichuan electricity spot market **price forecasting** at **15-minute resolution** (96 time steps/day), with a **24-hour-ahead horizon** (predict `price[t+96]`).

Production-grade ML service with: FastAPI serving, APScheduler orchestration, PostgreSQL storage, Docker Compose deployment.

## Directory Structure

```
em_prediction_service/
├── CLAUDE.md                          ← This file
├── README.md                          ← Project README
├── config.py                          ← Pydantic Settings (global config)
├── database.py                        ← SQLAlchemy 2.0 async ORM (7 tables)
├── docker-compose.yml                 ← 3 containers: db + api + scheduler
├── Dockerfile.api / Dockerfile.scheduler
├── requirements.txt / requirements_lock.txt
├── .env                               ← Credentials (not committed)
│
├── api/                               ← FastAPI REST layer
│   ├── main.py                        ← App + lifespan + 8 endpoints
│   └── schemas.py                     ← Pydantic response models
│
├── ingestion/                         ← Data acquisition
│   ├── weather_fetcher.py             ← Open-Meteo ECMWF API (19 sub-nodes)
│   ├── grid_fetcher.py                ← Grid platform API (lingfeng-saas)
│   ├── auth_login.py                  ← Auto JWT token refresh
│   ├── validator.py                   ← Data quality checks → data_quality_log
│   └── import_historical.py           ← One-time Excel → PostgreSQL
│
├── pipeline/                          ← ML pipeline
│   ├── data_loader.py                 ← DB → numpy arrays (PostgreSQL read)
│   ├── feature_engine.py              ← 177-dim feature matrix (A-P groups)
│   ├── output.py                      ← Save npz + verify vs reference
│   ├── train_stage1.py                ← Stage1: 4 vars × 2 seasons = 8 models
│   ├── train_stage2.py                ← Stage2: 3 periods × 2 seasons = 6 models
│   └── inference.py                   ← Full inference chain
│
├── scheduler/
│   └── main.py                        ← APScheduler (8 jobs, BackgroundScheduler)
│
├── shared/
│   ├── __init__.py
│   └── weather_config.py              ← Single truth: 19 nodes, 7 clusters, 6 vars
│
├── models/                            ← 28 .pkl model files (14 Normal + 14 Lag_192)
├── db/init_db.sql                     ← PostgreSQL schema + partitioning
├── export_base_table.py               ← Export DB → v11_15min_base.xlsx
└── doc/                               ← Design docs + reports
```

## Commands

```bash
# Weather data
python -m ingestion.weather_fetcher --backfill 2026-01-02 2026-06-25  # Historical
python -m ingestion.weather_fetcher --forecast                          # NWP forecast
python -m ingestion.weather_fetcher --date 2026-06-15                   # Single day

# Feature engineering (requires PostgreSQL)
python -m pipeline.feature_engine
# → pipeline/output/features_15min_dry.npz + features_15min_wet.npz

# Lag_192 gap-fill features (extends prediction horizon by 2 days)
python -m pipeline.feature_engine --grid-lag 192 --forward-extend 192
# → pipeline/output/features_15min_{dry,wet}_lag192.npz

# Training (requires feature npz files)
python -m pipeline.train_stage1           # 8 XGBoost models → models/stage1_*.pkl
python -m pipeline.train_stage2           # 6 price models → models/price_*.pkl
python -m pipeline.train_stage1 --grid-lag 192  # 8 lag_192 Stage1
python -m pipeline.train_stage2 --grid-lag 192  # 6 lag_192 Stage2
# Total: 28 models (14 Normal + 14 Lag_192)

# Export base table (requires PostgreSQL)
python export_base_table.py
# → EM_Pre3/Stage1/Data/v11_15min_base.xlsx

# Scheduler
python -m scheduler.main                    # Run all jobs continuously
python -m scheduler.main --job daily_inference  # Single job
python -m scheduler.main --list             # List all jobs

# API (requires model .pkl files)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Reference: original EM_Pre3 pipeline
cd G:\JAVA_Internship\EM_Pre3
python Stage1/feature/features_15min.py
python Stage1/prediction/train_solar_stage1.py
python Stage2/train_price_stage2.py
```

No linting, formatting, or test framework is configured.

## Architecture

### Two-Stage Prediction Pipeline (28 models: 14 Normal + 14 Lag_192)

```
Normal  — Stage 1: weather(t+96) + time(t+96) + grid[t] + L/N/O/P groups
                    → solar/wind/hydro/load[t+96]  (8 XGBoost models)
          Stage 2: 80-dim features (4 OOF + 70 safe + 6 interaction)
                    → price[t+96]  (3-period models + soft blend, 6 models)

Lag_192 — Same pipeline, but grid[t] replaced by grid[t-192] (2-day lag).
          Trained on shifted features. At inference, combined with
          forward_extend=192 to extend prediction horizon by 2 days.
          Used as gap-fill when grid data is delayed (t-2 constraint).
```

### Lag_192 Gap-Fill Architecture (Phase 9)

**Problem**: Grid data delayed 1-2 days. On 6/30 morning, latest grid = 6/28.
Normal model needs grid[t] for t=6/29 to predict 7/1 → impossible.

**Solution**: Train Lag_192 models that use grid[t-192] instead of grid[t].
At inference, `forward_extend=192` generates extra datetime rows beyond grid
data (NaN grid + real weather forecast). The lag_192 models fill these rows
with valid features from grid[t-192] → predicts 2 extra days.

```
6/30 morning, grid to 6/28:
├─ Normal:  grid[6/28] → feature[6/28] → predict price[6/29]  ✓
└─ Lag_192: grid[6/26] → feature[6/29] (extended) → price[6/30]  ✓
            grid[6/27] → feature[6/30] (extended) → price[7/1]   ✓
```

**Key files**:
- `pipeline/feature_engine.py`: `build_features(..., grid_lag=192)` shifts grid arrays
- `pipeline/data_loader.py`: `load_for_inference(forward_extend=192)` extends datetime
- `pipeline/inference.py`: `_load_s1_for_lag(192)`, `_load_s2_for_lag(192)` load lag models
- `api/main.py`: dual precompute (normal first, lag_192 fills gaps)
- `scheduler/main.py`: `job_daily_inference` runs both passes

### Data Flow

```
[Grid API / Excel]                          [Open-Meteo ECMWF API]
       |                                            |
       v                                            v
  grid_fetcher.py                          weather_fetcher.py
       |                                            |
       v                                            v
  grid_data table (15-min)           weather_obs table (hourly JSONB)
       |                                            |
       +────────────── data_loader.py ──────────────+
                            |
                            v
              feature_engine.py (177 dims, A-P)
                            |
                            v
       pipeline/output/features_15min_{dry,wet}.npz
                            |
              +─────────────┴──────────────+
              |                            |
              v                            v
        train_stage1.py              inference.py
        (8 XGBoost)                  (full chain)
              |                            |
              v                            v
        models/stage1_*.pkl         predictions table / API
              |
              v
        train_stage2.py
        (6 models: RF dry + XGB wet)
              |
              v
        models/price_*.pkl
```

### Feature Groups (177 dims, A–P)

| Group | Content | Dims |
|---|---|---|
| A. Price momentum | lag_96/192/288/672, chg_3d/7d, vol, ma, accel, max_30d | 12 |
| B. Generation | solar/wind/hydro/load current + multi-scale lags | 14 |
| C. Supply-demand | net_load, penetration, surplus, lags | 10 |
| D. Weather | 9 fused PP + 5 now + 6 model-specific (daytime, ramp, turbulence, etc.) | 20 |
| E. Grid market | bidspace + ratio, non-market + ratio | 4 |
| F. Time encoding | period/month sin/cos, holiday/weekend, 6 time slots | 14 |
| G. EDA flags | Extreme-condition binary flags (monthly quantile) | 6 |
| H. Rolling stats | 30-day max/ma for load/solar/wind/hydro | 4 |
| I. Baseline | sim7d (price lag_672) | 1 |
| J. Interaction | rad×period_cos, rain72h×flood, cloud_chg×temp_chg | 3 |
| K. Stage1-specific | intra-day lags, 24h/7d ma/vol/diff, diurnal range, rain accum | 19 |
| L. D-1 morning | mean/last/ramp/std × solar/hydro/wind/load | 16 |
| M. Spring Festival | days_from_sf, is_sf_window | 2 |
| N. D-2 daily curve | peak/range/std/duration/mean/total × variables | 18 |
| O. D-2 period-level | lag_192/193/196/200/288 + local ramp × wind/hydro/load | 16 |
| P. D-2 2h window | mean/std/trend/range/max_step/accel × 3var | 18 |

### Weather Configuration (shared/weather_config.py)

- **19 GPS sub-nodes** across **7 regional clusters**: Chengdu(3), Dazhou(3), Yibin(3), Liangshan(3), Ganzi(2), Yaan(2), Panzhihua(3)
- **6 variables**: temperature_2m, relative_humidity_2m, precipitation, cloud_cover, wind_speed_100m, shortwave_radiation
- **114 JSONB keys**: {prefix}_{node_name}
- **Two-step fusion**: Step 1: cluster-internal equal-weight average → Step 2: regional weighted fusion (REGION_WEIGHTS)
- Data source: Open-Meteo ECMWF API (archive + forecast)

### Season Split

| | Dry (枯水, Jan–Apr) | Wet (丰水, May–Jun) |
|---|---|---|
| Months | 1, 2, 3, 4 | 5, 6 |
| Train | Jan 2 – Mar 15 | May 1 – May 20 |
| Val | Mar 16 – Apr 7 | May 21 – May 28 |
| Test | Apr 8 – Apr 30 | May 29 – Jun 6 |

### Stage1 Prediction Strategies (v11 mixed anchors)

| Variable | Strategy | Features | Dry R² | Wet R² |
|---|---|---|---|---|
| Solar | Direct absolute | 35 solar-specific | 0.9632 | 0.9475 |
| Hydro | lag_672 residual + bias | 35 hydro-specific | -0.6893 | -0.2498 |
| Wind | lag_96 residual | 33 wind-specific | -0.3045 | -0.3272 |
| Load | lag_96 residual + SF | 37 load-specific | 0.3530 | 0.7055 |

### Stage2 Strategy (v14) + Lag_192

**Normal (grid[t] → price[t+96])**:
- **Both seasons**: predict `price[t+96] - anchor` (residual), anchor = (lag96 + lag672)/2
- **Dry season**: RandomForest 预测残差 → price = anchor + resid
- **Wet season**: XGBoost 预测残差 → price = anchor + resid (v14 改，避免日曲线全同过拟合)
- 3-period soft blend: valley (午谷) / peak (晚峰) / base (基荷)
- 89-dim input: 4 Stage1 OOF + 79 safe features + 6 interactions
- **Dry R²=0.23 MAE=29.43 | Wet R²=0.26 MAE=52.04** ✅ 均击败 lag96 基线

**Lag_192 (grid[t-192] → price[t+96])**:
- Same residual architecture, trained on grid_lag=192 features
- **Dry R²=0.14 MAE=31.28 | Wet R²=0.18 MAE=55.50** ✅ 均击败 lag96 基线
- Gap-fill 兜底: 精度下降 6-7%，但比无预测好

### Key Design Decisions

- **Perfect-Prog**: weather at t+96 treated as known (production: replace with NWP forecast from weather_forecast table)
- **Hourly-aware rolling sum** (`_hourly_rolling_sum`): prevents 4× overcount when accumulating precipitation on ffill'd 15-min data
- **Apparent temperature**: linear formula `1.07*T + 0.2*RH - 2.7` (replaced humidex)
- **Wind cubic**: `wind³` (replaced `0.6125 × wind³` wind power density)
- **Cluster averaging**: within-cluster equal-weight mean → regional weighted fusion (robustness against single-point failure)
- **Per-model feature selectors**: each Stage1 model only receives physics-relevant features (`solar_feat_cols`, etc.)
- **Feature engineering ceiling**: P组 (18 dims, 2h window stats) confirmed zero-gain — hard ceiling with current data. Biggest single breakthrough: new NWP data (100m wind)

### Database (PostgreSQL, 7 tables)

| Table | Resolution | Key Columns |
|---|---|---|
| `grid_data` | 15-min | datetime, price, load, solar, wind, hydro, bidspace, reserve, nonmarket, tieline, load_tie, day_type |
| `weather_obs` | hourly | datetime, variables (JSONB: 114 keys) |
| `weather_forecast` | hourly | fetch_time, target_time, variables (JSONB) |
| `model_versions` | metadata | version_name, model_type, metrics (JSONB), status |
| `predictions` | 15-min | target_time, predicted_price, actual_price, model_version, season, period |
| `data_quality_log` | daily | check_date, status, completeness_pct, details (JSONB) |
| `shadow_predictions` | 15-min | Same as predictions (A/B testing) |

## Paths

- **Project root**: `G:\JAVA_Internship\em_prediction_service\`
- **Config**: `config.py` (Pydantic Settings, .env)
- **Weather config**: `shared/weather_config.py` (single source of truth)
- **Weather fetcher**: `ingestion/weather_fetcher.py` (Open-Meteo ECMWF)
- **Feature engine**: `pipeline/feature_engine.py` (build_features, 177 dims)
- **Data loader**: `pipeline/data_loader.py` (load_from_db)
- **Output**: `pipeline/output.py` (save_outputs, verify)
- **Stage1 training**: `pipeline/train_stage1.py`
- **Stage2 training**: `pipeline/train_stage2.py`
- **Inference**: `pipeline/inference.py`
- **API**: `api/main.py` (FastAPI, 8 endpoints)
- **Scheduler**: `scheduler/main.py` (APScheduler, 8 jobs)
- **Base table export**: `export_base_table.py`
- **Models (28 .pkl)**: `models/` (14 Normal + 14 Lag_192)
- **Feature npz**: `pipeline/output/features_15min_{dry,wet}.npz`
- **Reference project**: `G:\JAVA_Internship\EM_Pre3\`
- **Reference V9**: `G:\JAVA_Internship\EM_Prediction2\v9\`
