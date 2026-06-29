# EM Prediction Service

四川电力现货市场电价预测生产服务。15 分钟分辨率，24 小时前推预测（t+96 horizon）。

## 架构

```
em_prediction_service/
├── api/                        # FastAPI 接口
│   ├── main.py                 #   App + lifespan + 4 endpoints
│   └── schemas.py              #   Pydantic models
├── ingestion/                  # 数据采集层
│   ├── grid_fetcher.py         #   电网数据采集（lingfeng-saas API）
│   ├── weather_fetcher.py      #   气象数据采集（Open-Meteo ECMWF API, 19 子节点）
│   ├── auth_login.py           #   JWT Token 自动刷新
│   ├── validator.py            #   数据质量校验 → data_quality_log
│   └── import_historical.py    #   一次性历史数据导入
├── pipeline/                   # ML 管线
│   ├── data_loader.py          #   DB → numpy 数组（替代 Excel 读取）
│   ├── feature_engine.py       #   特征工程（DB → 177 维, A-P 组, 集群平均）
│   ├── output.py               #   保存 npz + 验证 vs 参考
│   ├── train_stage1.py         #   Stage1 训练（4 变量 × 枯/丰 = 8 模型）
│   ├── train_stage2.py         #   Stage2 训练（3 时段 × 枯/丰 = 6 模型）
│   └── inference.py            #   推理（加载模型 → 96 价格预测）
├── shared/                     # 共享配置（唯一真相源）
│   └── weather_config.py       #   19 节点, 7 集群, 6 变量, 融合权重
├── scheduler/                  # 定时任务
│   └── main.py                 #   APScheduler（8 个 job）
├── config.py                   # 全局配置（Pydantic Settings）
├── database.py                 # SQLAlchemy ORM（7 表）
├── docker-compose.yml          # 3 容器：db + api + scheduler
├── Dockerfile.api
├── Dockerfile.scheduler
├── requirements.txt
└── .env
```

### 预测流程

```
电网数据 (15min) ──┐
                   ├──→ 特征工程 (177 维, A–P 组, 两阶段天气融合) ──→ Stage1 (4 XGBoost)
气象数据 (hourly) ─┘                                                    ├─ solar[t+96]
    ↑                                                                   ├─ hydro[t+96]
    Open-Meteo ECMWF (19 子节点 × 6 变量)                              ├─ wind[t+96]
    集群内等权平均 → 区域加权融合                                        └─ load[t+96]
                                                                              │
                                                          Stage2 (6 模型, 三时段) ← 4 OOF + 70 safe + 6 int
                                                                              │
                                                                    price[t+96] × 96 periods
                                                                    (valley/peak/base soft blend)
```

## 快速开始

### 前提条件

- Python 3.12+
- PostgreSQL 15 (或 Docker)

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env 填入实际路径
```

关键配置项：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `DATABASE_URL` | 异步数据库 URL (asyncpg) | `postgresql+asyncpg://em_user:em_pass_2026@localhost:5432/em_prediction` |
| `DATABASE_URL_SYNC` | 同步数据库 URL (psycopg2) | `postgresql+psycopg2://em_user:em_pass_2026@localhost:5432/em_prediction` |
| `MODEL_DIR` | 模型文件目录 | `./models` |
| `FEATURE_VERSION` | 特征版本号 | `v11` |

### 2. 数据库

```bash
# Docker 方式
docker compose up db -d

# 或本地 PostgreSQL
createdb em_prediction
```

### 3. 导入历史数据

```bash
# 电网数据（从 Excel 导入）
python -m ingestion.import_historical

# 气象实况（从 Open-Meteo archive API 回填）
python -m ingestion.weather_fetcher --backfill 2026-01-02 2026-06-25
```

### 4. 构建特征

```bash
python -m pipeline.feature_engine
# → pipeline/output/features_15min_dry.npz
# → pipeline/output/features_15min_wet.npz
```

### 5. 训练模型

```bash
# Stage1（4 变量 × 枯/丰 = 8 模型）
python -m pipeline.train_stage1
# → models/stage1_{solar,hydro,wind,load}_{dry,wet}.pkl (8 files)
# → pipeline/output/{solar,hydro,wind,load}_oof.npz (4 files)

# Stage2（3 时段 × 枯/丰 = 6 模型）
python -m pipeline.train_stage2
# → models/price_{valley,peak,base}_{dry,wet}.pkl (6 files)
# → pipeline/output/price_oof.npz
```

### 6. 导出基础表

```bash
python export_base_table.py
# → data/v11_15min_base.xlsx
```

### 7. 启动 API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 8. Docker Compose 一键启动

```bash
docker compose up -d
# → db (PostgreSQL 15)
# → api (FastAPI, port 8000)
# → scheduler (APScheduler)
```

## API 文档

启动后访问 `http://localhost:8000/docs` 查看 Swagger UI。

### 端点

| Method | Path | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/predictions?date=YYYY-MM-DD` | 查询指定日期的 96 条预测 |
| `GET` | `/api/v1/predictions/latest` | 查询最新日期的预测 |
| `GET` | `/api/v1/models` | 活跃模型清单 |

### 响应示例

**GET /health**
```json
{
  "status": "healthy",
  "db_connected": true,
  "models_loaded": 14,
  "feature_version": "v11",
  "data_date_range": "2026-01-02 → 2026-06-06"
}
```

**GET /api/v1/predictions?date=2026-06-06**
```json
{
  "date": "2026-06-06",
  "model_version": "v11",
  "generated_at": "2026-06-23T10:00:00",
  "predictions": [
    {"period": 0,  "time": "00:00", "price": 245.5, "segment": "base"},
    {"period": 1,  "time": "00:15", "price": 243.2, "segment": "base"},
    {"period": 36, "time": "09:00", "price": 152.0, "segment": "valley"},
    {"period": 72, "time": "18:00", "price": 412.5, "segment": "peak"}
  ],
  "summary": {
    "avg_price": 268.3,
    "peak_price": 412.5,
    "peak_period": 72,
    "valley_price": 152.0,
    "valley_period": 36
  }
}
```

**GET /api/v1/models**
```json
{
  "models": [
    {"version_name": "stage1_solar_dry",    "model_type": "stage1_solar",  "is_active": true},
    {"version_name": "stage2_dry_valley_v20260623_1729", "model_type": "stage2_price_valley", "is_active": true}
  ]
}
```

### Period → 时间映射

| Period | 时间 | Segment |
|---|---|---|
| 0–35 | 00:00–08:45 | base（基荷） |
| 36–67 | 09:00–16:45 | valley（午谷） |
| 68–87 | 17:00–21:45 | peak（晚峰） |
| 88–95 | 22:00–23:45 | base（基荷） |

## 定时任务

调度器独立运行（`docker compose up scheduler`），北京时间（CST）：

| 任务 | Cron | 说明 |
|---|---|---|
| `refresh_token` | `55 0 * * *` | 刷新电网 API Token |
| `fetch_grid` | `0 1 * * *` | 拉取昨日电网数据 |
| `fetch_weather` | `30 1 * * *` | 拉取 NWP 气象预报 → weather_forecast |
| `daily_inference` | `0 2 * * *` | 特征 + 推理 → predictions 表 |
| `validate_data` | `30 2 * * *` | 校验数据质量 → data_quality_log |
| `refresh_token_and_fetch` | `0 12 * * *` | 补拉电网数据（备份） |
| `weekly_retrain` | `0 3 * * 0` | 全量重训练（Stage1 + Stage2） |
| `hourly_health` | `0 * * * *` | 心跳日志 |

### 手动触发

```bash
python -m scheduler.main --once                       # 执行所有日常任务
python -m scheduler.main --job fetch_grid             # 只执行指定任务
python -m scheduler.main --list                       # 列出所有任务
```

## 数据库表

| 表 | 分辨率 | 说明 |
|---|---|---|
| `grid_data` | 15-min | 电网实况（宽表, 14 列, 按月分区） |
| `weather_obs` | hourly | 气象实况（JSONB, 114 keys: 6变量×19子节点） |
| `weather_forecast` | hourly | 气象预报（JSONB, 同 weather_obs 结构） |
| `model_versions` | — | 模型版本注册（metrics JSONB） |
| `predictions` | 15-min | 预测记录（scheduler 每日写入） |
| `data_quality_log` | daily | 数据质量日志 |
| `shadow_predictions` | 15-min | A/B 测试用（预留） |

## 天气数据架构 (v3)

### 数据源
- **API**: Open-Meteo ECMWF IFS（archive + forecast），无需 API Key
- **19 GPS 子节点** × **6 变量**（temp/rh/precip/cloud/wind/rad）= **114 JSONB key**

### 7 区域集群

| 集群 | 子节点 | 关联变量 | 用途 |
|---|---|---|---|
| 成都平原 | cd_center, cd_west, cd_east | temp, rh | 负荷（人口加权） |
| 川东城市群 | dz_main, dz_south, dz_north | temp, rh | 负荷 |
| 川南城市群 | yb_main, yb_north, yb_east | temp, rh | 负荷 |
| 凉山 | ls_main, ls_south, ls_west | rad, cloud, wind, precip | 光伏/风电/水电 |
| 甘孜 | gz_main, gz_north | rad, cloud, wind | 光伏/风电 |
| 雅安 | ya_main, ya_west | precip, wind | 水电（大渡河/岷江） |
| 攀枝花 | pzh_main, pzh_north, pzh_east | precip | 水电（金沙江/雅砻江） |

### 两阶段融合
1. **集群内等权平均**: 每集群 2-3 子节点取均值 → 消除单点噪音
2. **区域加权融合**: 按电力相关性加权 → `temp_avg`, `rad_avg`, `cloud_avg`, `wind_avg`, `precip_avg`, `rh_avg`

配置定义在 `shared/weather_config.py`（唯一真相源）。

## 开发指南

### 运行测试

```bash
# 电网数据采集（dry-run）
python -m ingestion.grid_fetcher --date 2026-06-06 --dry-run

# 气象数据采集（dry-run）
python -m ingestion.weather_fetcher --date 2026-06-06 --dry-run
python -m ingestion.weather_fetcher --forecast --dry-run

# 数据校验
python -m ingestion.validator --date 2026-06-06

# 特征等价性验证（DB vs 参考 npz）
python -m pipeline.feature_engine --verify-only

# 推理验证（vs 参考 OOF）
python -m pipeline.inference --verify
```

### 关键依赖

```
numpy==1.26.4
pandas==2.3.3
xgboost==3.2.0
scikit-learn==1.9.0
fastapi>=0.115.0
sqlalchemy>=2.0.30
apscheduler>=3.10.0
openmeteo_requests        # Open-Meteo API 客户端
requests_cache             # SQLite 缓存（避免重复 API 调用）
```

### 常见问题

**Q: 特征验证失败？**  
确认 PostgreSQL 中 `grid_data` 和 `weather_obs` 数据完整（grid 应 14,784 行，weather 应 ~4,128 行 hourly）。

**Q: API 返回 503？**  
检查 `pipeline/output/features_15min_dry.npz` 是否存在（需先运行 `feature_engine.py`），以及 `models/` 目录下是否有 14 个 `.pkl` 文件。

**Q: 调度器训练失败？**  
训练任务需要 ~4GB 内存。确认 docker-compose 中 scheduler 容器的 memory limit ≥ 4G。

**Q: 天气数据如何更新？**  
历史实况回填: `python -m ingestion.weather_fetcher --backfill <start> <end>`  
每日预报: `python -m ingestion.weather_fetcher --forecast`（scheduler 每日 01:30 自动执行）  
Open-Meteo 免费 API 无速率限制，19 节点单次请求即可覆盖。

## 与 EM_Pre3 的关系

```
EM_Pre3/                           # 实验分析项目（参考，不修改）
    ├── Stage1/feature/features_15min.py   ← 原始特征逻辑（基于 ERA5 Excel）
    ├── Stage1/prediction/train_*.py       ← 训练参考
    ├── Stage2/train_price_stage2.py       ← Stage2 训练参考
    └── Stage2/output/models/*.pkl         ← 初始模型权重（ERA5 数据）

em_prediction_service/             # 生产服务（本项目）
    ├── pipeline/feature_engine.py   ← 升级版：Open-Meteo ECMWF + 集群平均 + 177 维
    ├── pipeline/train_stage1.py     ← 移植版，适配新变量名（D_rad_avg_pp 等）
    ├── pipeline/train_stage2.py     ← 移植版 (RF+XGB, v13)
    └── models/                      ← 自训练模型（需用新数据重训练）
```

## 当前状态

| Phase | 内容 | 状态 |
|---|---|---|
| 1 | 项目骨架 + Docker + 历史数据导入 | ✅ |
| 2 | 特征工程适配（DB↔Excel 数值等价 + 集群平均升级） | ✅ |
| 3 | 推理 API（FastAPI + 14 模型） | ✅ |
| 4 | 爬虫 + 定时任务（Open-Meteo ECMWF + lingfeng-saas） | ✅ |
| 5 | 训练管线（Stage1 OOF + Stage2 + model versioning） | ✅ |
| 6 | 联调 + 文档 | ✅ |
| 7 | 天气数据升级 v3（19 子节点集群平均 + 特征解耦） | ✅ |
| 8 | 代码结构解耦（shared/ + data_loader + output 模块化） | ✅ |

### 模型指标（v13, 旧 ERA5 数据）

| 模型 | 枯水 Test R² | 丰水 Test R² | 策略 |
|---|---|---|---|
| Stage1 Solar | 0.9632 | 0.9475 | direct absolute |
| Stage1 Hydro | -0.6893 | -0.2498 | lag_672 residual + bias |
| Stage1 Wind | -0.3045 | -0.3272 | lag_96 residual |
| Stage1 Load | 0.3530 | 0.7055 | lag_96 residual + SF |
| Stage2 Price | 0.2289 (MAE=29.43) | 0.2574 (MAE=52.04) | dry RF残差 + wet XGB绝对值 |

> ⚠️ 以上指标基于旧 ERA5 数据。使用 Open-Meteo ECMWF 新数据（100m风速/湿度等）重训练后指标预期会变化。
