# EM Prediction Service — 项目全貌

> 四川电力现货市场电价预测服务（生产版）
> 15 分钟分辨率 × 96 时段/天 × 24 小时前向预测
> 最后更新：2026-06-29

---

## 1. 项目定位

| | EM_Pre3（研究项目） | em_prediction_service（本项目） |
|---|---|---|
| 路径 | `G:\JAVA_Internship\EM_Pre3\` | `G:\JAVA_Internship\em_prediction_service\` |
| 用途 | 特征分析、策略实验、模型选型 | **生产推理服务** + 定时数据拉取 + 自动重训练 |
| 运行方式 | 手动执行 Python 脚本 | Docker Compose（API + Scheduler + DB） |
| 数据源 | 本地 .xlsx / .npz 文件 | PostgreSQL（grid_data / weather_obs / weather_forecast） |
| 模型 | 实验产物（.npz OOF） | 28 个生产 .pkl 模型（14 Normal + 14 Lag_192） |

**关系**：EM_Pre3 是"实验室"，em_prediction_service 是"工厂"。策略在 EM_Pre3 验证后，移植到 em_prediction_service 的生产训练管线中。

---

## 2. 目录结构

```
em_prediction_service/
├── PROJECT.md                          ← 本文件
├── README.md                           ← 项目说明（旧版，部分过时）
├── CLAUDE.md                           ← Claude Code 指导文件
├── config.py                           ← 全局配置（DB URL、模型路径、特征版本）
├── docker-compose.yml                  ← Docker 编排
├── Dockerfile.api                      ← API 容器
├── Dockerfile.scheduler                ← 调度器容器
├── requirements.txt                    ← Python 依赖
│
├── api/                                # FastAPI 服务
│   ├── main.py                         ← 应用入口（启动加载、预计算、路由）
│   └── schemas.py                      ← Pydantic 响应模型
│
├── pipeline/                           # 核心推理 + 训练管线
│   ├── data_loader.py                  ← DB → DataFrame + numpy 数组
│   ├── feature_engine.py               ← 185 维特征构建（DB 适配版）
│   ├── inference.py                    ← 完整推理链（Stage1 + Stage2）
│   ├── train_stage1.py                 ← Stage1 训练（8 Normal + 8 Lag_192 = 16 模型）
│   ├── train_stage2.py                 ← Stage2 训练（6 Normal + 6 Lag_192 = 12 模型）
│   ├── output.py                       ← 特征 npz 持久化
│   └── output/                         # 生成物
│       ├── features_15min_dry.npz      ← 枯水期特征矩阵
│       ├── features_15min_wet.npz      ← 丰水期特征矩阵
│       ├── price_oof.npz               ← Stage2 OOF 预测
│       ├── solar_oof.npz               ← Stage1 光伏 OOF
│       ├── hydro_oof.npz               ← Stage1 水电 OOF
│       ├── wind_oof.npz                ← Stage1 风电 OOF
│       ├── load_oof.npz                ← Stage1 负荷 OOF
│       ├── reports/                    ← 训练报告（.md）
│       └── charts/                     ← 评估图表（.png）
│
├── models/                             # 生产模型文件
│   ├── stage1_solar_dry.pkl            ← Stage1 光伏 枯水
│   ├── stage1_solar_wet.pkl            ← Stage1 光伏 丰水
│   ├── stage1_hydro_dry.pkl            ← Stage1 水电 枯水
│   ├── stage1_hydro_wet.pkl            ← Stage1 水电 丰水
│   ├── stage1_wind_dry.pkl             ← Stage1 风电 枯水
│   ├── stage1_wind_wet.pkl             ← Stage1 风电 丰水
│   ├── stage1_load_dry.pkl             ← Stage1 负荷 枯水
│   ├── stage1_load_wet.pkl             ← Stage1 负荷 丰水
│   ├── price_valley_dry.pkl            ← Stage2 午谷 枯水（RF）
│   ├── price_peak_dry.pkl              ← Stage2 晚峰 枯水（RF）
│   ├── price_base_dry.pkl              ← Stage2 基荷 枯水（RF）
│   ├── price_valley_wet.pkl            ← Stage2 午谷 丰水（XGB）
│   ├── price_peak_wet.pkl              ← Stage2 晚峰 丰水（XGB）
│   └── price_base_wet.pkl              ← Stage2 基荷 丰水（XGB）
│
├── ingestion/                          # 数据拉取
│   ├── grid_fetcher.py                 ← 电网数据 API 拉取
│   ├── weather_fetcher.py              ← 气象预报/实况拉取（Open-Meteo ECMWF）
│   ├── auth_login.py                   ← 电网平台登录认证
│   └── validator.py                    ← 数据质量校验
│
├── scheduler/                          # 定时任务
│   └── main.py                         ← APScheduler 调度器（8 个 Job）
│
└── db/                                 # 数据库
    └── schema.sql                      ← DDL（grid_data / weather_obs / weather_forecast / predictions / model_versions）
```

---

## 3. 数据架构

### 3.1 数据源

| 数据 | 来源 | 表名 | 粒度 | 列数 |
|------|------|------|------|------|
| 电网运行数据 | 四川电力交易中心 API（每日披露） | `grid_data` | 15min | 13 |
| 气象实况 | Open-Meteo ECMWF archive API | `weather_obs` | hourly → 15min（ffill） | 114 JSONB key |
| 气象预报 | Open-Meteo ECMWF forecast API | `weather_forecast` | hourly → 15min（ffill） | 114 JSONB key |
| 预测结果 | 模型推理产出 | `predictions` | 15min | 6 |
| 模型版本 | 训练后自动写入 | `model_versions` | — | 元数据 JSONB |

### 3.2 天气数据：19 子节点 × 6 变量 = 114 维

```
子节点: 成都, 德阳, 绵阳, 广元, 雅安, 乐山, 宜宾, 泸州, 南充, 达州,
        巴中, 甘孜, 阿坝, 凉山, 攀枝花, 自贡, 内江, 遂宁, 广安

变量:   temp(温度), rh(湿度), precip(降水), cloud(云量), wind(风速), rad(辐射)
```

原始数据以 JSONB 格式存储在 `weather_obs.variables` / `weather_forecast.variables` 中。`data_loader._expand_weather()` 将其展开为 114 列。特征工程阶段按 7 个气象聚类做等权平均，降维至 42 列集群平均。

### 3.3 季节划分

| 季节 | 月份 | 天数 | 特征 |
|------|------|------|------|
| 枯水期 (dry) | 1–4 月 | ~117 天 | 水库调度主导，电价温和 |
| 丰水期 (wet) | 5–6 月 | ~37 天 | 来水充沛，水电过剩 → 负电价风险 |

> 丰水期仅 37 天训练数据是当前最大瓶颈。

### 3.4 安全滞后约束

预测在 D-1 09:00 提交，数据可见性截止到 D-1 08:45。因此：
- 所有特征最小滞后 = **lag_96**（24 小时前同时段）
- `[t]` 当前值不可用（对 ~60% 预测点不可见）
- 天气 PP（Perfect-Prog）假设：t+96 时刻天气已知（模拟完美 NWP 预报）

---

## 4. 模型架构：两阶段预测管道 (28 模型 = 14 Normal + 14 Lag_192)

```
┌──────────────────────────────────────────────────────────┐
│                    Stage 1 (4 变量)                       │
│                                                          │
│  185 维特征 ─┬─→ XGBoost 光伏模型 → 光伏[t+96]            │
│              ├─→ XGBoost 水电模型 → 水电[t+96]            │
│              ├─→ XGBoost 风电模型 → 风电[t+96]            │
│              └─→ XGBoost 负荷模型 → 负荷[t+96]            │
│                                                          │
│  每变量 × 枯水/丰水 = 8 Normal + 8 Lag_192 = 16 模型      │
│  策略: 光伏/风电=直接绝对值, 水电/负荷=lag_96残差          │
├──────────────────────────────────────────────────────────┤
│                    Stage 2 (电价)                         │
│                                                          │
│  4 OOF + 79 安全特征 + 6 交互 = 89 维                      │
│         │                                                │
│         ├─→ 午谷模型 (period 36-67)                       │
│         ├─→ 晚峰模型 (period 68-87)  ← 余弦软融合          │
│         └─→ 基荷模型 (period 0-35, 88-95)                 │
│                                                          │
│  每时段 × 枯水/丰水 = 6 Normal + 6 Lag_192 = 12 模型      │
│  枯水: RandomForest 预测残差 → price = anchor + resid      │
│  丰水: XGBoost 预测残差 → price = anchor + resid (v14)     │
│  anchor = (lag96 + lag672) / 2                            │
└──────────────────────────────────────────────────────────┘
```

### 4.1 Stage1 细节 (Normal)

| 变量 | 特征数 | 策略 | 枯水 Test R² | 丰水 Test R² |
|------|--------|------|-------------|-------------|
| 光伏 (solar) | 36 | direct absolute | 0.9632 | 0.9475 |
| 水电 (hydro) | 45 | lag_96 residual | -0.6893 | -0.2498 |
| 风电 (wind) | 32 | direct absolute | -0.3045 | -0.3272 |
| 负荷 (load) | 37 | lag_96 residual | 0.3530 | 0.7055 |

### 4.1b Stage1 细节 (Lag_192: grid[t-192] → 变量[t+96])

| 变量 | 枯水 Test R² | 丰水 Test R² | 备注 |
|------|-------------|-------------|------|
| 光伏 | 0.9450 | 0.9560 | ✅ 辐射信号主导，几乎不退化 |
| 水电 | -0.7302 | -1.3011 | ⚠️ 依赖当前调度状态，2天前数据不足 |
| 风电 | 0.1929 | -0.3200 | 🔄 枯水反而更好（陈旧数据滤噪） |
| 负荷 | 0.3014 | 0.7107 | ✅ 强日周期，lag_192 可用 |

### 4.2 Stage2 细节

| 季节 | 模型 | Test MAE (元/MWh) | vs lag96 基线 |
|------|------|-------------------|--------------|
| 枯水 Normal | RF 残差 | 29.43 | ✅ -2.8 |
| 丰水 Normal | XGB 残差 | 52.04 | ✅ -6.9 |
| 枯水 Lag_192 | RF 残差 | 31.28 | ✅ -0.96 |
| 丰水 Lag_192 | XGB 残差 | 55.50 | ✅ -3.42 |

### 4.3 Lag_192 延迟容忍架构

**问题**: 电网数据延迟 1-2 天。6/30 凌晨 grid 到 6/28，Normal 只能预测 6/29。

**方案**: 训练 Lag_192 变体，将所有 `grid[t]` 特征替换为 `grid[t-192]`。推理时
用 `forward_extend=192` 生成未来 2 天的特征行（NaN grid + 天气预报），模型
用 grid[t-192] 填充 → 预测 horizon 自然延伸 2 天。

```
Normal:     grid[t]     → feature[t]     → price[t+96]
Lag_192:    grid[t-192] → feature[t]     → price[t+96]  (forward_extend=192)
```

### 4.4 模型文件格式

每个 `.pkl` 文件是一个 Python dict：

```python
# Stage1 模型
{
    'model': XGBRegressor(...),       # 实际模型对象
    'var': 'solar',
    'season': 'dry',
    'strategy': 'direct absolute',    # 推理时用来重建预测值
    'horizon': 96,
    'feat_names': ['D_rad_avg_pp', ...],
    'feat_indices': [42, 44, ...],    # 在全量特征中的列位置
}

# Stage2 模型
{
    'model': RandomForestRegressor(...),  # 或 XGBRegressor
    'season': 'dry',
    'segment': 'valley',
    'feat_names': ['oof_solar', ...],
    'safe_indices': [0, 1, 2, ...],   # Stage2 80 维中的安全特征位置
}
```

---

## 5. 特征体系（185 维，A–P 组）

| 组 | 内容 | 维度 | 安全滞后 |
|----|------|------|----------|
| A | 价格动量（lag_96/192/288/672, ma, vol, accel, max_30d） | 12 | lag_96 |
| B | 发电量（solar/wind/hydro/load 当前 + 多尺度滞后） | 14 | lag_96 |
| C | 供需（net_load, penetration, surplus, lags） | 10 | lag_96 |
| D | 天气 PP + Now（temp/rad/cloud/rain 等） | 42+ | ✓ PP |
| E | 电网市场（bidspace, ratio, nonmarket share） | 4 | lag_96 |
| F | 时间编码（period sin/cos 96-cycle, month, slots, holiday） | 14 | ✓ 确定性 |
| G | EDA 极端条件标记（月分位数阈值） | 6 | lag_96 |
| H | 滚动统计（30 天 max, rolling means） | 4 | 近似安全 |
| I | 基线（sim7d 持续性） | 1 | lag_96 |
| J | 交互（rad×period_cos, rain72h×flood 等） | 3 | ✓ PP |
| K | Stage1 专属（lag_4, 24h ma/vol, 7d diff, rain accumulation） | 19 | 不可用于推理 |
| L | D-1 晨间 8.75h 快照（mean/last/ramp/std × 4 变量） | 16 | lag_33–96 |
| M | 春节特征（days_from_sf, is_sf_window） | 2 | ✓ 确定性 |
| N | D-2 完整日曲线（peak/range/std/duration 等 × 变量） | 18 | lag_192 |
| O | D-2 时段级短滞后（lag_192/193/196/200/288 + ramp） | 16 | lag_192 |
| P | D-2 2h 窗口轨迹（mean/std/trend/range/max_step/accel） | 18 | lag_192 |

> **关键结论**：P 组零增量 → 特征工程已到天花板。真正瓶颈在数据源（无 100m 风速、无水库调度数据、丰水期仅 37 天）。

---

## 6. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 → DB 状态、模型数、数据范围 |
| GET | `/api/v1/predictions?date=YYYY-MM-DD` | 获取指定日期 96 时段预测 |
| GET | `/api/v1/predictions/latest` | 获取最新可用日期的预测 |
| GET | `/api/v1/models` | 列出所有活跃模型（28 个） |
| GET | `/api/v1/chart?date=YYYY-MM-DD` | 返回预测曲线的 PNG 图片 |
| GET | `/api/v1/predictions/history?start=...&end=...` | 历史预测 vs 实际值（含前端页面） |
| GET | `/api/v1/chart/history?start=...&end=...` | 历史对比 PNG 图表 |

### 响应示例

```json
GET /api/v1/predictions?date=2026-06-25

{
  "date": "2026-06-25",
  "model_version": "v13",
  "generated_at": "2026-06-25T02:00:05",
  "predictions": [
    {"period": 0, "time": "00:00", "price": 245.3, "segment": "base"},
    {"period": 1, "time": "00:15", "price": 243.1, "segment": "base"},
    ...
    {"period": 95, "time": "23:45", "price": 251.7, "segment": "base"}
  ],
  "summary": {
    "avg_price": 278.5,
    "peak_price": 485.2,
    "peak_period": 80,
    "valley_price": 120.3,
    "valley_period": 48
  }
}
```

### 时段定义

| 时段 | Period | 时间 | 说明 |
|------|--------|------|------|
| 基荷 (base) | 0–35 | 00:00–08:45 | 夜间低负荷 |
| 午谷 (valley) | 36–67 | 09:00–16:45 | 光伏大发，电价低谷 |
| 晚峰 (peak) | 68–87 | 17:00–21:45 | 光伏退坡+晚高峰，电价峰值 |
| 基荷 (base) | 88–95 | 22:00–23:45 | 夜间回落 |

---

## 7. 调度器（8 个定时任务）

> 时区：Asia/Shanghai (CST = UTC+8)

| 时间 | 任务 | 说明 |
|------|------|------|
| 00:55 | `refresh_token` | 刷新电网 API Token |
| 01:00 | `fetch_grid` | 拉取昨日 96 条电网数据 |
| 01:30 | `fetch_weather` | 拉取 NWP 气象预报（4 天） |
| 02:00 | `daily_inference` | 特征工程 → Stage1 → Stage2 → 写入 predictions 表 |
| 02:30 | `validate_data` | 校验昨日数据质量（完整性、值域） |
| 12:00 | `refresh_token_and_fetch` | 中午备份：刷新 Token + 补拉电网数据 |
| 每周日 03:00 | `weekly_retrain` | 全量重训练（28 个模型，~4GB 内存） |
| 每小时 | `hourly_health` | 心跳日志 |

### 调度器启动方式

```bash
python -m scheduler.main                    # 前台持续运行（生产）
python -m scheduler.main --once             # 执行一次全部日常任务后退出（测试）
python -m scheduler.main --job daily_inference  # 只执行指定任务
python -m scheduler.main --list             # 列出所有任务
```

---

## 8. 推理链路完整流程

```
┌─────────────────────────────────────────────────────────┐
│  1. 数据加载 (data_loader)                               │
│     PostgreSQL → grid_data + weather_obs/forecast        │
│     → 合并 → 气象上采样 (hourly→15min ffill)              │
│     → _add_columns (枯水期/汛期/主汛期标记)                │
└──────────────┬──────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────┐
│  2. 特征工程 (feature_engine)                            │
│     185 维构建 (A-P 组) → 时变安全筛选                    │
│     → features_15min_dry.npz + features_15min_wet.npz   │
└──────────────┬──────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────┐
│  3. Stage1 推理 (inference.predict_stage1)               │
│     dry_mask行 → 4 个枯水XGB模型   → 光伏/水电/风电/负荷   │
│     wet_mask行 → 4 个丰水XGB模型   → 光伏/水电/风电/负荷   │
│     残差策略: 模型预测残差 + lag_anchor 重建绝对值          │
└──────────────┬──────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────┐
│  4. Stage2 推理 (inference.build_stage2_features)        │
│     4 OOF + 70 安全特征 + 6 交互 = 80 维                  │
│     枯水: RF 预测残差 → price = anchor + residual          │
│     丰水: XGBoost 预测残差 → price = anchor + residual     │
│     三时段余弦软融合 → 96 个连续价格预测                    │
└──────────────┬──────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────┐
│  5. 输出                                                 │
│     API模式: 全部日期预计算 → 内存缓存 → REST响应          │
│     调度器模式: 最后96行 → predictions 表                 │
└─────────────────────────────────────────────────────────┘
```

---

## 9. 关键代码路径

### 启动时的调用链

```
api/main.py::lifespan()
  ├── DB 连接检查 (sqlalchemy)
  ├── np.load(features_15min_dry.npz) → X_full, feat_names, period, price, dry/wet_mask, dt_arr
  ├── inference.load_stage1_models() → models/ 下 8 个 .pkl
  ├── inference.load_stage2_models() → models/ 下 6 个 .pkl
  └── _precompute_predictions()
        ├── inference.predict_stage1() × 2 (dry + wet)
        │     └── 读取 model['strategy'] → 决定残差重建方式
        ├── inference.build_stage2_features()
        │     └── 从训练元数据获取 safe_indices（确保特征列对齐）
        ├── 3 时段模型预测 + blend_weights 软融合
        └── 按日期缓存 → state["predictions_cache"]
```

### 训练时的调用链

```
scheduler::job_weekly_retrain()
  ├── pipeline.train_stage1::train_and_save()
  │     ├── 加载 features_15min_{dry,wet}.npz
  │     ├── 每变量每季：XGBoost + TimeSeriesSplit(OOF) → .pkl (含 metadata)
  │     └── 保存 *_oof.npz (供 Stage2 训练)
  └── pipeline.train_stage2::train_and_save()
        ├── 加载 Stage1 OOF + 特征
        ├── 枯水：RandomForest 预测 anchor残差
        ├── 丰水：XGBoost 预测残差（v14 统一策略）
        ├── 保存 6 个 .pkl (含 safe_indices)
        └── 写入 model_versions 表
```

---

## 10. 配置（config.py）

```python
class Settings:
    # 数据库
    database_url: str          # asyncpg (async)
    database_url_sync: str     # psycopg2 (sync, 训练用)

    # 模型
    model_dir: str = "models/"
    feature_version: str = "v14"

    # 推理
    n_periods: int = 96
    horizon_steps: int = 96
    dry_season_months: tuple = (1, 2, 3, 4)
    wet_season_months: tuple = (5, 6)
```

---

## 11. 已知限制与改进方向

| 限制 | 影响 | 改进方向 |
|------|------|----------|
| 丰水期仅 37 天训练数据 | 丰水模型 R²≈0.26，泛化弱 | 积累更多丰水期数据 |
| ERA5 无 100m 风速 | 风电预测 R² 为负 | 接入含 u100m/v100m 的新 NWP |
| 无水库调度数据 | 水电预测 R² 为负 | 接入水库水位/出库流量 |
| 天气用前向填充而非插值 | 15min 精度略低 | 影响可忽略（天气缓慢变化） |
| 特征工程已到天花板（P 组零增量） | 继续做特征无收益 | 转向数据源升级 |

---

## 12. 常用命令

```bash
# 启动 API（开发）
cd em_prediction_service
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 启动调度器
python -m scheduler.main

# 单次执行推理
python -m scheduler.main --job daily_inference

# 全量重训练
python -m pipeline.train_stage1
python -m pipeline.train_stage2

# 独立推理 + 验证
python -m pipeline.inference --verify

# 从 DB 重新构建特征
python -m pipeline.feature_engine

# Docker 部署
docker compose up -d

# 查看日志
docker compose logs -f api
docker compose logs -f scheduler

# 健康检查
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/predictions/latest
```

---

## 13. 日志

所有核心模块使用 Python `logging` 标准库：

| Logger 名 | 模块 | 关键日志内容 |
|-----------|------|-------------|
| `api` | `api/main.py` | 启动耗时、DB/特征/模型加载状态、缓存日期范围及价格区间、每次 API 请求 |
| `pipeline.inference` | `pipeline/inference.py` | Stage1 逐变量统计、Stage2 逐季预测范围、缺失模型预警、NaN 填充告警 |
| `pipeline.data_loader` | `pipeline/data_loader.py` | DB 行数/日期范围、天气行数、合并后维度 |
| `scheduler` | `scheduler/main.py` | 每个 Job 的起止时间及结果、重训练进度 |

**生产环境日志格式**：
```
2026-06-26 09:00:05 [INFO] api:   Cache: 154 dates (2026-01-02 → 2026-06-24)  price range=[-45.2, 685.0]  (5.2s)
2026-06-26 09:05:00 [WARNING] pipeline.inference:   S1 hydro/wet: 4305 NaN OOF → filled with 0
2026-06-26 09:05:00 [ERROR] api:   Missing Stage2 models: ['valley_wet', 'peak_wet']
```

---

## 14. 版本历史

| 版本 | 日期 | 关键变更 |
|------|------|----------|
| v11 | 2026-05 | EM_Pre3: 混合锚点（水电 lag_672 + 风电/负荷 lag_96）+ 五层时间结构 |
| v12 | 2026-06 | 生产训练: 策略简化（水电 lag_96、风电 direct）→ 适配新天气数据 |
| v13 | 2026-06 | 生产推理: 枯水 RF 残差 + 丰水 XGB 绝对值 → Stage2 首次全面超越 lag96 基线 |
| v13+ | 2026-06 | 日志全链路覆盖 + Bug 修复（枯水期定义、空 m2 崩溃防护、静默 NaN 告警） |
| **v14 (Lag_192 + Wet Residual)** | 2026-06 | **延迟容忍架构**: 14 个 lag_192 模型 + forward_extend horizon 延伸 + API 双模型集自动切换 + 调度器 gap-fill pass + **丰水 XGB 改残差策略**（避免日曲线全同过拟合） |
