# 电价预测模型生产化 — 架构设计与实施计划

> 日期: 2026-06-23 | 状态: 待确认

---

## 一、技术选型

| 层 | 技术 | 理由 |
|---|---|---|
| 语言 | **Python 3.11+** | 现有 ML 管线全 Python，无需重写 |
| Web 框架 | **FastAPI** | 异步、自动 OpenAPI 文档、Pydantic 验证 |
| ORM | **SQLAlchemy 2.0 + asyncpg** | 异步 PostgreSQL 驱动 |
| 调度 | **APScheduler** | 轻量进程内调度，无需 Celery/Redis |
| 数据库 | **PostgreSQL 15** | Time-series 友好，JSONB 存指标 |
| 部署 | **Docker Compose** | Python app + PostgreSQL 一键启动 |
| ML 推理 | **现有 XGBoost/sklearn .pkl** | 直接 pickle 加载，零改写 |

---

## 二、项目结构（独立项目）

```
em_prediction_service/          # 独立项目，与 EM_Pre3 同级
├── docker-compose.yml          # api + scheduler + db 三容器
├── Dockerfile.api              # FastAPI 镜像
├── Dockerfile.scheduler        # 调度器镜像（共用代码，不同入口）
├── requirements.txt
├── requirements_lock.txt       # pip freeze 精确版本锁死
├── .env
│
├── config.py                   # 全局配置（Pydantic Settings）
├── database.py                 # SQLAlchemy async engine + ORM models
│
├── ingestion/                  # 数据采集层
│   ├── grid_fetcher.py         #   电网数据爬取（价格/负荷/出力）
│   ├── weather_fetcher.py      #   气象预报 API 调用（NWP）
│   ├── validator.py            #   数据校验（完整性/异常值/延迟）
│   └── import_historical.py    #   一次性从现有 Excel 导入历史数据
│
├── pipeline/                   # ML 管线（复用现有逻辑）
│   ├── feature_engine.py       #   特征工程（从 DB 读 → numpy → 167 维）
│   ├── train_stage1.py         #   Stage1 训练（光伏/水电/风电/负荷）
│   ├── train_stage2.py         #   Stage2 训练（枯水 RF + 丰水 XGB）
│   └── inference.py            #   推理（加载 .pkl → 特征 → 预测）
│
├── api/                        # FastAPI 接口
│   ├── main.py                 #   FastAPI app（纯 HTTP，无调度器）
│   ├── routes.py               #   /api/v1/predictions, /api/v1/models, /health
│   └── schemas.py              #   Pydantic 请求/响应模型
│
├── scheduler/                  # 独立容器入口
│   ├── main.py                 #   调度器进程入口
│   └── jobs.py                 #   定时任务定义
│
├── models/                     #   训练好的 .pkl 文件存放
│
└── tests/
    ├── test_feature_engine.py
    ├── test_inference.py
    └── test_api.py
```

**项目位置**：`G:\JAVA_Internship\em_prediction_service\`（与 EM_Pre3 同级）

---

## 三、数据库设计

### 3.1 核心表（7 张）

```sql
-- 电网实况数据（15 分钟粒度，爬虫写入）
CREATE TABLE grid_data (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMPTZ NOT NULL UNIQUE,
    price NUMERIC(10,2),              -- 出清价格 元/MWh
    load NUMERIC(10,2),               -- 省内负荷 MW
    solar NUMERIC(10,2),              -- 光伏出力 MW
    wind NUMERIC(10,2),               -- 风电出力 MW
    hydro NUMERIC(10,2),              -- 水电出力 MW
    renewable_total NUMERIC(10,2),
    bidspace NUMERIC(10,2),           -- 竞价空间 MW
    reserve NUMERIC(10,2),            -- 系统备用 MW
    nonmarket NUMERIC(10,2),          -- 非市场机组 MW
    tieline NUMERIC(10,2),
    load_tie NUMERIC(10,2),
    day_type VARCHAR(20),             -- 工作日/周末/节假日
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 气象预报数据（天气 API 写入，JSONB 模式）
-- 每行 = 一个时间点的全部气象变量，避免 EAV 的 PIVOT 开销
CREATE TABLE weather_forecast (
    id BIGSERIAL PRIMARY KEY,
    fetch_time TIMESTAMPTZ NOT NULL,   -- API 调用时间
    target_time TIMESTAMPTZ NOT NULL,  -- 预报有效时间
    variables JSONB NOT NULL,          -- {"temp_chengdu":22.5,"rad_panxi":680,...}
    UNIQUE(fetch_time, target_time)
);
CREATE INDEX idx_wf_target ON weather_forecast(target_time);

-- 气象实况（用于训练，同样 JSONB）
CREATE TABLE weather_obs (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMPTZ NOT NULL UNIQUE,
    variables JSONB NOT NULL           -- 同上结构
);

-- 模型版本注册
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version_name VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(30) NOT NULL,   -- stage1_solar/stage2_dry_valley/...
    file_path VARCHAR(500),
    metrics JSONB,                     -- {"R2": 0.96, "MAE": 310}
    status VARCHAR(20) DEFAULT 'shadow'
        CHECK (status IN ('active', 'shadow', 'archived')),
    feature_cache BYTEA,               -- 缓存的特征矩阵（序列化 numpy）
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 数据质量日志（validator 写入，health 端点读取）
CREATE TABLE data_quality_log (
    id BIGSERIAL PRIMARY KEY,
    check_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL,       -- ok / warning / critical
    completeness_pct NUMERIC(5,2),     -- 96 时段中有效占比
    anomaly_count INTEGER DEFAULT 0,
    details JSONB,                     -- {"frozen_price": false, "missing_periods": [...]}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Shadow 预测（A/B 测试用，与生产预测隔离）
CREATE TABLE shadow_predictions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    target_time TIMESTAMPTZ NOT NULL,
    predicted_price NUMERIC(10,2),
    model_version VARCHAR(50),
    season VARCHAR(10),
    period INTEGER
);
CREATE INDEX idx_shadow_version ON shadow_predictions(model_version, target_time);

-- 预测记录
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    target_time TIMESTAMPTZ NOT NULL,  -- 预测的目标时间 (t+96)
    predicted_price NUMERIC(10,2),
    actual_price NUMERIC(10,2),        -- 实况填充（事后）
    model_version VARCHAR(50),
    season VARCHAR(10),
    period INTEGER                     -- 0-95
);
CREATE INDEX idx_pred_target ON predictions(target_time);
```

### 3.2 设计说明

- **weather_forecast/obs 用 JSONB**：每行一个时间点，全部气象变量打在一个 JSON 字段里。读取时 `pd.json_normalize()` 一步转宽表，无 EAV 的 PIVOT 性能开销。同时保留了灵活性——新增变量无需 ALTER TABLE
- **grid_data 为宽表**：列固定（11 个数值列），不会再变，宽表查询效率最高
- **model_versions.metrics 用 JSONB**：不同模型类型的指标不同（Stage1 有 R²/MAE，Stage2 有分时段 MAE）

### 3.3 表分区（防时间序列性能退化）

`grid_data` 和 `weather_obs` 按月范围分区，避免数据积累后全表扫描：

```sql
CREATE TABLE grid_data (
    ...
) PARTITION BY RANGE (datetime);

CREATE TABLE grid_data_2026_01 PARTITION OF grid_data
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE grid_data_2026_02 PARTITION OF grid_data
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
-- ... 按月创建，可脚本自动化
```

特征工程查询时 PostgreSQL 会自动分区裁剪——只扫描相关月份分区。

---

## 四、数据采集层

### 4.1 电网爬虫 (`ingestion/grid_fetcher.py`)

```
接口: fetch_grid_data(date: date) -> list[GridRecord]
```

- 从电网数据平台获取指定日期的 96 条 15 分钟数据
- 包含：价格、负荷、光伏、风电、水电、竞价空间、备用、非市场、联络线
- **具体实现取决于数据源**：如果官方提供 API → httpx 调用；如果是网页 → BeautifulSoup/playwright；如果是定期发布的 Excel → pandas 读取
- 初期可对接现有的 `运行数据披露-.xlsx`，后续切换为 API

### 4.2 气象 API (`ingestion/weather_fetcher.py`)

```
接口: fetch_weather_forecast() -> list[WeatherRecord]
```

- 调用 NWP 商业 API，获取未来 24-72 小时预报（覆盖 t+96 窗口）
- 变量至少包括：温度、辐射、云量、降水、100m 风速
- 接口设计为抽象基类，具体 provider 可插拔

### 4.3 数据校验与质量预警 (`ingestion/validator.py`)

```
接口: validate_date(date: date) -> ValidationReport
```

校验分三级，失败逐级升级：

**Level 1 — 完整性**
- 每日必须有 96 个时段（00:00-23:45，缺一不可）
- 每个时段 11 个数值列必须非空
- 天气 JSONB 必须包含全部必需 key（temp/rad/cloud/precip/wind）

**Level 2 — 合理性**
- 价格 [0, 1500] 元/MWh，负荷 [0, 50000] MW
- 光伏 [0, 15000] MW，风电 [0, 10000] MW，水电 [0, 30000] MW
- 天气：温度 [-20, 50]℃，辐射 [0, 1500] W/m²，云量 [0, 1]

**Level 3 — 异常检测（静默数据损坏发现）**
- **恒定值检测**：连续 6 个时段值完全不变 → 疑似爬虫抓取失败返回固定值
  ```python
  def detect_frozen_data(series: np.ndarray, window: int = 6) -> bool:
      return any(np.diff(series[i:i+window]).sum() == 0 
                 for i in range(len(series) - window))
  ```
- **同比断崖**：price[t] 与 price[t-96]（昨日同时刻）差异 > 5σ → 标记异常（不拒绝，仅告警）
- **天气 API 时效性**：`fetch_time` 距当前 > 6h → 预报已过期，触发紧急重拉

**输出**：ValidationReport 写入 DB（`data_quality_log` 表），包含 `status: ok/warning/critical`。warning 级继续推理，critical 级阻断推理并触发 `POST /api/v1/predictions/compute` 拒绝请求。

**健康检查集成**：`GET /health` 返回数据新鲜度：
```json
{
  "status": "healthy",
  "db_connected": true,
  "model_loaded": true,
  "data_freshness": {
    "latest_grid_data": "2026-06-23T08:45:00",
    "latest_weather_forecast": "2026-06-23T06:00:00",
    "days_since_last_data": 0,
    "anomalies_24h": 0
  }
}
```

---

## 五、ML 管线层

### 5.1 特征工程 (`pipeline/feature_engine.py`)

**策略：适配而非重写**。`features_15min.py` 的核心逻辑已数学验证（167 维，R² 已确认），不应重写。

1. 从 DB 查询 grid_data → pandas DataFrame（替代原 `pd.read_excel()`）
2. 从 DB 查询 weather JSONB → `pd.json_normalize()` 一步转宽表
3. 构造与原 `v9_15min_base.xlsx` 列结构一致的 DataFrame
4. 复用现有的特征计算函数（A–P 组，lag/rolling/interaction）
5. 输出 numpy 数组 + 特征名称列表

关键改造点：
- **输入源**：`pd.read_excel()` → `pd.read_sql()`
- **天气读取**：JSONB → `json_normalize()` 直接展开为多列，零 PIVOT 开销
- **增量计算**：新增数据时，只计算新行的特征

### 5.2 特征缓存机制

特征工程的全量重算（167 维 × 14,784+ 行，含 7 天滑动窗口 / 30 天滚动统计）是管线中计算量最大的环节。每日推理只需新增 96 行，全量重算严重浪费。

**缓存策略**：

```
每日推理流程：
  DB中新到 96 行 grid_data
       │
       ▼
  读取缓存的特征矩阵（上次训练时预计算并存库）
       │
       ▼
  增量计算：仅计算新 96 行的特征
    - 短窗口特征 (lag_4/96/192)：直接用 DB 最近行计算
    - 长窗口特征 (lag_672)：从缓存矩阵取
    - 滚动统计 (30d max/ma)：更新 rolling buffer
    - D-2 曲线特征：从缓存矩阵取 D-2 的 96 行
       │
       ▼
  np.vstack([cached_matrix, new_96_rows]) → 完整矩阵
```

**缓存存储**：在 `model_versions` 表增加 `feature_cache BYTEA` 列，存储序列化后的 numpy 数组。训练完成后自动刷新。

**缓存失效**：
- 重训练完成后立即刷新
- 历史数据回填后刷新
- 每天推理前校验缓存行数 == DB 行数，不一致则自动全量重算

### 5.3 缺失值自动兜底 (`pipeline/feature_engine.py` — NaN Guard)

推理链路必须保证输入矩阵**绝对不出现 NaN**。仅靠 validator 告警不够——需要三级自动兜底：

```
Level 1: 前向填充 (Forward Fill)
  单点缺失 → 用最近一个有效值填充。覆盖 95% 的网络抖动。

Level 2: 同类型日历史均值
  连续多时段缺失 (>4 个) → 取同 season + 同 day_type + 同时段的历史中位数。

Level 3: 断言兜底
  特征矩阵构建完成后 assert not np.any(np.isnan(X))
  → 如果仍然 NaN，拒绝推理 + 写错误日志 + 返回 API 503。
```

天气 JSONB 字段如果 API 完全未返回（整次调用失败），则复用最近一次成功获取的预报（`ORDER BY fetch_time DESC LIMIT 1`）。

### 5.4 推理 (`pipeline/inference.py`)

完整推理链路：

```
DB → 特征矩阵 (167 维)
    → Stage1 模型预测 solar/wind/hydro/load[t+96]  (4 个 .pkl)
    → 构造 Stage2 输入 (80 维: 4 预测 + 70 safe + 6 interaction)
    → Stage2 模型预测 price[t+96]                   (6 个 .pkl, 三时段)
    → 软融合 → 96 个价格预测
```

### 5.5 训练 (`pipeline/train_stage1.py`, `train_stage2.py`)

- 从 DB 拉取全量历史数据 → 构建完整特征矩阵
- Stage1: 4 个 XGBoost，TimeSeriesSplit OOF
- Stage2: 枯水 RF 残差 + 丰水 XGB 绝对值，三时段
- 保存 .pkl → 写入 model_versions 表 → 以 shadow 身份上线
- **训练频次**：每周一次全量重训练（或数据累积 ≥ 7 天后触发）
- **初始训练**：先用 `import_historical.py` 导入现有 154 天数据，跑一次完整训练得到基线模型

**内存管理**（训练进程隔离 + 资源监控）：

```
scheduler 容器
    │
    │  ┌─ 不直接在 scheduler 进程内跑训练 ─────────┐
    │  │  训练通过 subprocess 隔离到独立 Python 进程  │
    │  └────────────────────────────────────────────┘
    │
    ├─ 1. scheduler 触发 weekly_retrain 任务
    ├─ 2. subprocess.run([sys.executable, 'train_stage1.py'])
    │      ├─ 独立进程，独立内存空间
    │      ├─ 写入 /tmp/train_{timestamp}.log
    │      └─ 完成后进程自然退出 → OS 回收全部内存
    ├─ 3. subprocess.run([sys.executable, 'train_stage2.py'])  # 串行
    ├─ 4. 校验输出指标 → shadow 注册
    └─ 5. 显式 gc.collect() + 清理临时文件
```

```python
# scheduler/jobs.py
import subprocess, resource, gc

def weekly_retrain():
    """每周全量重训练 — 子进程隔离，防 OOM 扩散"""
    try:
        # Stage1
        result = subprocess.run(
            [sys.executable, '-m', 'pipeline.train_stage1'],
            capture_output=True, text=True, timeout=3600,  # 1h 超时
            cwd=PROJECT_ROOT
        )
        if result.returncode != 0:
            raise TrainingError(f"Stage1 failed:\n{result.stderr}")
        
        gc.collect()  # 清理 Python 内存碎片
        
        # Stage2
        result = subprocess.run(
            [sys.executable, '-m', 'pipeline.train_stage2'],
            capture_output=True, text=True, timeout=1800,
            cwd=PROJECT_ROOT
        )
        ...
    except subprocess.TimeoutExpired:
        alert("Training timeout — 可能数据膨胀或死循环")
```

**资源限制**（Docker Compose scheduler 容器）：
```yaml
scheduler:
  deploy:
    resources:
      limits:
        memory: 4G      # 硬限制，防 OOM 扩散到 DB
      reservations:
        memory: 1G
```

### 5.6 增量训练策略（演进路径）

全量重训练在数据 < 1 年时完全可接受（154 天 ~10 分钟完成）。数据累积 > 1 年后可选择性切换增量模式：

**方案 A：XGBoost Warm-Start（丰水 + Stage1）**
```python
# XGBoost 支持增量加树，无需从头训练
model.fit(X_new, y_new, xgb_model=old_model.get_booster())
```
适用：丰水期 XGBoost 模型（绝对值预测）。在旧模型基础上追加 `n_estimators` 棵树拟合新数据。

**方案 B：RF 快照集成（枯水）**
```python
# sklearn RF 不支持增量学习 → 新训练 + 加权集成
new_rf = RandomForestRegressor(...).fit(X_all, y_all)
old_rf = load_model('price_valley_dry_prev.pkl')
# 推理时：pred = 0.7 * new_rf.predict(X) + 0.3 * old_rf.predict(X)
```
适用：枯水期 RF 模型。新旧模型按数据量加权集成，平滑过渡。

**切换阈值**：数据 > 400 天 且 全量训练 > 30 分钟时触发，否则保持全量重训练。

### 5.7 模型持久化：Pickle → ONNX（演进路径）

**短期**（Phase 1-5）：`requirements_lock.txt` 锁死依赖版本，确保 `.pkl` 反序列化安全。

**中长期**（Phase 5+）：训练管线增加 ONNX 导出环节，将模型运行层与 Python 业务环境彻底解耦：

```python
# XGBoost → ONNX
from onnxmltools import convert_xgboost
onnx_model = convert_xgboost(model, ...)

# sklearn RF → ONNX  
from skl2onnx import convert_sklearn
onnx_model = convert_sklearn(model, ...)
```

收益：推理不再依赖 xgboost/scikit-learn/numpy 精确版本 → Docker 镜像体积缩小 → 推理延迟降低 — ONNX Runtime 比 pickle 快 2-5×。

### 5.8 模型 A/B 测试 — Shadow Mode 安全上线

新模型上线前必须经过生产数据验证，但不能影响线上服务。Shadow Mode 实现零风险灰度测试：

```
生产流量
    │
    ▼
┌──────────────────────────┐
│  Active Model (当前版本)  │──────► API 响应返回前端
└──────────────────────────┘
    │
    │ (同一输入并行调用)
    ▼
┌──────────────────────────┐
│  Shadow Model (候选版本)  │──────► 仅写日志，不影响响应
└──────────────────────────┘
```

**实现机制**：

`model_versions` 表增加 `status` 字段：`active` | `shadow` | `archived`

```sql
ALTER TABLE model_versions ADD COLUMN status VARCHAR(20) 
    CHECK (status IN ('active', 'shadow', 'archived')) DEFAULT 'shadow';
```

推理时：active 模型结果返回 API → shadow 模型结果异步写入 `shadow_predictions` 表。

**晋级条件**（自动判定）：
- Shadow 模型连续 **7 天** MAE 低于 active 模型
- 无 critical 级数据异常日
- → 自动晋级：shadow → active，原 active → archived

**回滚条件**（自动触发）：
- 晋级后 3 天内 MAE 劣化 > 20%
- → 自动回滚：archived 模型恢复为 active

```python
# pipeline/inference.py
async def predict_with_shadow(features, period, season):
    active_model = registry.get_active(season, period)
    shadow_models = registry.get_shadows(season, period)
    
    result = active_model.predict(features)
    
    # 异步写 shadow 预测，不阻塞主响应
    for shadow in shadow_models:
        background_tasks.add_task(
            shadow_predict_and_log, shadow, features, season, period
        )
    
    return result
```

**存储**：新增 `shadow_predictions` 表（结构与 `predictions` 相同，增加 `model_version` 索引），与生产预测隔离。

---

## 六、API 层

### 6.1 端点设计

| Method | Path | 功能 |
|---|---|---|
| `GET` | `/health` | 健康检查（DB 连接、模型加载状态） |
| `GET` | `/api/v1/predictions` | 查询预测（默认返回最新 96 条） |
| `GET` | `/api/v1/predictions?date=2026-06-24` | 查询指定日期的预测 |
| `GET` | `/api/v1/predictions/range?from=...&to=...` | 查询日期范围内的预测+实况对比 |
| `GET` | `/api/v1/models` | 当前活跃模型信息（版本、指标、更新时间） |
| `POST` | `/api/v1/predictions/compute` | 触发实时推理（返回 96 条新预测） |

### 6.2 响应格式示例

```json
// GET /api/v1/predictions?date=2026-06-24
{
  "date": "2026-06-24",
  "model_version": "v14_20260623",
  "generated_at": "2026-06-23T09:05:00+08:00",
  "predictions": [
    {"period": 0, "time": "00:00", "price": 245.50, "segment": "base"},
    {"period": 1, "time": "00:15", "price": 243.20, "segment": "base"},
    ...
    {"period": 95, "time": "23:45", "price": 251.80, "segment": "base"}
  ],
  "summary": {
    "avg_price": 268.30,
    "peak_price": 412.50,
    "peak_period": 72,
    "valley_price": 152.00,
    "valley_period": 42
  }
}
```

### 6.3 设计决策

- **预计算为主**：每日 09:05 定时生成预测并存库，API 直接返回
- **实时兜底**：`POST /compute` 可随时触发实时推理
- **前端友好**：period 0-95 → 映射为 HH:MM 时间，标注 segment
- **predicted vs actual**：`/range` 端点同时返回两列，便于前端画对比图

---

## 七、调度层

### 7.1 定时任务

| 任务 | Cron | 说明 |
|---|---|---|
| `fetch_weather` | `0 8 * * *` | 每日 08:00 获取最新 NWP 预报 |
| `daily_inference` | `5 9 * * *` | 每日 09:05 执行推理 → 存 predictions 表 |
| `fetch_grid_data` | `15 9 * * *` | 每日 09:15 爬取昨日电网数据 |
| `validate_data` | `30 9 * * *` | 每日 09:30 校验昨日数据完整性 |
| `weekly_retrain` | `0 2 * * 0` | 每周日凌晨 2:00 全量重训练 |

### 7.2 防重复执行

调度器运行在**独立单进程容器**中，与 FastAPI 容器彻底分离：

```yaml
services:
  api:        # FastAPI — 可安全开多 worker
  scheduler:  # 独立容器，单进程 — 天然无重复执行问题
  db:
```

同时 APScheduler 配置 PostgreSQL 为 JobStore，即使 scheduler 容器重启也不会丢失任务状态。

---

## 八、部署

### 8.1 容器架构

```yaml
# docker-compose.yml
services:
  api:          # FastAPI — 纯 HTTP 服务，可开多 worker
    deploy:
      resources:
        limits:
          memory: 2G
  scheduler:    # 独立容器 — 单进程 APScheduler + 训练子进程隔离
    deploy:
      resources:
        limits:
          memory: 4G      # 训练峰值内存硬限制
        reservations:
          memory: 1G
  db:           # PostgreSQL 15
    deploy:
      resources:
        limits:
          memory: 2G
```

**为什么要拆分 scheduler？** APScheduler 随 FastAPI 进程启动后，如果 Uvicorn 开启多 worker，每个 worker 都会启动一个独立的调度器，导致定时任务被重复执行。同时训练任务内存峰值可能 >2GB，与 API 容器隔离防止 OOM 扩散影响前端服务。

### 8.2 环境变量 (`.env`)

```
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/em_prediction
WEATHER_API_KEY=xxx
WEATHER_API_ENDPOINT=https://...
GRID_DATA_SOURCE=...
MODEL_DIR=/app/models
```

### 8.3 依赖版本锁死

`pickle` 反序列化对 Python 版本及依赖包版本极其敏感。Dockerfile 中将使用从当前训练环境导出的精确版本清单：

```dockerfile
# 先导出训练环境依赖：pip freeze > requirements_lock.txt
# Docker 构建时严格按锁定版本安装
RUN pip install -r requirements_lock.txt
```

核心依赖版本（需与 EM_Pre3 训练环境完全一致）：
- `xgboost==2.1.x`
- `scikit-learn==1.5.x`
- `numpy==1.26.x`
- `pandas==2.2.x`

---

## 九、与现有代码的关系

```
EM_Pre3/                                # 实验分析项目（保留不动）
├── Stage1/feature/features_15min.py        ← 核心特征逻辑，pipeline/ 会复用
├── Stage1/prediction/train_*.py            ← 训练逻辑参考
├── Stage2/train_price_stage2.py            ← Stage2 训练逻辑参考
└── Stage2/output/models/*.pkl              ← 初始模型权重 (复制到新项目)

em_prediction_service/                  # 新的生产服务
├── pipeline/feature_engine.py              ← 适配 DB 输入，复用核心计算
├── pipeline/train_stage{1,2}.py            ← 移植训练逻辑，改文件 I/O 为 DB
├── pipeline/inference.py                   ← 新写（加载模型 + 前向传播）
└── models/                                 ← 初始从 EM_Pre3 复制 .pkl
```

---

## 十、实施分阶段

### Phase 1：基础设施 + 数据入库
1. 创建项目骨架：目录结构、config.py、database.py、Dockerfile、docker-compose.yml
2. 定义 ORM 模型 + 建表
3. 实现 `import_historical.py`：从现有 Excel → PostgreSQL
4. 验证：数据库中 154 天数据完整

### Phase 2：特征工程适配
1. 实现 `pipeline/feature_engine.py`：从 DB 读取 → 构造 DataFrame → 复用特征计算
2. 对比验证：新特征矩阵 vs 现有 `features_15min_dry.npz` 数值一致
3. 性能测试：全量特征构建耗时

### Phase 3：推理 API
1. 实现 `pipeline/inference.py`：加载 10 个 .pkl → 完整推理链路
2. 实现 FastAPI + `/api/v1/predictions` + schemas
3. 端到端验证：API 返回 vs `price_oof.npz` 一致

### Phase 4：爬虫 + 定时任务
1. 实现 `ingestion/` 模块（初期可对接现有 Excel 做"伪爬虫"）
2. 实现 APScheduler 任务
3. 验证：手动触发 → DB 中看到新数据

### Phase 5：训练管线
1. 移植 Stage1 + Stage2 训练逻辑到 pipeline/
2. 实现 model versioning（训练完成 → 写记录 → 设 active）
3. 验证：重训练后模型指标与现有报告一致

### Phase 6：联调 + 文档
1. 全部流程联调
2. README 编写

---

## 十一、验证方案

1. **特征等价性**：DB 读取 → feature_engine → 输出 vs 原始 `features_15min_dry.npz`，`np.allclose(rtol=1e-10)`
2. **推理等价性**：加载现有 .pkl → 对历史数据推理 → 输出 vs `price_oof.npz`，MAE 差异 < 0.01
3. **训练可复现**：用相同数据重训练 → 模型 R² 与现有报告偏差 < 0.01
4. **API 正确性**：`curl /api/v1/predictions?date=2026-06-06` → 返回 96 条合理价格

---

## 待确认

- [x] Python 作为技术栈
- [x] PostgreSQL + Docker Compose
- [x] 调度器独立容器（防多 worker 重复执行）
- [x] 表分区（防时间序列性能退化）
- [x] 版本锁死 + ONNX 演进路径
- [x] JSONB 替代 EAV（气象表）
- [x] NaN 三级自动兜底（推理链路安全）
- [x] 数据质量三级校验 + 恒值检测 + 健康检查
- [x] 特征缓存机制（增量计算，避免全量重算）
- [x] Shadow Mode A/B 测试（安全上线 + 自动晋级/回滚）
- [x] 训练子进程隔离 + 内存硬限制（防 OOM 扩散）
- [x] 增量训练演进路径（XGBoost warm-start / RF 快照集成）
- [ ] 初期爬虫可用"解析现有 Excel 文件"作为模拟数据源——真正的爬虫需要你提供电网数据平台 URL/API
- [ ] 天气 API：你打算用哪家？
- [ ] 前端需要什么数据？（只有价格预测，还是也要负荷/新能源预测？要不要历史对比图？）
