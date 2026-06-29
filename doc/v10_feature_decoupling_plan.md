# 特征解耦计划：4 个物理模型专属 Builder 函数

> 状态：待实施 | 日期：2026-06-24 | 依赖：weather_fetcher v2 完成、weather_obs 历史回填完成

## 一、问题与目标

当前 `pipeline/feature_engine.py` 的 `build_features()` 生成统一特征矩阵（当前 171 维），所有 Stage1 模型共享全部特征（"大锅饭"模式）。问题：

- **维度冗余**：光伏模型不需要体感温度，水电模型不需要风速
- **过拟合风险**：噪声特征挤占树模型分裂，在小数据集（丰水仅 37 天）上尤其危险
- **物理信号稀释**：关键物理变量被淹没在大批不相关特征中

**目标**：按发电物理机制，拆分为 4 个独立的特征 builder 函数，每个只产生该模型真正需要的定制化特征。

## 二、架构设计

```
build_features()
  ├── build_load_features(temp_avg, rh_avg, load, n)
  │     └→ 体感温度 + 省级加权温度预报
  ├── build_solar_features(rad_avg, cloud_avg, n)
  │     └→ 有效辐射 + 日间掩码 + 15min 爬坡率
  ├── build_wind_features(wind_avg, n)
  │     └→ 风速三次方 + 启停掩码 + 1h 湍流
  ├── build_hydro_features(precip_avg, n)
  │     └→ 3天/15天 流域汇流累计降水
  └── (现有共享特征 A–P 组保持不变)
```

每个 builder 函数：
- 输入：加权融合后的天气变量（已在 `build_features()` 前置计算好）
- 输出：`dict[str, np.ndarray]`，键=特征名，值=(n,) 数组
- 主函数负责把这些 dict 合并到统一的 `feature_list` 中

## 三、各模型专属特征明细

### 3.1 负荷 (Load) — 体感温度驱动空调/采暖

| 特征名 | 公式 | 说明 |
|---|---|---|
| `apparent_temp` | `1.07×T + 0.2×RH − 2.7` | 线性简化体感温度（替换当前 humidex 非线性公式） |
| `apparent_temp_pp` | `sf(apparent_temp, 96)` | t+96 体感温度预报 |
| `provincial_temp_pp` | `sf(temp_avg, 96)` | 全省加权温度预报（从 D 组移入 load 专属） |

> 当前 humidex 公式 `T + 0.5555×(vp − 10)` 替换为 `1.07T + 0.2RH − 2.7`（用户指定）。删除 `sat_vp`、`vp` 的计算。

### 3.2 光伏 (Solar) — 辐射断崖 + 云遮蔽

| 特征名 | 公式 | 说明 |
|---|---|---|
| `rad_eff` | `rad_avg × max(1−cloud_avg/100, 0)` | 云遮蔽非线性衰减（D 组已有，移入 solar 专属） |
| `is_daytime` | `(rad_avg > 10.0).astype(float)` | 日间布尔标识：辐射>10W/m² 为白天 |
| `is_daytime_pp` | `sf(is_daytime, 96)` | t+96 日间标识 |
| `ghi_ramp_15min` | `rad_avg − roll(rad_avg, 1)` | 15 分钟辐射一阶差分，捕捉云层快速移动 |

### 3.3 风电 (Wind) — 风能立方律 + 启停逻辑

| 特征名 | 公式 | 说明 |
|---|---|---|
| `wind_cubic` | `wind_avg³` | 风速三次方：风能与 V³ 正比（替换当前 `wind_power_density`，去掉 ρ/2 系数） |
| `is_generating_pp` | `(3 < wind_pp < 25).astype(float)` | 风机工作区间掩码：切入 3m/s，切出 25m/s |
| `wind_turbulence_1h` | `roll_std(wind_avg, 4)` | 1 小时湍流强度：过去 4 个 15min 步的滚动标准差 |

> ⚠️ **风向分解 (U/V) 暂不实施**：需要 API 新增拉取 `wind_direction_100m` 变量。当前 `HOURLY_VARS` 仅含 `wind_speed_100m`。后续可在 `weather_fetcher.py` 的 `HOURLY_VARS` 中加入，再在 builder 中实现 `Wind_U = V×sin(θ)`, `Wind_V = V×cos(θ)`。

### 3.4 水电 (Hydro) — 流域汇流滞后

| 特征名 | 公式 | 说明 |
|---|---|---|
| `precip_3d_sum` | 72h 滚动降水累计 | 短期暴雨洪峰（已有 `rain_72h_acc`，改用 hourly-aware 实现） |
| `precip_15d_sum` | 360h 滚动降水累计 | 水库基础水位 |
| `flood_pp` | `sf(is_flood, 96)` | 丰水期掩码（已有，归入 hydro 专属） |

## 四、关键技术点

### 4.1 降水分辨率对齐

天气数据为小时级，forward-fill 到 15 分钟。降水值在连续 4 个 15 分钟步上重复。直接用 15 分钟做滚动和会高估 4 倍。

**解决方案**：新增 `_hourly_rolling_sum()` 工具函数：

```python
def _hourly_rolling_sum(arr, n_hours):
    """对 ffill 后的 15-min 数组做小时级滚动求和。
    
    通过每隔 4 步取一次值重建小时级数据，避免重复计数。
    
    Parameters
    ----------
    arr : (n,) array — 15 分钟分辨率数组（从小时级 ffill 得来）
    n_hours : int — 滚动窗口（小时数）
    
    Returns
    -------
    (n,) array — 15 分钟分辨率结果
    """
    n = len(arr)
    arr_h = arr[::4]                          # 每 4 步取 1，重建小时级
    hourly_sum = roll_sum(arr_h, n_hours)     # 小时级滚动和
    result = np.repeat(hourly_sum, 4)          # 广播回 15-min
    if len(result) < n:
        result = np.pad(result, (0, n - len(result)), constant_values=np.nan)
    return result[:n]
```

### 4.2 加权权重占位

当前所有权重（0.5/0.2/0.3 等）为等权占位。用户提供各地用电量占比、装机容量分布数据后，替换 `LOAD_TEMP_WEIGHTS` 等字典即可。

## 五、实施步骤

### Step 1: 新增 `_hourly_rolling_sum()` 工具函数
位置：`feature_engine.py` 第 195 行附近（与其他 `roll_*` 函数并列）

### Step 2: 新增 `build_load_features()`
- 输入：`temp_avg, rh_avg, n`
- 用 `1.07T + 0.2RH - 2.7` 替换当前 humidex 公式
- 输出 `apparent_temp`, `apparent_temp_pp`, `provincial_temp_pp`

### Step 3: 新增 `build_solar_features()`
- 输入：`rad_avg, cloud_avg, n`
- 输出 `rad_eff`, `is_daytime`, `is_daytime_pp`, `ghi_ramp_15min`

### Step 4: 新增 `build_wind_features()`
- 输入：`wind_avg, n`
- 输出 `wind_cubic`, `is_generating_pp`, `wind_turbulence_1h`

### Step 5: 新增 `build_hydro_features()`
- 输入：`precip_avg, n`
- 用 `_hourly_rolling_sum` 计算 3d 和 15d 降水累积
- 输出 `precip_3d_sum`, `precip_15d_sum`

### Step 6: 集成到 `build_features()`
- 在天气特征计算完成后调用 4 个 builder
- 将返回值合并到 `feature_list`
- 更新 `solar_feat_cols`, `wind_feat_cols`, `hydro_feat_cols`, `load_feat_cols` 特征选择器

### Step 7: 清理被替换的旧代码
- 删除旧的 `apparent_temp` humidex 计算（`sat_vp`, `vp`）
- 删除旧的 `wind_power_density`（`0.6125 × V³`）
- 删除旧的 `rain_72h_acc`（改用 `_hourly_rolling_sum` 计算 `precip_3d_sum`）

### Step 8: 验证
```bash
python -m pipeline.feature_engine
# → 总特征维度：~178（原 171 − 4 替换 + 7 新增）
# → 枯水/丰水切分正常
# → 无 NaN 爆炸
# → apparent_temp 公式确认为 1.07T + 0.2RH - 2.7
```

## 六、涉及文件

| 文件 | 改动 | 行数估计 |
|---|---|---|
| `pipeline/feature_engine.py` | 新增 4 个 builder + 1 个工具函数 + 公式替换 + 选择器更新 | +120 / -30 |

## 七、实际特征维度变化（已实施 2026-06-25）

| 变化 | 来源 | 数量 |
|---|---|---|
| 原维度 | — | 171 |
| 公式替换 `apparent_temp` (now+pp) | humidex → 线性 `1.07T+0.2RH-2.7` | ±0 |
| 重命名 `wind_power` → `wind_cubic` (now+pp) | 去掉 ρ/2 系数 | ±0 |
| + 新增 `is_daytime` (now+pp) | solar builder | +2 |
| + 新增 `ghi_ramp_15min` | solar builder | +1 |
| + 新增 `is_generating_pp` | wind builder | +1 |
| + 新增 `wind_turbulence_1h` | wind builder | +1 |
| + 新增 `precip_15d_sum` | hydro builder（hourly-aware） | +1 |
| BUGFIX `rain_24h_mean` + `rain_72h_acc` | 原 roll_sum(24)=6h → _hourly_rolling_sum | ±0 |
| **实际总维度** | | **177** |
