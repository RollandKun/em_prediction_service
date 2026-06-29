# -*- coding: utf-8 -*-
"""
shared/weather_config.py — 天气节点/集群/变量配置（唯一真相源）
============================================================
被以下模块共用：
  - ingestion/weather_fetcher.py  (NODES, VAR_PREFIXES)
  - pipeline/feature_engine.py    (CLUSTERS, NODE_NAMES, ALL_WEATHER_KEYS, REGION_WEIGHTS)
  - export_base_table.py          (CLUSTERS)
"""

# ═══════════════════════════════════════════════════════════════════
# 1. 19 子节点 GPS 坐标（Open-Meteo API 拉取用）
# ═══════════════════════════════════════════════════════════════════

NODES = [
    # ── 负荷集群：成都平原（3 子节点）──
    {"name": "cd_center", "lat": 30.67, "lon": 104.07},  # 天府广场，参考点
    {"name": "cd_west",   "lat": 30.68, "lon": 103.85},  # 温江/郫都，西郊平原
    {"name": "cd_east",   "lat": 30.38, "lon": 104.53},  # 简阳/东部新区

    # ── 负荷集群：川东城市群（3 子节点）──
    {"name": "dz_main",   "lat": 31.22, "lon": 107.50},  # 达州市区，极端高温极点
    {"name": "dz_south",  "lat": 30.83, "lon": 107.23},  # 大竹/邻水
    {"name": "dz_north",  "lat": 31.86, "lon": 107.79},  # 万源，北端

    # ── 负荷集群：川南城市群（3 子节点）──
    {"name": "yb_main",   "lat": 28.75, "lon": 104.62},  # 宜宾市区，高湿"桑拿天"
    {"name": "yb_north",  "lat": 29.03, "lon": 104.58},  # 自贡/内江方向
    {"name": "yb_east",   "lat": 28.90, "lon": 105.43},  # 泸州方向

    # ── 风光集群：凉山（3 子节点）──
    {"name": "ls_main",   "lat": 27.88, "lon": 102.26},  # 昭觉/布拖，全省最大风电基地
    {"name": "ls_south",  "lat": 26.66, "lon": 102.25},  # 会东/会理，高原光伏
    {"name": "ls_west",   "lat": 27.42, "lon": 101.51},  # 木里，雅砻江中游水电

    # ── 光伏集群：甘孜（2 子节点）──
    {"name": "gz_main",   "lat": 30.00, "lon": 100.27},  # 理塘/道孚，高海拔强日照
    {"name": "gz_north",  "lat": 31.62, "lon": 100.00},  # 炉霍/甘孜县

    # ── 水电集群：雅安（2 子节点）──
    {"name": "ya_main",   "lat": 29.98, "lon": 102.98},  # 雨城，大渡河/青衣江径流区
    {"name": "ya_west",   "lat": 29.86, "lon": 102.22},  # 天全/宝兴，青衣江上游

    # ── 水电集群：攀枝花（3 子节点）──
    {"name": "pzh_main",  "lat": 26.58, "lon": 101.71},  # 攀枝花市区
    {"name": "pzh_north", "lat": 26.90, "lon": 101.55},  # 二滩水库
    {"name": "pzh_east",  "lat": 26.60, "lon": 102.10},  # 金沙江东段
]

# ═══════════════════════════════════════════════════════════════════
# 2. 派生列表（从 NODES 自动生成，保证一致性）
# ═══════════════════════════════════════════════════════════════════

NODE_NAMES = [n["name"] for n in NODES]                      # 19 个节点名称

VAR_PREFIXES = ["temp", "rh", "precip", "cloud", "wind", "rad"]  # 6 个变量前缀

ALL_WEATHER_KEYS = [f"{p}_{n}" for p in VAR_PREFIXES for n in NODE_NAMES]  # 114 个 key

# ═══════════════════════════════════════════════════════════════════
# 3. 7 区域集群定义（集群内等权平均）
# ═══════════════════════════════════════════════════════════════════

CLUSTERS = {
    "chengdu":    ["cd_center", "cd_west", "cd_east"],           # 成都平原 → 负荷
    "dazhou":     ["dz_main", "dz_south", "dz_north"],           # 川东城市群 → 负荷
    "yibin":      ["yb_main", "yb_north", "yb_east"],            # 川南城市群 → 负荷
    "liangshan":  ["ls_main", "ls_south", "ls_west"],            # 凉山 → 风电/光伏/水电
    "ganzi":      ["gz_main", "gz_north"],                       # 甘孜 → 光伏
    "yaan":       ["ya_main", "ya_west"],                        # 雅安 → 水电
    "panzhihua":  ["pzh_main", "pzh_north", "pzh_east"],         # 攀枝花 → 水电
}

# ═══════════════════════════════════════════════════════════════════
# 4. 区域间加权融合权重（Step 2）
# ═══════════════════════════════════════════════════════════════════
# 等权占位符，真实权重（用电量占比/装机容量分布）待用户提供

REGION_WEIGHTS = {
    # 负荷相关：温度 + 湿度
    "temp": {"chengdu": 0.5, "dazhou": 0.2, "yibin": 0.3},
    "rh":   {"chengdu": 0.5, "dazhou": 0.2, "yibin": 0.3},
    # 风光相关：辐射 + 云量
    "rad":   {"liangshan": 0.6, "ganzi": 0.4},
    "cloud": {"liangshan": 0.6, "ganzi": 0.4},
    # 风电：风速
    "wind": {"liangshan": 0.7, "ganzi": 0.3},
    # 水电：降水
    "precip": {"yaan": 0.4, "panzhihua": 0.3, "liangshan": 0.3},
}

# ═══════════════════════════════════════════════════════════════════
# 5. 区域标签（用于导出/显示）
# ═══════════════════════════════════════════════════════════════════

CLUSTER_LABELS = {
    "chengdu":    "成都平原",
    "dazhou":     "川东城市群",
    "yibin":      "川南城市群",
    "liangshan":  "凉山",
    "ganzi":      "甘孜",
    "yaan":       "雅安",
    "panzhihua":  "攀枝花",
}
