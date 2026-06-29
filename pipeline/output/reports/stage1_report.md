# Stage1 训练报告

生成时间: 2026-06-26 14:52

## 指标汇总

| 变量 | 季节 | 策略 | Test R² | Test MAE (MW) |
|------|------|------|---------|--------------|
| 光伏 (MW) | 枯水期 | direct absolute | 0.9484 | 354.9 |
| 光伏 (MW) | 丰水期 | direct absolute | 0.9564 | 317.4 |
| 水电 (MW) | 枯水期 | lag_96 residual | -0.0940 | 2916.9 |
| 水电 (MW) | 丰水期 | lag_96 residual | -1.2193 | 8968.9 |
| 风电 (MW) | 枯水期 | direct absolute | 0.1299 | 971.1 |
| 风电 (MW) | 丰水期 | direct absolute | -0.5073 | 913.0 |
| 负荷 (MW) | 枯水期 | lag_96 residual | 0.2784 | 1136.2 |
| 负荷 (MW) | 丰水期 | lag_96 residual | 0.7986 | 1655.0 |

图表保存至: `G:\JAVA_Internship\em_prediction_service\pipeline\output\charts`