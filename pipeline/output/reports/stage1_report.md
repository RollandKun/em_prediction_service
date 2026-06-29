# Stage1 训练报告

生成时间: 2026-06-29 15:26

## 指标汇总

| 变量 | 季节 | 策略 | Test R² | Test MAE (MW) |
|------|------|------|---------|--------------|
| 光伏 (MW) | 枯水期 | direct absolute | 0.9455 | 355.1 |
| 光伏 (MW) | 丰水期 | direct absolute | 0.9563 | 318.9 |
| 水电 (MW) | 枯水期 | lag_96 residual | -0.5150 | 3137.2 |
| 水电 (MW) | 丰水期 | lag_96 residual | -1.3152 | 9165.6 |
| 风电 (MW) | 枯水期 | direct absolute | 0.0875 | 1009.6 |
| 风电 (MW) | 丰水期 | direct absolute | -0.1874 | 873.7 |
| 负荷 (MW) | 枯水期 | lag_96 residual | 0.3695 | 1006.4 |
| 负荷 (MW) | 丰水期 | lag_96 residual | 0.7139 | 1341.0 |

图表保存至: `G:\JAVA_Internship\em_prediction_service\pipeline\output\charts`