# Stage1 训练报告

生成时间: 2026-06-29 10:41

## 指标汇总

| 变量 | 季节 | 策略 | Test R² | Test MAE (MW) |
|------|------|------|---------|--------------|
| 光伏 (MW) | 枯水期 | direct absolute | 0.9450 | 365.1 |
| 光伏 (MW) | 丰水期 | direct absolute | 0.9560 | 319.2 |
| 水电 (MW) | 枯水期 | lag_96 residual | -0.7302 | 3513.5 |
| 水电 (MW) | 丰水期 | lag_96 residual | -1.3011 | 9149.4 |
| 风电 (MW) | 枯水期 | direct absolute | 0.1929 | 922.9 |
| 风电 (MW) | 丰水期 | direct absolute | -0.3200 | 913.7 |
| 负荷 (MW) | 枯水期 | lag_96 residual | 0.3014 | 1082.5 |
| 负荷 (MW) | 丰水期 | lag_96 residual | 0.7107 | 1287.5 |

图表保存至: `G:\JAVA_Internship\em_prediction_service\pipeline\output\charts`