**Table 10. Six-arm ablation tournament (5-seed ensemble, OOS 1% VaR vs GJR-GARCH)**

| Model Variant | Asset | Breaches | Total Obs | Kupiec p-value | DM Stat (neg=TFT) | DM p-value | Mean Loss Diff |
|---|---|---|---|---|---|---|---|
| 0_GJR_GARCH_Baseline | NIFTY50 | 3.000 | 500 | 0.3315 | 0.0000 | 1.000 | 0.0000 |
| 0_GJR_GARCH_Baseline | BANKNIFTY | 2.000 | 500 | 0.1250 | 0.0000 | 1.000 | 0.0000 |
| 0_GJR_GARCH_Baseline | NIFTYIT | 10.000 | 500 | 0.0479 | 0.0000 | 1.000 | 0.0000 |
| 1_Raw_TFT | NIFTY50 | 2.000 | 500 | 0.1250 | 1.832 | 0.0669 | 0.0021 |
| 1_Raw_TFT | BANKNIFTY | 2.000 | 500 | 0.1250 | 0.1939 | 0.8463 | 0.0002 |
| 1_Raw_TFT | NIFTYIT | 11.000 | 500 | 0.0199 | 0.4301 | 0.6671 | 0.0013 |
| 2_GARCH_TFT | NIFTY50 | 3.000 | 500 | 0.3315 | 1.269 | 0.2046 | 0.0012 |
| 2_GARCH_TFT | BANKNIFTY | 2.000 | 500 | 0.1250 | 1.298 | 0.1941 | 0.0009 |
| 2_GARCH_TFT | NIFTYIT | 5.000 | 500 | 1.000 | -1.001 | 0.3168 | -0.0026 |
| 3_US_VIX_TFT | NIFTY50 | 1.000 | 500 | 0.0282 | 0.4833 | 0.6289 | 0.0007 |
| 3_US_VIX_TFT | BANKNIFTY | 2.000 | 500 | 0.1250 | 0.2412 | 0.8094 | 0.0003 |
| 3_US_VIX_TFT | NIFTYIT | 7.000 | 500 | 0.3966 | 0.1973 | 0.8436 | 0.0002 |
| 4_India_VIX_TFT | NIFTY50 | 3.000 | 500 | 0.3315 | 0.1494 | 0.8813 | 0.0001 |
| 4_India_VIX_TFT | BANKNIFTY | 2.000 | 500 | 0.1250 | -0.1219 | 0.9030 | -0.0000 |
| 4_India_VIX_TFT | NIFTYIT | 8.000 | 500 | 0.2149 | -0.7535 | 0.4511 | -0.0010 |
| 5_Full_ECTFT | NIFTY50 | 1.000 | 500 | 0.0282 | 0.2816 | 0.7782 | 0.0004 |
| 5_Full_ECTFT | BANKNIFTY | 2.000 | 500 | 0.1250 | -0.3932 | 0.6942 | -0.0006 |
| 5_Full_ECTFT | NIFTYIT | 11.000 | 500 | 0.0199 | 0.6740 | 0.5003 | 0.0009 |
