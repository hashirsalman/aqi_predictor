# SHAP Explainability Summary

Generated from the selected Phase 7 local model artifacts using the same chronological test split.
The local feature/target file used here was previously uploaded to Hopsworks and cloud read-back validated.

## day1

- Model family: `ridge`
- Explanation sample rows: `500`
- CSV: `C:\Users\henna\Downloads\10Pearls_Shine\reports\explainability\shap_importance_day1.csv`
- Figure: `C:\Users\henna\Downloads\10Pearls_Shine\reports\explainability\shap_importance_day1.png`

| rank | feature | mean_abs_shap |
| --- | --- | --- |
| 1 | pm2_5 | 10.790660 |
| 2 | sulphur_dioxide_lag_1h | 9.706414 |
| 3 | sulphur_dioxide | 8.300261 |
| 4 | aqi_roll_mean_12h | 6.440057 |
| 5 | aqi_roll_max_24h | 5.972157 |
| 6 | aqi_roll_mean_24h | 5.949773 |
| 7 | temperature_2m_lag_24h | 4.899285 |
| 8 | pm2_5_roll_mean_6h | 4.408344 |
| 9 | pm2_5_lag_24h | 4.233008 |
| 10 | aqi_roll_mean_6h | 3.908056 |

## day2

- Model family: `gradient_boosting`
- Explanation sample rows: `500`
- CSV: `C:\Users\henna\Downloads\10Pearls_Shine\reports\explainability\shap_importance_day2.csv`
- Figure: `C:\Users\henna\Downloads\10Pearls_Shine\reports\explainability\shap_importance_day2.png`

| rank | feature | mean_abs_shap |
| --- | --- | --- |
| 1 | month_cos | 3.336943 |
| 2 | pm2_5 | 3.276295 |
| 3 | day_of_year | 2.204138 |
| 4 | pm2_5_roll_mean_12h | 1.632060 |
| 5 | pm2_5_roll_mean_6h | 1.217359 |
| 6 | day_of_month | 0.643408 |
| 7 | sulphur_dioxide_lag_12h | 0.623133 |
| 8 | pm10_lag_24h | 0.592585 |
| 9 | pm2_5_roll_mean_24h | 0.586536 |
| 10 | pm10_roll_mean_24h | 0.513247 |

## day3

- Model family: `gradient_boosting`
- Explanation sample rows: `500`
- CSV: `C:\Users\henna\Downloads\10Pearls_Shine\reports\explainability\shap_importance_day3.csv`
- Figure: `C:\Users\henna\Downloads\10Pearls_Shine\reports\explainability\shap_importance_day3.png`

| rank | feature | mean_abs_shap |
| --- | --- | --- |
| 1 | month_cos | 4.815054 |
| 2 | day_of_year | 3.252683 |
| 3 | temperature_2m_roll_mean_24h | 1.891783 |
| 4 | pm2_5 | 0.926994 |
| 5 | wind_speed_10m_roll_mean_24h | 0.866469 |
| 6 | day_of_month | 0.855969 |
| 7 | pm2_5_roll_mean_12h | 0.837589 |
| 8 | surface_pressure | 0.661284 |
| 9 | precipitation_roll_sum_24h | 0.584167 |
| 10 | sulphur_dioxide_lag_24h | 0.525809 |
