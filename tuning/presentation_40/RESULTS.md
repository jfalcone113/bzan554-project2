# Presentation trials: 40-epoch-cap follow-up

Ten new seed-42 fits and three reused seed-42 confirmation fits. Same source data and split. Only epoch cap changes. Early restoration steps still evaluate final-epoch weights; expanded trials and steps 6–8 restore best validation checkpoints. No new test evaluation or selected-model replacement.

All new runs must reproduce their original epoch-20 validation R² within 0.00001 before acceptance.

| Trial | Validation R² | RMSE | MSE | MAE | Selected / completed epoch | Reused |
|---|---:|---:|---:|---:|---:|---|
| 00_basic | 0.363592 | 39.6943 | 1575.6401 | 14.1049 | 40 / 40 | False |
| 01_log | 0.333642 | 40.6176 | 1649.7911 | 13.7717 | 40 / 40 | False |
| 02_sqrt | 0.338907 | 40.4568 | 1636.7539 | 13.7148 | 40 / 40 | False |
| 03_hidden_dropout | 0.404601 | 38.3941 | 1474.1076 | 13.8538 | 40 / 40 | False |
| 04_sku_dropout | 0.583712 | 32.1039 | 1030.6600 | 12.6170 | 40 / 40 | False |
| 05_l2 | 0.599605 | 31.4851 | 991.3114 | 13.2571 | 40 / 40 | False |
| c_seven | -0.000000 | 49.7577 | 2475.8314 | 24.6744 | 21 / 33 | False |
| d_l2_high | 0.598059 | 31.5458 | 995.1378 | 13.5143 | 39 / 40 | False |
| d_batch4096 | 0.619403 | 30.6968 | 942.2951 | 12.7995 | 38 / 40 | False |
| c_bn_before | 0.615603 | 30.8497 | 951.7033 | 12.5144 | 35 / 40 | False |
| 09_checkpoint | 0.617674 | 30.7665 | 946.5750 | 12.9910 | 15 / 27 | True |
| b_rmsprop001 | 0.620966 | 30.6337 | 938.4240 | 12.7014 | 21 / 33 | True |
| d_l2_low | 0.621686 | 30.6046 | 936.6421 | 12.5839 | 21 / 33 | True |
