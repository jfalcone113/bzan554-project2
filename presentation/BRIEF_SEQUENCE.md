# Model progression at a 40-epoch cap

Same split, seed 42, batch 1024, and 40-epoch cap. Validation drives selection. These are the original displayed configurations rerun or reused at cap 40; historical cap-20 logs remain separate.

| Step | Change | Validation R² | Validation RMSE |
|---:|:---|---:|---:|
| 0 | Baseline | 0.3636 | 39.69 |
| 1 | Add log features | 0.3336 | 40.62 |
| 2 | Add square-root features | 0.3389 | 40.46 |
| 3 | Add hidden dropout | 0.4046 | 38.39 |
| 4 | Add SKU dropout | 0.5837 | 32.10 |
| 5 | Add L2 penalties | 0.5996 | 31.49 |
| 6 | Add validation-based training controls | 0.6177 | 30.77 |
| 7 | Switch Adam to RMSprop | 0.6210 | 30.63 |
| 8 | Reduce the L2 penalty | 0.6217 | 30.60 |

The final configuration uses RMSprop, L2 0.00001, unchanged widths 256/128/64, sigmoid activation, hidden dropout 5%, SKU dropout 50%, and a linear output. The step tables above use seed 42 and cap 40.

Confirmation uses seeds 42/101/202 with matched 40-epoch caps. Previous-configuration mean validation R2 0.61695283; new mean 0.62224816; gain +0.00529533, positive in all three paired seeds. Selected checkpoint: seed 202, epoch 26, validation R2 0.62350911, RMSE 30.53077042. Test R2 0.62453456, RMSE 31.40465929, MAE 12.74164712.

Step 6 bundles training controls and its net gain cannot be attributed solely to checkpoint restoration. The following two displayed changes isolate optimizer and then L2 at the same 40-epoch cap and seed 42. Final mean confirmation scores use equal 40-epoch caps and are distinguished from single-checkpoint scores. Configuration ranking was frozen before test scoring.

All non-improving trials remain in the full logs: original restoration in `all_runs.csv`, expanded tuning in `../tuning/all_runs.csv`. The selected improved bundle is `../tuning/best_model/`; the earlier model remains at `../model/`. Contributions remain placeholders.
