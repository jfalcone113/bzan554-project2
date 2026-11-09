# Confirmed measurable improvement

| Evaluation | Previous | Improved | Change |
|:---|---:|---:|---:|
| Mean validation R2, 3 seeds, matched cap 40 | 0.616953 | 0.622248 | +0.005295 |
| Mean validation RMSE, same budget | 30.7954 | 30.5818 | -0.2136 |
| Selected checkpoint validation R2 | 0.617674 | 0.623509 | +0.005835 |
| Test R2 after frozen selection | 0.619600 | 0.624535 | +0.004935 |
| Test RMSE | 31.6104 | 31.4047 | -0.2057 |

The matched three-seed mean comparison isolates the configuration gain from an extra-training-budget gain. The single-checkpoint rows use each selected checkpoint and therefore can differ in seed/budget. The new inference checkpoint uses seed 202, epoch 26, selected only after the configuration won on mean validation MSE.

The winning changes are **Adam to RMSprop**, followed by **L2 0.0001 to 0.00001**. Keep **50% SKU dropout**, **5% hidden dropout**, **three sigmoid layers**, and a **linear output**. All three paired confirmation seeds improved. This is a modest gain, not a claimed universal best architecture.

The user accepted any measurable gain, superseding the assistant's proposed +0.01 threshold. The original threshold result remains recorded in `selection.json`. No test score affected this change or configuration selection.

The presentation includes only these improving changes. All 42 trial results remain in `all_runs.csv`; the complete original plan is in `PLAN.md`. The active improved bundle is `best_model/`; the previous fitted baseline remains intact at `../model/`.
