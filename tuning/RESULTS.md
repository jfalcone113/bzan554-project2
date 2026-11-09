# Expanded tuning: measured results

Completed fits: 42. Failed fits: 0. Fixed split seed 42. Original model validation R² **0.6177**, RMSE **30.77**.

## Screening leaderboard

| Run | Validation R² | RMSE | Selected epoch | Seconds |
|:---|---:|---:|---:|---:|
| d_l2_low | 0.6209 | 30.64 | 19 | 91.0 |
| b_rmsprop001 | 0.6187 | 30.73 | 19 | 88.2 |
| d_dropout10 | 0.6181 | 30.75 | 19 | 91.2 |
| c_two | 0.6181 | 30.75 | 16 | 54.2 |
| a_baseline | 0.6177 | 30.77 | 15 | 84.5 |
| a_sigmoid_scaled | 0.6173 | 30.78 | 15 | 82.7 |
| c_flat_three | 0.6173 | 30.78 | 19 | 82.5 |
| d_dropout0 | 0.6164 | 30.82 | 19 | 76.3 |
| c_wide | 0.6161 | 30.83 | 20 | 141.8 |
| d_batch256 | 0.6160 | 30.84 | 15 | 171.8 |
| c_five | 0.6159 | 30.84 | 20 | 114.5 |
| d_exponential | 0.6151 | 30.87 | 19 | 83.5 |
| a_swish_he | 0.6146 | 30.89 | 7 | 94.7 |
| a_elu_he | 0.6140 | 30.91 | 13 | 95.1 |
| a_prelu_he | 0.6138 | 30.92 | 13 | 105.9 |
| a_relu_glorot | 0.6129 | 30.96 | 7 | 81.1 |
| a_tanh | 0.6127 | 30.97 | 17 | 94.7 |
| a_relu_he | 0.6118 | 31.00 | 16 | 89.2 |
| c_bn_after | 0.6107 | 31.05 | 18 | 105.8 |
| c_one | 0.6097 | 31.08 | 20 | 45.6 |
| a_leaky_he | 0.6093 | 31.10 | 16 | 90.5 |
| b_nesterov1 | 0.6018 | 31.40 | 18 | 80.5 |
| c_bn_before | 0.6014 | 31.41 | 20 | 105.3 |
| d_batch4096 | 0.5992 | 31.50 | 20 | 50.5 |
| b_rmsprop0003 | 0.5953 | 31.65 | 20 | 88.0 |
| d_lr_low | 0.5953 | 31.65 | 20 | 91.3 |
| b_momentum01 | 0.5889 | 31.90 | 19 | 79.1 |
| b_sgd1 | 0.5879 | 31.94 | 19 | 69.6 |
| b_nesterov01 | 0.5869 | 31.98 | 19 | 82.1 |
| d_l2_high | 0.5855 | 32.03 | 19 | 89.3 |
| b_adagrad01 | 0.5491 | 33.41 | 20 | 85.8 |
| b_sgd01 | 0.4924 | 35.45 | 20 | 69.8 |
| c_seven | -0.0000 | 49.76 | 12 | 169.4 |

## Improving stage winners

| Stage | Configuration | R² | RMSE |
|:---|---:|---:|---:|
| A | a_baseline | 0.6177 | 30.77 |
| B | b_rmsprop001 | 0.6187 | 30.73 |
| D | d_l2_low | 0.6209 | 30.64 |

Each stage winner may change multiple settings. Full one-configuration comparisons and losing results remain in the CSV. Paired seed numbers do not ensure identical initialization after shape or layer changes. Screening is one-seed evidence. Confirmation uses three matched seeds and a longer equal budget.

## Three-seed confirmation

| Configuration | Mean R² | Range | Mean RMSE |
|:---|---:|---:|---:|
| e_baseline | 0.6170 | 0.6162-0.6177 | 30.80 |
| e_finalist1 | 0.6222 | 0.6215-0.6235 | 30.58 |
| e_finalist2 | 0.6213 | 0.6210-0.6217 | 30.62 |

Mean paired validation R² gain: **+0.0053**. User-requested measurable-improvement criterion met: **True**. These three-seed ranges are not confidence intervals.

Selected configuration: `e_finalist1`, seed 202, epoch 26. Hyperparameters:

```json
{
  "name": "e_finalist1",
  "stage": "E",
  "parent": "d_l2_low",
  "change": "Confirm screening configuration d_l2_low",
  "widths": [
    256,
    128,
    64
  ],
  "activation": "sigmoid",
  "initializer": "glorot_uniform",
  "bn": "none",
  "hidden_dropout": 0.05,
  "sku_dropout": 0.5,
  "l1": 0.0,
  "l2": 1e-05,
  "optimizer": "rmsprop",
  "lr": 0.001,
  "schedule": "plateau",
  "batch": 1024,
  "max_epochs": 40,
  "patience": 12,
  "clip": 10.0
}
```

## Frozen-model final evaluation

| Model / split | R² | MSE | RMSE | MAE |
|:---|---:|---:|---:|---:|
| New / train | 0.6788 | 819.24 | 28.62 | 11.48 |
| New / validation | 0.6235 | 932.13 | 30.53 | 12.51 |
| New / test | 0.6245 | 986.25 | 31.40 | 12.74 |
| Old / test | 0.6196 | 999.21 | 31.61 | 13.21 |

Configuration selection was frozen before new test scoring. The test partition has historical exposure. A different split was not used to manufacture a better score. Reloaded validation matched the selected checkpoint. BN inference-statistic checks passed where applicable. The original model remains unchanged. The active deck includes the two improving tuning steps; superseded presentation backups were removed.

## Reproduce and interpret

Run `../.venv/bin/python tuning/run_tuning.py`, then `../.venv/bin/python tuning/finalize_user_result.py` and `../.venv/bin/python tuning/report.py` from the project root. Completed requests are reused only with exact source/configuration/code signatures. `PLAN.md` gives proposed ranges, course-page citations, stopping rules, and the 42-fit bound. Installed APIs are recorded in `environment.json`. All checkpoints are inference checkpoints without the custom optimizer state. No sequence/image architecture or unrealistic augmentation has been implied by these tabular inputs.
