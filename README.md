# Neural network quantity prediction

BZAN 554 · Group 6 · Deep Learning for Business Applications

The selected model uses **RMSprop and L2 = 0.00001** with three sigmoid hidden layers. Its validation R² is **0.623509**, and its test R² is **0.624535**. Across three matched seeds, mean validation R² improved from **0.616953 to 0.622248** (**+0.005295**), with gains in all three seeds.

This README documents **all 82 logged fits**: 30 original restoration/search fits, 42 expanded-tuning fits, and 10 presentation follow-up fits at a 40-epoch cap. Repeated seeds and reproduced baselines count as separate fits, not separate configurations. Tables include successful and unsuccessful experiments.

[Selected model](tuning/best_model/) · [Presentation PDF](presentation/Presentation.pdf) · [Editable slides](presentation/Presentation.qmd) · [Original trial CSV](presentation/all_runs.csv) · [Expanded trial CSV](tuning/all_runs.csv)

## Reading the results

- **R²:** higher is better. **RMSE, MSE, and MAE:** lower is better. Error metrics use the supplied scaled quantity units; MSE uses squared units and excludes regularization penalties.
- **Parent:** the configuration a trial branched from. Changes accumulate only along that branch; rows are not one continuous sequence.
- **Δ R²:** validation R² minus the same-seed parent's validation R², when both appear in the same study log. A dash means that matched parent is not logged there. It does not mean zero improvement.
- **Epoch selected / run:** the epoch whose weights were evaluated, followed by the number of training epochs completed. Early restoration trials retained final weights; later trials restored the best validation-MSE checkpoint.
- Values are rounded for browser readability. Linked CSV files preserve full precision, configuration JSON, parameter counts, updates, and timings. A value that rounds to zero may still be negative.

## Data and comparison protocol

The dataset contains **500,000 completed selling episodes**. Inputs are SKU, category, price, order, and duration; the target is quantity. The deterministic split, with split seed 42, groups duplicate SKU/order pairs:

| Training | Validation | Test |
| ---: | ---: | ---: |
| 349,975 | 74,736 | 75,289 |

Preprocessing is fitted on training data only. Validation selects settings and checkpoints. Test scoring follows frozen model selection, but the test partition has historical exposure. Different episodes of the same SKU can cross splits. Since duration describes a completed episode, these results do not establish advance forecasting or entirely unseen-product performance.

Most screening fits use seed **42**, batch **1,024**, and a **20-epoch cap**. Batch-size experiments change the number of updates per epoch, so equal epoch caps do not imply equal update counts. Expanded confirmation uses seeds **42, 101, and 202**, a **40-epoch cap**, and early-stopping patience **12**. Comparisons describe this dataset and budget, not universal rankings.

## Selected model

| Split | R² | RMSE | MSE | MAE |
| --- | --- | --- | --- | --- |
| Training | 0.678773 | 28.6223 | 819.2361 | 11.4846 |
| Validation | 0.623509 | 30.5308 | 932.1279 | 12.5083 |
| Test | 0.624535 | 31.4047 | 986.2526 | 12.7416 |

The checkpoint uses **seed 202, epoch 26**. The configuration was selected by three-seed mean validation MSE, then its best validation seed checkpoint was packaged. The single-checkpoint scores above differ from the three-seed means below.

| Setting | Selected value |
| --- | --- |
| Numeric inputs | Raw, log1p, and square-root price/order/duration; 9 standardized inputs |
| Embeddings | SKU: 8 dimensions; category: 6 dimensions |
| Hidden layers | 256 → 128 → 64, sigmoid, Glorot uniform initialization |
| Dropout | Hidden: 5%; whole-SKU embedding: 50% |
| Penalty | Hidden-kernel L2 = 0.00001 |
| Output / loss | Linear / MSE; standardized target, inverted for reported predictions |
| Optimizer | RMSprop, initial learning rate 0.001, rho 0.9 |
| Gradient clipping | Per-variable norm limit 10 |
| Learning-rate control | Halve after 4 unimproved validation epochs; minimum 0.00003 |
| Stopping / checkpoint | Patience 12; cap 40; restore minimum-validation-MSE checkpoint |
| Batch size | 1,024 |

The frozen comparison baseline at [`model/`](model/) uses Adam and L2 = 0.0001. The improved bundle is [`tuning/best_model/`](tuning/best_model/). No output clipping is applied. Saved inference checkpoints do not include restorable optimizer state.

## Presentation follow-up — 40-epoch budget

After the original studies, all displayed seed-42 configurations were checked with a **40-epoch cap**: **10 new fits**, plus **3 reused confirmation fits**. The earlier 20-epoch results below remain historical measurements. Only the epoch cap changes in this follow-up; early steps still evaluate final weights, while the training-control step adds best-validation checkpoint restoration. New runs reproduce their original epoch-20 validation R² within 0.00001 before acceptance. No test scores were recomputed and the selected bundle was not replaced.

A longer budget caused overfitting in some early models. Those declines are reported rather than filtered out; this follow-up sequence is not uniformly improving.

| Trial | Validation R² | RMSE | MSE | MAE | Selected / completed epoch | Existing fit reused? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `00_basic` | 0.363592 | 39.6943 | 1575.6401 | 14.1049 | 40 / 40 | False |
| `01_log` | 0.333642 | 40.6176 | 1649.7911 | 13.7717 | 40 / 40 | False |
| `02_sqrt` | 0.338907 | 40.4568 | 1636.7539 | 13.7148 | 40 / 40 | False |
| `03_hidden_dropout` | 0.404601 | 38.3941 | 1474.1076 | 13.8538 | 40 / 40 | False |
| `04_sku_dropout` | 0.583712 | 32.1039 | 1030.6600 | 12.6170 | 40 / 40 | False |
| `05_l2` | 0.599605 | 31.4851 | 991.3114 | 13.2571 | 40 / 40 | False |
| `c_seven` | -0.000000 | 49.7577 | 2475.8314 | 24.6744 | 21 / 33 | False |
| `d_l2_high` | 0.598059 | 31.5458 | 995.1378 | 13.5143 | 39 / 40 | False |
| `d_batch4096` | 0.619403 | 30.6968 | 942.2951 | 12.7995 | 38 / 40 | False |
| `c_bn_before` | 0.615603 | 30.8497 | 951.7033 | 12.5144 | 35 / 40 | False |
| `09_checkpoint` | 0.617674 | 30.7665 | 946.5750 | 12.9910 | 15 / 27 | True |
| `b_rmsprop001` | 0.620966 | 30.6337 | 938.4240 | 12.7014 | 21 / 33 | True |
| `d_l2_low` | 0.621686 | 30.6046 | 936.6421 | 12.5839 | 21 / 33 | True |

<details>
<summary>Training metrics and runtime for the 10 new follow-up fits</summary>

| Trial | Train R² | Train RMSE | Train MSE | Train MAE | Fit seconds | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `00_basic` | 0.793691 | 22.9381 | 526.1576 | 9.1410 | 131.43 | 1,094,799 |
| `01_log` | 0.811248 | 21.9404 | 481.3818 | 8.8863 | 133.12 | 1,095,567 |
| `02_sqrt` | 0.811922 | 21.9012 | 479.6618 | 8.8796 | 134.31 | 1,096,335 |
| `03_hidden_dropout` | 0.811781 | 21.9094 | 480.0213 | 8.9781 | 157.29 | 1,096,335 |
| `04_sku_dropout` | 0.709017 | 27.2416 | 742.1052 | 10.9542 | 156.17 | 1,096,335 |
| `05_l2` | 0.692541 | 28.0022 | 784.1237 | 11.8885 | 156.14 | 1,096,335 |
| `c_seven` | -0.000000 | 50.5008 | 2550.3347 | 24.5908 | 214.62 | 1,184,847 |
| `d_l2_high` | 0.599342 | 31.9658 | 1021.8107 | 13.3809 | 150.34 | 1,096,335 |
| `d_batch4096` | 0.663032 | 29.3152 | 859.3823 | 12.1952 | 84.23 | 1,096,335 |
| `c_bn_before` | 0.674696 | 28.8033 | 829.6328 | 11.1969 | 159.57 | 1,097,679 |

</details>

[Full-precision follow-up CSV](tuning/presentation_40/all_runs.csv) · [Protocol](tuning/presentation_40/protocol.json) · [Result and epoch records](tuning/presentation_40/runs/) · [Reproduction script](tuning/presentation_40.py)

## Original study — all 30 fits

The starting network uses three sigmoid layers (256/128/64), SKU/category embeddings, standardized raw numeric inputs, Adam at 0.001, and a linear output. Controls and transformations are added along the parent chain. Full configurations appear in the expandable section below.

### Feature restoration and training controls

| Trial | Parent | Change |
| --- | --- | --- |
| `00_basic` | — | Fresh basic model |
| `01_log` | `00_basic` | Add log1p features (price multiplied by 100 before log1p) |
| `02_sqrt` | `01_log` | Add square-root features |
| `03_hidden_dropout` | `02_sqrt` | Add 5% dropout after each hidden layer |
| `04_sku_dropout` | `03_hidden_dropout` | Add 50% whole-SKU embedding dropout |
| `05_l2` | `04_sku_dropout` | Add hidden-kernel L2 coefficient 0.0001 |
| `06_clipping` | `05_l2` | Add per-variable gradient norm clipping at 10 |
| `07_lr_reduction` | `06_clipping` | Halve learning rate after 4 unimproved validation epochs, floor 0.00003 |
| `08_early_stop` | `07_lr_reduction` | Stop after 12 unimproved validation epochs, retain final epoch |
| `09_checkpoint` | `08_early_stop` | Restore best validation MSE checkpoint |
| `10_exponential` | `09_checkpoint` | Replace linear output with exponential quantity link |
| `11_duration_rate` | `10_exponential` | Replace exponential quantity with positive rate times duration |

| Trial / seed | Validation R² | Δ R² vs parent | RMSE | MSE | MAE | Epoch selected / run |
| --- | --- | --- | --- | --- | --- | --- |
| `00_basic` / 42 | 0.421691 | — | 37.8391 | 1431.7962 | 14.4157 | 20 / 20 |
| `01_log` / 42 | 0.459058 | +0.037367 | 36.5962 | 1339.2811 | 13.6004 | 20 / 20 |
| `02_sqrt` / 42 | 0.459301 | +0.000243 | 36.5880 | 1338.6800 | 13.6419 | 20 / 20 |
| `03_hidden_dropout` / 42 | 0.486812 | +0.027511 | 35.6450 | 1270.5677 | 13.5723 | 20 / 20 |
| `04_sku_dropout` / 42 | 0.607463 | +0.120651 | 31.1746 | 971.8550 | 12.7222 | 20 / 20 |
| `05_l2` / 42 | 0.614490 | +0.007026 | 30.8943 | 954.4588 | 12.5423 | 20 / 20 |
| `06_clipping` / 42 | 0.614490 | -0.000000 | 30.8943 | 954.4588 | 12.5423 | 20 / 20 |
| `07_lr_reduction` / 42 | 0.611548 | -0.002941 | 31.0120 | 961.7412 | 12.3286 | 20 / 20 |
| `08_early_stop` / 42 | 0.611548 | +0.000000 | 31.0120 | 961.7412 | 12.3286 | 20 / 20 |
| `09_checkpoint` / 42 | 0.617674 | +0.006126 | 30.7665 | 946.5750 | 12.9910 | 15 / 20 |
| `10_exponential` / 42 | 0.611130 | -0.006544 | 31.0286 | 962.7763 | 13.3813 | 10 / 20 |
| `11_duration_rate` / 42 | 0.606811 | -0.004319 | 31.2005 | 973.4696 | 13.0979 | 11 / 20 |

The clipping, rate-reduction, stopping, and checkpoint trials are individually logged here. The slides combine them into one training-control step; that combined gain cannot be attributed to clipping alone.

### Historical search and branch trials

These trials include multi-setting edits and output-link alternatives. Their parent identifies branch provenance, not a claim that only one hyperparameter changed.

| Trial / seed | Validation R² | Δ R² vs parent | RMSE | MSE | MAE | Epoch selected / run |
| --- | --- | --- | --- | --- | --- | --- |
| `search_compact` / 42 | 0.572752 | -0.034059 | 32.5237 | 1057.7942 | 13.2542 | 2 / 14 |
| `search_wide` / 42 | 0.583012 | -0.023800 | 32.1309 | 1032.3932 | 13.2404 | 2 / 14 |
| `search_regularized` / 42 | 0.589826 | -0.016985 | 31.8673 | 1015.5221 | 13.0968 | 3 / 15 |
| `search_small_step` / 42 | 0.580005 | -0.026806 | 32.2465 | 1039.8364 | 13.3849 | 3 / 15 |
| `search_exponential` / 42 | 0.611128 | +0.004317 | 31.0287 | 962.7818 | 13.5136 | 7 / 19 |
| `search_exponential_no_sku` / 42 | 0.592657 | -0.014154 | 31.7571 | 1008.5123 | 13.3111 | 15 / 20 |
| `search_exponential_wide` / 42 | 0.608796 | +0.001985 | 31.1216 | 968.5552 | 13.1930 | 15 / 20 |
| `search_linear_no_sku` / 42 | 0.599969 | -0.006842 | 31.4708 | 990.4096 | 13.6601 | 15 / 20 |
| `search_duration_rate` / 42 | 0.604250 | -0.002561 | 31.3019 | 979.8103 | 13.9703 | 7 / 19 |
| `search_duration_rate_no_sku` / 42 | 0.587583 | -0.019228 | 31.9543 | 1021.0754 | 13.7210 | 15 / 20 |
| `branch_exponential_no_sku` / 42 | 0.595101 | -0.016029 | 31.6617 | 1002.4613 | 13.3388 | 15 / 20 |
| `branch_duration_rate_no_sku` / 42 | 0.592381 | -0.014430 | 31.7678 | 1009.1961 | 13.1952 | 15 / 20 |
| `12_match_hidden_dropout` / 42 | 0.609966 | -0.001164 | 31.0750 | 965.6575 | 13.7216 | 5 / 17 |
| `13_match_sku_dropout` / 42 | 0.611136 | +0.001170 | 31.0284 | 962.7605 | 13.0373 | 15 / 20 |

<details>
<summary>Settings and parent references for historical search trials</summary>

| Trial | Parent | Hidden widths | SKU embedding / enabled | Hidden / SKU dropout | L2 | Learning rate | Output link |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `search_compact` | `11_duration_rate` | 128 / 64 / 32 | 8 / True | 0% / 0% | 1e-05 | 0.003 | linear |
| `search_wide` | `11_duration_rate` | 256 / 128 / 64 | 12 / True | 0% / 0% | 1e-05 | 0.002 | linear |
| `search_regularized` | `11_duration_rate` | 192 / 96 / 48 | 8 / True | 5% / 0% | 0.0001 | 0.002 | linear |
| `search_small_step` | `11_duration_rate` | 128 / 64 / 32 | 16 / True | 0% / 0% | 1e-05 | 0.001 | linear |
| `search_exponential` | `11_duration_rate` | 128 / 64 / 32 | 8 / True | 0% / 50% | 0.0001 | 0.001 | exponential |
| `search_exponential_no_sku` | `11_duration_rate` | 128 / 64 / 32 | 4 / False | 0% / 0% | 0.0001 | 0.001 | exponential |
| `search_exponential_wide` | `11_duration_rate` | 256 / 128 / 64 | 8 / True | 0% / 70% | 0.0001 | 0.0005 | exponential |
| `search_linear_no_sku` | `11_duration_rate` | 128 / 64 / 32 | 4 / False | 0% / 0% | 1e-05 | 0.002 | linear |
| `search_duration_rate` | `11_duration_rate` | 128 / 64 / 32 | 8 / True | 0% / 50% | 0.0001 | 0.001 | duration_rate |
| `search_duration_rate_no_sku` | `11_duration_rate` | 128 / 64 / 32 | 4 / False | 0% / 0% | 0.0001 | 0.001 | duration_rate |
| `branch_exponential_no_sku` | `10_exponential` | 256 / 128 / 64 | 8 / False | 5% / 0% | 0.0001 | 0.001 | exponential |
| `branch_duration_rate_no_sku` | `11_duration_rate` | 256 / 128 / 64 | 8 / False | 5% / 0% | 0.0001 | 0.001 | duration_rate |
| `12_match_hidden_dropout` | `10_exponential` | 256 / 128 / 64 | 8 / True | 0% / 50% | 0.0001 | 0.001 | exponential |
| `13_match_sku_dropout` | `12_match_hidden_dropout` | 256 / 128 / 64 | 8 / True | 0% / 70% | 0.0001 | 0.001 | exponential |

</details>

### Additional original-study seeds

Seed-42 results are listed above; these four additional fits complete the original study’s two three-seed groups.

| Trial / seed | Validation R² | Δ R² vs parent | RMSE | MSE | MAE | Epoch selected / run |
| --- | --- | --- | --- | --- | --- | --- |
| `00_basic` / 101 | 0.450137 | — | 36.8967 | 1361.3684 | 14.3470 | 20 / 20 |
| `09_checkpoint` / 101 | 0.616188 | — | 30.8262 | 950.2527 | 12.3911 | 17 / 20 |
| `00_basic` / 202 | 0.469953 | — | 36.2258 | 1312.3060 | 14.3109 | 20 / 20 |
| `09_checkpoint` / 202 | 0.616057 | — | 30.8314 | 950.5774 | 12.9342 | 14 / 20 |

| Original configuration | Seeds | Mean validation R² | Minimum R² | Maximum R² | Mean RMSE |
| --- | --- | --- | --- | --- | --- |
| `00_basic` | 42 / 101 / 202 | 0.447260 | 0.421691 | 0.469953 | 36.9872 |
| `09_checkpoint` | 42 / 101 / 202 | 0.616640 | 0.616057 | 0.617674 | 30.8080 |

These original-study means use their original budgets. Use the matched 40-epoch confirmation below to assess the expanded-tuning improvement.

## Expanded tuning — all 42 fits

The reference is the frozen `09_checkpoint` model: raw/log/sqrt inputs, 256/128/64 sigmoid layers, 5% hidden dropout, 50% SKU dropout, L2 = 0.0001, Adam at 0.001, clipping, plateau reductions, early stopping, and best-checkpoint selection. All expanded trials keep 50% SKU dropout and a linear output.

### A. Activations and initialization

Parent: `a_baseline`, the reproduced frozen baseline. He initialization is tested jointly with several activations; those comparisons do not isolate activation alone. Sigmoid-scaled initialization uses variance scale 16 with fan-average mode.

| Trial | Tested setting |
| --- | --- |
| `a_baseline` | Reproduce packaged sigmoid baseline |
| `a_elu_he` | ELU alpha 1 and He initialization |
| `a_leaky_he` | Leaky ReLU slope .01 and He initialization |
| `a_prelu_he` | Learned PReLU and He initialization |
| `a_relu_glorot` | ReLU activation with matched Glorot initialization |
| `a_relu_he` | ReLU and He fan-in initialization |
| `a_sigmoid_scaled` | Sigmoid-scaled initialization |
| `a_swish_he` | Swish and He initialization |
| `a_tanh` | Tanh activation |

| Trial / seed | Validation R² | Δ R² vs parent | RMSE | MSE | MAE | Epoch selected / run |
| --- | --- | --- | --- | --- | --- | --- |
| `a_baseline` / 42 | 0.617674 | — | 30.7665 | 946.5750 | 12.9910 | 15 / 20 |
| `a_elu_he` / 42 | 0.614049 | -0.003625 | 30.9120 | 955.5497 | 12.4986 | 13 / 20 |
| `a_leaky_he` / 42 | 0.609345 | -0.008329 | 31.0998 | 967.1960 | 12.5230 | 16 / 20 |
| `a_prelu_he` / 42 | 0.613760 | -0.003914 | 30.9235 | 956.2650 | 12.5472 | 13 / 20 |
| `a_relu_glorot` / 42 | 0.612881 | -0.004793 | 30.9587 | 958.4408 | 12.4599 | 7 / 19 |
| `a_relu_he` / 42 | 0.611787 | -0.005887 | 31.0024 | 961.1491 | 12.5843 | 16 / 20 |
| `a_sigmoid_scaled` / 42 | 0.617336 | -0.000338 | 30.7801 | 947.4126 | 12.8605 | 15 / 20 |
| `a_swish_he` / 42 | 0.614610 | -0.003064 | 30.8895 | 954.1617 | 12.4817 | 7 / 19 |
| `a_tanh` / 42 | 0.612657 | -0.005017 | 30.9677 | 958.9963 | 13.5972 | 17 / 20 |

### B. Optimizers and learning rates

Parent: `a_baseline`. Momentum and Nesterov use momentum 0.9; RMSprop uses rho 0.9.

| Trial | Tested setting |
| --- | --- |
| `b_adagrad01` | adagrad learning rate 0.01 |
| `b_momentum01` | momentum learning rate 0.01 |
| `b_nesterov01` | nesterov learning rate 0.01 |
| `b_nesterov1` | nesterov learning rate 0.1 |
| `b_rmsprop0003` | rmsprop learning rate 0.0003 |
| `b_rmsprop001` | rmsprop learning rate 0.001 |
| `b_sgd01` | sgd learning rate 0.01 |
| `b_sgd1` | sgd learning rate 0.1 |

| Trial / seed | Validation R² | Δ R² vs parent | RMSE | MSE | MAE | Epoch selected / run |
| --- | --- | --- | --- | --- | --- | --- |
| `b_adagrad01` / 42 | 0.549091 | -0.068583 | 33.4122 | 1116.3743 | 13.6928 | 20 / 20 |
| `b_momentum01` / 42 | 0.588906 | -0.028767 | 31.9029 | 1017.7982 | 12.8642 | 19 / 20 |
| `b_nesterov01` / 42 | 0.586852 | -0.030822 | 31.9826 | 1022.8847 | 12.8022 | 19 / 20 |
| `b_nesterov1` / 42 | 0.601825 | -0.015849 | 31.3977 | 985.8140 | 12.6418 | 18 / 20 |
| `b_rmsprop0003` / 42 | 0.595350 | -0.022324 | 31.6519 | 1001.8456 | 12.7376 | 20 / 20 |
| `b_rmsprop001` / 42 | 0.618700 | +0.001026 | 30.7251 | 944.0338 | 13.1636 | 19 / 20 |
| `b_sgd01` / 42 | 0.492405 | -0.125269 | 35.4502 | 1256.7188 | 15.0857 | 20 / 20 |
| `b_sgd1` / 42 | 0.587922 | -0.029751 | 31.9411 | 1020.2344 | 13.0444 | 19 / 20 |

### C. Depth, width, and batch normalization

Parent: `b_rmsprop001` (RMSprop, L2 = 0.0001; validation R² = 0.618700). The one- and two-layer trials also change width, so these are architecture comparisons rather than pure depth effects.

| Trial | Tested setting |
| --- | --- |
| `c_bn_after` | Batch normalization after hidden activation |
| `c_bn_before` | Batch normalization before hidden activation |
| `c_five` | Hidden widths [256, 128, 128, 64, 64] |
| `c_flat_three` | Hidden widths [128, 128, 128] |
| `c_one` | Hidden widths [128] |
| `c_seven` | Hidden widths [256, 256, 128, 128, 64, 64, 32] |
| `c_two` | Hidden widths [128, 64] |
| `c_wide` | Hidden widths [512, 256, 128] |

| Trial / seed | Validation R² | Δ R² vs parent | RMSE | MSE | MAE | Epoch selected / run |
| --- | --- | --- | --- | --- | --- | --- |
| `c_bn_after` / 42 | 0.610685 | -0.008015 | 31.0464 | 963.8781 | 12.6249 | 18 / 20 |
| `c_bn_before` / 42 | 0.601397 | -0.017303 | 31.4146 | 986.8741 | 12.5585 | 20 / 20 |
| `c_five` / 42 | 0.615916 | -0.002784 | 30.8371 | 950.9264 | 12.8715 | 20 / 20 |
| `c_flat_three` / 42 | 0.617289 | -0.001411 | 30.7819 | 947.5269 | 13.0343 | 19 / 20 |
| `c_one` / 42 | 0.609720 | -0.008980 | 31.0848 | 966.2663 | 12.5113 | 20 / 20 |
| `c_seven` / 42 | -0.000006 | -0.618706 | 49.7579 | 2475.8453 | 24.7299 | 12 / 20 |
| `c_two` / 42 | 0.618136 | -0.000565 | 30.7479 | 945.4317 | 12.4903 | 16 / 20 |
| `c_wide` / 42 | 0.616150 | -0.002551 | 30.8277 | 950.3487 | 13.0453 | 20 / 20 |

### D. Batch size, regularization, and scheduling

Parent: `b_rmsprop001`, with L2 = 0.0001. Each trial branches independently from that parent; batch-size trials were not run with the final lower-L2 setting. Exponential refers to learning-rate decay, not an exponential output.

| Trial | Tested setting |
| --- | --- |
| `d_batch256` | Batch size 256 |
| `d_batch4096` | Batch size 4096 |
| `d_dropout0` | Hidden dropout 0 |
| `d_dropout10` | Hidden dropout .10 |
| `d_exponential` | Exponential LR decay .95 per epoch instead of plateau reductions |
| `d_l2_high` | L2 coefficient 0.001 |
| `d_l2_low` | L2 coefficient 0.00001 |
| `d_lr_low` | Initial rate 0.3 times parent |

| Trial / seed | Validation R² | Δ R² vs parent | RMSE | MSE | MAE | Epoch selected / run |
| --- | --- | --- | --- | --- | --- | --- |
| `d_batch256` / 42 | 0.615964 | -0.002736 | 30.8352 | 950.8073 | 12.5690 | 15 / 20 |
| `d_batch4096` / 42 | 0.599209 | -0.019492 | 31.5007 | 992.2916 | 12.7009 | 20 / 20 |
| `d_dropout0` / 42 | 0.616376 | -0.002324 | 30.8186 | 949.7883 | 13.4651 | 19 / 20 |
| `d_dropout10` / 42 | 0.618146 | -0.000554 | 30.7475 | 945.4063 | 13.0948 | 19 / 20 |
| `d_exponential` / 42 | 0.615115 | -0.003586 | 30.8693 | 952.9113 | 12.7568 | 19 / 20 |
| `d_l2_high` / 42 | 0.585506 | -0.033194 | 32.0346 | 1026.2175 | 13.5083 | 19 / 20 |
| `d_l2_low` / 42 | 0.620904 | +0.002204 | 30.6362 | 938.5771 | 12.7295 | 19 / 20 |
| `d_lr_low` / 42 | 0.595350 | -0.023350 | 31.6519 | 1001.8456 | 12.7376 | 20 / 20 |

### E. Matched three-seed confirmation

`e_baseline`: Adam, L2 = 0.0001. `e_finalist1`: RMSprop, L2 = 0.00001. `e_finalist2`: RMSprop, L2 = 0.0001. All use the same architecture and matched 40-epoch caps.

| Trial / seed | Validation R² | Δ R² vs parent | RMSE | MSE | MAE | Epoch selected / run |
| --- | --- | --- | --- | --- | --- | --- |
| `e_baseline` / 101 | 0.617030 | — | 30.7924 | 948.1696 | 12.7056 | 33 / 40 |
| `e_baseline` / 202 | 0.616155 | — | 30.8275 | 950.3359 | 12.6344 | 24 / 36 |
| `e_baseline` / 42 | 0.617674 | — | 30.7665 | 946.5750 | 12.9910 | 15 / 27 |
| `e_finalist1` / 101 | 0.621550 | — | 30.6101 | 936.9794 | 12.4152 | 33 / 40 |
| `e_finalist1` / 202 | 0.623509 | — | 30.5308 | 932.1279 | 12.5083 | 26 / 38 |
| `e_finalist1` / 42 | 0.621686 | — | 30.6046 | 936.6421 | 12.5839 | 21 / 33 |
| `e_finalist2` / 101 | 0.621748 | — | 30.6021 | 936.4889 | 12.6922 | 33 / 40 |
| `e_finalist2` / 202 | 0.621111 | — | 30.6279 | 938.0660 | 13.0744 | 26 / 38 |
| `e_finalist2` / 42 | 0.620966 | — | 30.6337 | 938.4240 | 12.7014 | 21 / 33 |

Δ values are omitted here because the parent screening fits used a different budget. The matched comparisons are the group means and paired seeds below.

### Confirmation summary

| Configuration | Mean validation R² | R² range | Mean RMSE |
| --- | --- | --- | --- |
| e_baseline | 0.616953 | 0.616155–0.617674 | 30.7954 |
| e_finalist1 | 0.622248 | 0.621550–0.623509 | 30.5818 |
| e_finalist2 | 0.621275 | 0.620966–0.621748 | 30.6212 |

| Seed | Adam baseline R² | RMSprop + lower L2 R² | Paired gain |
| --- | --- | --- | --- |
| 42 | 0.617674 | 0.621686 | +0.004012 |
| 101 | 0.617030 | 0.621550 | +0.004520 |
| 202 | 0.616155 | 0.623509 | +0.007354 |

The user accepted a measurable improvement, superseding the initially proposed +0.01 mean R² threshold. The measured mean gain is **+0.005295**, positive in **3/3 paired seeds**. Three seeds do not establish statistical significance or a universal best model.

<details>
<summary>Original study: training metrics, runtime, and parameter counts for every fit</summary>

| Trial / seed | Train R² | Train RMSE | Train MSE | Train MAE | Fit seconds | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| `00_basic` / 42 | 0.764500 | 24.5072 | 600.6044 | 10.1268 | 74.86 | 1,094,799 |
| `01_log` / 42 | 0.786039 | 23.3596 | 545.6724 | 9.5099 | 75.49 | 1,095,567 |
| `02_sqrt` / 42 | 0.785509 | 23.3885 | 547.0236 | 9.5324 | 76.32 | 1,096,335 |
| `03_hidden_dropout` / 42 | 0.781184 | 23.6232 | 558.0544 | 9.8876 | 93.45 | 1,096,335 |
| `04_sku_dropout` / 42 | 0.714348 | 26.9909 | 728.5087 | 11.1156 | 86.81 | 1,096,335 |
| `05_l2` / 42 | 0.711550 | 27.1227 | 735.6432 | 10.9493 | 85.14 | 1,096,335 |
| `06_clipping` / 42 | 0.711550 | 27.1227 | 735.6432 | 10.9493 | 89.51 | 1,096,335 |
| `07_lr_reduction` / 42 | 0.695036 | 27.8884 | 777.7605 | 10.9547 | 84.89 | 1,096,335 |
| `08_early_stop` / 42 | 0.695036 | 27.8884 | 777.7605 | 10.9547 | 84.87 | 1,096,335 |
| `09_checkpoint` / 42 | 0.696150 | 27.8373 | 774.9180 | 11.5811 | 86.09 | 1,096,335 |
| `10_exponential` / 42 | 0.701377 | 27.5969 | 761.5895 | 12.0275 | 94.20 | 1,096,335 |
| `11_duration_rate` / 42 | 0.702304 | 27.5540 | 759.2244 | 11.8818 | 95.09 | 1,096,335 |
| `search_compact` / 42 | 0.677571 | 28.6758 | 822.3023 | 11.0929 | 32.93 | 1,062,415 |
| `search_wide` / 42 | 0.686615 | 28.2708 | 799.2373 | 11.0791 | 56.64 | 1,621,651 |
| `search_regularized` / 42 | 0.691086 | 28.0684 | 787.8347 | 10.9619 | 50.49 | 1,076,815 |
| `search_small_step` / 42 | 0.675738 | 28.7572 | 826.9754 | 11.0505 | 46.72 | 2,112,023 |
| `search_exponential` / 42 | 0.686568 | 28.2729 | 799.3567 | 12.3393 | 47.56 | 1,062,415 |
| `search_exponential_no_sku` / 42 | 0.592343 | 32.2438 | 1039.6612 | 13.3286 | 41.70 | 537,611 |
| `search_exponential_wide` / 42 | 0.657877 | 29.5386 | 872.5288 | 12.4876 | 74.49 | 1,096,335 |
| `search_linear_no_sku` / 42 | 0.594680 | 32.1512 | 1033.7023 | 13.6942 | 41.31 | 537,611 |
| `search_duration_rate` / 42 | 0.681364 | 28.5066 | 812.6290 | 12.9095 | 48.83 | 1,062,415 |
| `search_duration_rate_no_sku` / 42 | 0.596130 | 32.0937 | 1030.0044 | 13.7381 | 43.27 | 537,611 |
| `branch_exponential_no_sku` / 42 | 0.588606 | 32.3912 | 1049.1929 | 13.3762 | 85.26 | 1,096,335 |
| `branch_duration_rate_no_sku` / 42 | 0.594632 | 32.1531 | 1033.8232 | 13.2298 | 90.27 | 1,096,335 |
| `00_basic` / 101 | 0.759237 | 24.7796 | 614.0269 | 10.2714 | 68.77 | 1,094,799 |
| `09_checkpoint` / 101 | 0.701762 | 27.5791 | 760.6054 | 10.9292 | 96.71 | 1,096,335 |
| `00_basic` / 202 | 0.763134 | 24.5782 | 604.0873 | 10.3138 | 70.53 | 1,094,799 |
| `09_checkpoint` / 202 | 0.695049 | 27.8878 | 777.7280 | 11.5455 | 99.61 | 1,096,335 |
| `12_match_hidden_dropout` / 42 | 0.665399 | 29.2121 | 853.3439 | 12.7918 | 67.23 | 1,096,335 |
| `13_match_sku_dropout` / 42 | 0.664273 | 29.2612 | 856.2163 | 12.2775 | 79.53 | 1,096,335 |

</details>

<details>
<summary>Original study: exact configuration for each distinct trial name</summary>

#### `00_basic`

```json
{
  "change": "Fresh basic model",
  "checkpoint": false,
  "clip": 0.0,
  "dropout": 0.0,
  "early_stop": false,
  "embedding": 8,
  "l2": 0.0,
  "link": "linear",
  "lr": 0.001,
  "name": "00_basic",
  "parent": "",
  "reduce_lr": false,
  "sku_dropout": 0.0,
  "transforms": 0,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `01_log`

```json
{
  "change": "Add log1p features (price multiplied by 100 before log1p)",
  "checkpoint": false,
  "clip": 0.0,
  "dropout": 0.0,
  "early_stop": false,
  "embedding": 8,
  "l2": 0.0,
  "link": "linear",
  "lr": 0.001,
  "name": "01_log",
  "parent": "00_basic",
  "reduce_lr": false,
  "sku_dropout": 0.0,
  "transforms": 1,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `02_sqrt`

```json
{
  "change": "Add square-root features",
  "checkpoint": false,
  "clip": 0.0,
  "dropout": 0.0,
  "early_stop": false,
  "embedding": 8,
  "l2": 0.0,
  "link": "linear",
  "lr": 0.001,
  "name": "02_sqrt",
  "parent": "01_log",
  "reduce_lr": false,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `03_hidden_dropout`

```json
{
  "change": "Add 5% dropout after each hidden layer",
  "checkpoint": false,
  "clip": 0.0,
  "dropout": 0.05,
  "early_stop": false,
  "embedding": 8,
  "l2": 0.0,
  "link": "linear",
  "lr": 0.001,
  "name": "03_hidden_dropout",
  "parent": "02_sqrt",
  "reduce_lr": false,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `04_sku_dropout`

```json
{
  "change": "Add 50% whole-SKU embedding dropout",
  "checkpoint": false,
  "clip": 0.0,
  "dropout": 0.05,
  "early_stop": false,
  "embedding": 8,
  "l2": 0.0,
  "link": "linear",
  "lr": 0.001,
  "name": "04_sku_dropout",
  "parent": "03_hidden_dropout",
  "reduce_lr": false,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `05_l2`

```json
{
  "change": "Add hidden-kernel L2 coefficient 0.0001",
  "checkpoint": false,
  "clip": 0.0,
  "dropout": 0.05,
  "early_stop": false,
  "embedding": 8,
  "l2": 0.0001,
  "link": "linear",
  "lr": 0.001,
  "name": "05_l2",
  "parent": "04_sku_dropout",
  "reduce_lr": false,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `06_clipping`

```json
{
  "change": "Add per-variable gradient norm clipping at 10",
  "checkpoint": false,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": false,
  "embedding": 8,
  "l2": 0.0001,
  "link": "linear",
  "lr": 0.001,
  "name": "06_clipping",
  "parent": "05_l2",
  "reduce_lr": false,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `07_lr_reduction`

```json
{
  "change": "Halve learning rate after 4 unimproved validation epochs, floor 0.00003",
  "checkpoint": false,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": false,
  "embedding": 8,
  "l2": 0.0001,
  "link": "linear",
  "lr": 0.001,
  "name": "07_lr_reduction",
  "parent": "06_clipping",
  "reduce_lr": true,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `08_early_stop`

```json
{
  "change": "Stop after 12 unimproved validation epochs, retain final epoch",
  "checkpoint": false,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "linear",
  "lr": 0.001,
  "name": "08_early_stop",
  "parent": "07_lr_reduction",
  "reduce_lr": true,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `09_checkpoint`

```json
{
  "change": "Restore best validation MSE checkpoint",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "linear",
  "lr": 0.001,
  "name": "09_checkpoint",
  "parent": "08_early_stop",
  "reduce_lr": true,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `10_exponential`

```json
{
  "change": "Replace linear output with exponential quantity link",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "exponential",
  "lr": 0.001,
  "name": "10_exponential",
  "parent": "09_checkpoint",
  "reduce_lr": true,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `11_duration_rate`

```json
{
  "change": "Replace exponential quantity with positive rate times duration",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "duration_rate",
  "lr": 0.001,
  "name": "11_duration_rate",
  "parent": "10_exponential",
  "reduce_lr": true,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `search_compact`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 8,
  "l2": 1e-05,
  "link": "linear",
  "lr": 0.003,
  "name": "search_compact",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    128,
    64,
    32
  ]
}
```

#### `search_wide`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 12,
  "l2": 1e-05,
  "link": "linear",
  "lr": 0.002,
  "name": "search_wide",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `search_regularized`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "linear",
  "lr": 0.002,
  "name": "search_regularized",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    192,
    96,
    48
  ]
}
```

#### `search_small_step`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 16,
  "l2": 1e-05,
  "link": "linear",
  "lr": 0.001,
  "name": "search_small_step",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    128,
    64,
    32
  ]
}
```

#### `search_exponential`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "exponential",
  "lr": 0.001,
  "name": "search_exponential",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    128,
    64,
    32
  ]
}
```

#### `search_exponential_no_sku`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 4,
  "l2": 0.0001,
  "link": "exponential",
  "lr": 0.001,
  "name": "search_exponential_no_sku",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": false,
  "widths": [
    128,
    64,
    32
  ]
}
```

#### `search_exponential_wide`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "exponential",
  "lr": 0.0005,
  "name": "search_exponential_wide",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.7,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `search_linear_no_sku`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 4,
  "l2": 1e-05,
  "link": "linear",
  "lr": 0.002,
  "name": "search_linear_no_sku",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": false,
  "widths": [
    128,
    64,
    32
  ]
}
```

#### `search_duration_rate`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "duration_rate",
  "lr": 0.001,
  "name": "search_duration_rate",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    128,
    64,
    32
  ]
}
```

#### `search_duration_rate_no_sku`

```json
{
  "change": "Historical search candidate (multiple settings vary)",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 4,
  "l2": 0.0001,
  "link": "duration_rate",
  "lr": 0.001,
  "name": "search_duration_rate_no_sku",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": false,
  "widths": [
    128,
    64,
    32
  ]
}
```

#### `branch_exponential_no_sku`

```json
{
  "change": "Remove SKU information, otherwise identical",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "exponential",
  "lr": 0.001,
  "name": "branch_exponential_no_sku",
  "parent": "10_exponential",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": false,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `branch_duration_rate_no_sku`

```json
{
  "change": "Remove SKU information, otherwise identical",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.05,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "duration_rate",
  "lr": 0.001,
  "name": "branch_duration_rate_no_sku",
  "parent": "11_duration_rate",
  "reduce_lr": true,
  "sku_dropout": 0.0,
  "transforms": 2,
  "use_sku": false,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `12_match_hidden_dropout`

```json
{
  "change": "Match original selected model: hidden dropout 5% to 0%",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "exponential",
  "lr": 0.001,
  "name": "12_match_hidden_dropout",
  "parent": "10_exponential",
  "reduce_lr": true,
  "sku_dropout": 0.5,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `13_match_sku_dropout`

```json
{
  "change": "Increase SKU dropout from 50% to 70%",
  "checkpoint": true,
  "clip": 10.0,
  "dropout": 0.0,
  "early_stop": true,
  "embedding": 8,
  "l2": 0.0001,
  "link": "exponential",
  "lr": 0.001,
  "name": "13_match_sku_dropout",
  "parent": "12_match_hidden_dropout",
  "reduce_lr": true,
  "sku_dropout": 0.7,
  "transforms": 2,
  "use_sku": true,
  "widths": [
    256,
    128,
    64
  ]
}
```

</details>

<details>
<summary>Expanded tuning: training metrics, runtime, and parameter counts for every fit</summary>

| Trial / seed | Train R² | Train RMSE | Train MSE | Train MAE | Fit seconds | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| `a_baseline` / 42 | 0.696150 | 27.8373 | 774.9180 | 11.5811 | 84.52 | 1,096,335 |
| `a_elu_he` / 42 | 0.705991 | 27.3828 | 749.8200 | 11.0387 | 95.05 | 1,096,335 |
| `a_leaky_he` / 42 | 0.710553 | 27.1696 | 738.1875 | 11.0298 | 90.53 | 1,096,335 |
| `a_prelu_he` / 42 | 0.712074 | 27.0981 | 734.3081 | 11.1156 | 105.89 | 1,096,783 |
| `a_relu_glorot` / 42 | 0.700300 | 27.6466 | 764.3355 | 11.1427 | 81.09 | 1,096,335 |
| `a_relu_he` / 42 | 0.722567 | 26.5998 | 707.5482 | 10.8454 | 89.16 | 1,096,335 |
| `a_sigmoid_scaled` / 42 | 0.692948 | 27.9837 | 783.0858 | 11.5133 | 82.70 | 1,096,335 |
| `a_swish_he` / 42 | 0.692853 | 27.9880 | 783.3282 | 11.2563 | 94.71 | 1,096,335 |
| `a_tanh` / 42 | 0.699181 | 27.6982 | 767.1898 | 11.9743 | 94.68 | 1,096,335 |
| `b_adagrad01` / 42 | 0.531885 | 34.5521 | 1193.8489 | 13.8072 | 85.84 | 1,096,335 |
| `b_momentum01` / 42 | 0.575015 | 32.9219 | 1083.8536 | 12.9315 | 79.12 | 1,096,335 |
| `b_nesterov01` / 42 | 0.573191 | 32.9925 | 1088.5060 | 12.8673 | 82.15 | 1,096,335 |
| `b_nesterov1` / 42 | 0.624476 | 30.9469 | 957.7110 | 12.4747 | 80.54 | 1,096,335 |
| `b_rmsprop0003` / 42 | 0.589888 | 32.3407 | 1045.9234 | 12.6868 | 87.98 | 1,096,335 |
| `b_rmsprop001` / 42 | 0.656370 | 29.6036 | 876.3703 | 12.4425 | 88.25 | 1,096,335 |
| `b_sgd01` / 42 | 0.474226 | 36.6183 | 1340.8999 | 15.2184 | 69.80 | 1,096,335 |
| `b_sgd1` / 42 | 0.573922 | 32.9642 | 1086.6417 | 13.1019 | 69.60 | 1,096,335 |
| `c_bn_after` / 42 | 0.672257 | 28.9111 | 835.8540 | 11.4338 | 105.77 | 1,098,127 |
| `c_bn_before` / 42 | 0.642760 | 30.1841 | 911.0821 | 11.4182 | 105.26 | 1,097,679 |
| `c_five` / 42 | 0.644996 | 30.0895 | 905.3798 | 12.2261 | 114.53 | 1,117,007 |
| `c_flat_three` / 42 | 0.654759 | 29.6729 | 880.4812 | 12.3111 | 82.53 | 1,085,199 |
| `c_one` / 42 | 0.630110 | 30.7139 | 943.3433 | 11.7068 | 45.61 | 1,052,175 |
| `c_seven` / 42 | -0.000006 | 50.5010 | 2550.3503 | 24.6461 | 169.37 | 1,184,847 |
| `c_two` / 42 | 0.653101 | 29.7441 | 884.7087 | 11.7692 | 54.17 | 1,060,367 |
| `c_wide` / 42 | 0.648206 | 29.9532 | 897.1917 | 12.4667 | 141.80 | 1,225,615 |
| `d_batch256` / 42 | 0.645087 | 30.0857 | 905.1477 | 11.7986 | 171.75 | 1,096,335 |
| `d_batch4096` / 42 | 0.596627 | 32.0739 | 1028.7352 | 12.6445 | 50.47 | 1,096,335 |
| `d_dropout0` / 42 | 0.657444 | 29.5573 | 873.6332 | 12.6318 | 76.29 | 1,096,335 |
| `d_dropout10` / 42 | 0.649797 | 29.8853 | 893.1341 | 12.4752 | 91.23 | 1,096,335 |
| `d_exponential` / 42 | 0.634851 | 30.5164 | 931.2518 | 12.3351 | 83.54 | 1,096,335 |
| `d_l2_high` / 42 | 0.574851 | 32.9283 | 1084.2734 | 13.5254 | 89.28 | 1,096,335 |
| `d_l2_low` / 42 | 0.676130 | 28.7398 | 825.9764 | 11.7435 | 91.00 | 1,096,335 |
| `d_lr_low` / 42 | 0.589888 | 32.3407 | 1045.9234 | 12.6868 | 91.33 | 1,096,335 |
| `e_baseline` / 101 | 0.709295 | 27.2286 | 741.3954 | 11.1763 | 186.22 | 1,096,335 |
| `e_baseline` / 202 | 0.707081 | 27.3321 | 747.0417 | 11.1460 | 171.22 | 1,096,335 |
| `e_baseline` / 42 | 0.696150 | 27.8373 | 774.9180 | 11.5811 | 133.47 | 1,096,335 |
| `e_finalist1` / 101 | 0.690264 | 28.1057 | 789.9308 | 11.1278 | 186.17 | 1,096,335 |
| `e_finalist1` / 202 | 0.678773 | 28.6223 | 819.2361 | 11.4846 | 176.88 | 1,096,335 |
| `e_finalist1` / 42 | 0.679546 | 28.5878 | 817.2638 | 11.5357 | 152.51 | 1,096,335 |
| `e_finalist2` / 101 | 0.672883 | 28.8835 | 834.2565 | 11.7165 | 182.21 | 1,096,335 |
| `e_finalist2` / 202 | 0.665465 | 29.2092 | 853.1756 | 12.1963 | 167.70 | 1,096,335 |
| `e_finalist2` / 42 | 0.662197 | 29.3515 | 861.5096 | 11.9111 | 147.93 | 1,096,335 |

</details>

<details>
<summary>Expanded tuning: exact configuration for each distinct trial name</summary>

#### `a_baseline`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Reproduce packaged sigmoid baseline",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_baseline",
  "optimizer": "adam",
  "parent": "packaged_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `a_elu_he`

```json
{
  "activation": "elu",
  "batch": 1024,
  "bn": "none",
  "change": "ELU alpha 1 and He initialization",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "he_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_elu_he",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `a_leaky_he`

```json
{
  "activation": "leaky_relu",
  "batch": 1024,
  "bn": "none",
  "change": "Leaky ReLU slope .01 and He initialization",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "he_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_leaky_he",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `a_prelu_he`

```json
{
  "activation": "prelu",
  "batch": 1024,
  "bn": "none",
  "change": "Learned PReLU and He initialization",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "he_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_prelu_he",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `a_relu_glorot`

```json
{
  "activation": "relu",
  "batch": 1024,
  "bn": "none",
  "change": "ReLU activation with matched Glorot initialization",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_relu_glorot",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `a_relu_he`

```json
{
  "activation": "relu",
  "batch": 1024,
  "bn": "none",
  "change": "ReLU and He fan-in initialization",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "he_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_relu_he",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `a_sigmoid_scaled`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Sigmoid-scaled initialization",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "sigmoid_scaled",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_sigmoid_scaled",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `a_swish_he`

```json
{
  "activation": "swish",
  "batch": 1024,
  "bn": "none",
  "change": "Swish and He initialization",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "he_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_swish_he",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `a_tanh`

```json
{
  "activation": "tanh",
  "batch": 1024,
  "bn": "none",
  "change": "Tanh activation",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "a_tanh",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "A",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `b_adagrad01`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "adagrad learning rate 0.01",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.01,
  "max_epochs": 20,
  "name": "b_adagrad01",
  "optimizer": "adagrad",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "B",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `b_momentum01`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "momentum learning rate 0.01",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.01,
  "max_epochs": 20,
  "name": "b_momentum01",
  "optimizer": "momentum",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "B",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `b_nesterov01`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "nesterov learning rate 0.01",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.01,
  "max_epochs": 20,
  "name": "b_nesterov01",
  "optimizer": "nesterov",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "B",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `b_nesterov1`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "nesterov learning rate 0.1",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.1,
  "max_epochs": 20,
  "name": "b_nesterov1",
  "optimizer": "nesterov",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "B",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `b_rmsprop0003`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "rmsprop learning rate 0.0003",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.0003,
  "max_epochs": 20,
  "name": "b_rmsprop0003",
  "optimizer": "rmsprop",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "B",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `b_rmsprop001`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "rmsprop learning rate 0.001",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "b_rmsprop001",
  "optimizer": "rmsprop",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "B",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `b_sgd01`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "sgd learning rate 0.01",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.01,
  "max_epochs": 20,
  "name": "b_sgd01",
  "optimizer": "sgd",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "B",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `b_sgd1`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "sgd learning rate 0.1",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.1,
  "max_epochs": 20,
  "name": "b_sgd1",
  "optimizer": "sgd",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "B",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `c_bn_after`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "after",
  "change": "Batch normalization after hidden activation",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "c_bn_after",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "C",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `c_bn_before`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "before",
  "change": "Batch normalization before hidden activation",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "c_bn_before",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "C",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `c_five`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Hidden widths [256, 128, 128, 64, 64]",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "c_five",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "C",
  "widths": [
    256,
    128,
    128,
    64,
    64
  ]
}
```

#### `c_flat_three`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Hidden widths [128, 128, 128]",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "c_flat_three",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "C",
  "widths": [
    128,
    128,
    128
  ]
}
```

#### `c_one`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Hidden widths [128]",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "c_one",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "C",
  "widths": [
    128
  ]
}
```

#### `c_seven`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Hidden widths [256, 256, 128, 128, 64, 64, 32]",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "c_seven",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "C",
  "widths": [
    256,
    256,
    128,
    128,
    64,
    64,
    32
  ]
}
```

#### `c_two`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Hidden widths [128, 64]",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "c_two",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "C",
  "widths": [
    128,
    64
  ]
}
```

#### `c_wide`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Hidden widths [512, 256, 128]",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "c_wide",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "C",
  "widths": [
    512,
    256,
    128
  ]
}
```

#### `d_batch256`

```json
{
  "activation": "sigmoid",
  "batch": 256,
  "bn": "none",
  "change": "Batch size 256",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "d_batch256",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "D",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `d_batch4096`

```json
{
  "activation": "sigmoid",
  "batch": 4096,
  "bn": "none",
  "change": "Batch size 4096",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "d_batch4096",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "D",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `d_dropout0`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Hidden dropout 0",
  "clip": 10.0,
  "hidden_dropout": 0.0,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "d_dropout0",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "D",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `d_dropout10`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Hidden dropout .10",
  "clip": 10.0,
  "hidden_dropout": 0.1,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "d_dropout10",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "D",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `d_exponential`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Exponential LR decay .95 per epoch instead of plateau reductions",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "d_exponential",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "exponential",
  "sku_dropout": 0.5,
  "stage": "D",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `d_l2_high`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "L2 coefficient 0.001",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.001,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "d_l2_high",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "D",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `d_l2_low`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "L2 coefficient 0.00001",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 1e-05,
  "lr": 0.001,
  "max_epochs": 20,
  "name": "d_l2_low",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "D",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `d_lr_low`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Initial rate 0.3 times parent",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.0003,
  "max_epochs": 20,
  "name": "d_lr_low",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "D",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `e_baseline`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Baseline under matched 40-epoch confirmation budget",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 40,
  "name": "e_baseline",
  "optimizer": "adam",
  "parent": "a_baseline",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "E",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `e_finalist1`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Confirm screening configuration d_l2_low",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 1e-05,
  "lr": 0.001,
  "max_epochs": 40,
  "name": "e_finalist1",
  "optimizer": "rmsprop",
  "parent": "d_l2_low",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "E",
  "widths": [
    256,
    128,
    64
  ]
}
```

#### `e_finalist2`

```json
{
  "activation": "sigmoid",
  "batch": 1024,
  "bn": "none",
  "change": "Confirm screening configuration b_rmsprop001",
  "clip": 10.0,
  "hidden_dropout": 0.05,
  "initializer": "glorot_uniform",
  "l1": 0.0,
  "l2": 0.0001,
  "lr": 0.001,
  "max_epochs": 40,
  "name": "e_finalist2",
  "optimizer": "rmsprop",
  "parent": "b_rmsprop001",
  "patience": 12,
  "schedule": "plateau",
  "sku_dropout": 0.5,
  "stage": "E",
  "widths": [
    256,
    128,
    64
  ]
}
```

</details>

## Run the selected model

Use an environment matching [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock). The measured environment used TensorFlow 2.20.0, Keras 3.15.1, and four CPU threads per fit. With dependencies installed, run from the project root:

```bash
python model/pricing_model.py verify --bundle tuning/best_model
python model/pricing_model.py predict --bundle tuning/best_model --input data/pricing.csv --output predictions.csv
python tuning/retrain_best.py --output tuning/retrained_best
```

Prediction input requires `sku`, `price`, `order`, `duration`, and `category`; `quantity` is optional. Output adds `predicted_quantity` in the supplied target units. Keep `model.keras`, `preprocessing.json`, and `seen.npz` together. Verification recomputes metrics against the selected bundle’s saved reference. Retraining uses only the frozen winning configuration and requires a new or empty output folder. Exact floating-point results can vary across hardware and software versions.

To reproduce the previous baseline instead:

```bash
python model/pricing_model.py train --data data/pricing.csv --output model/retrained
```

## Rebuild the presentation

Requires Quarto and XeLaTeX:

```bash
python presentation/build_slides.py
quarto render presentation/Presentation.qmd --to beamer
```

The [16-slide presentation](presentation/Presentation.pdf) shows the improving sequence, four unsuccessful RMSprop parameter tests, final confirmation, and evaluation context. Team contributions remain placeholders. AI assistance: Codex implemented experiments, logged results, and drafted the presentation and documentation.

## Project files and provenance

| Location | Purpose |
| --- | --- |
| [`model/`](model/) | Frozen Adam baseline, shared preprocessing/prediction code, and verification evidence |
| [`tuning/best_model/`](tuning/best_model/) | Selected RMSprop/lower-L2 checkpoint, preprocessing, configuration, and metrics |
| [`presentation/`](presentation/) | Current deck, builder, original numeric trial log, and slide evidence |
| [`tuning/all_runs.csv`](tuning/all_runs.csv) | Full-precision expanded trial metrics and configurations |
| [`tuning/epoch_metrics.csv`](tuning/epoch_metrics.csv) | Per-epoch expanded tuning metrics |
| [`tuning/runs/`](tuning/runs/) | Individual expanded trial result records and epoch histories |
| [`tuning/PLAN.md`](tuning/PLAN.md) / [`tuning/IMPROVEMENT_LOG.md`](tuning/IMPROVEMENT_LOG.md) | Search protocol and accepted improvement |
| [`data/pricing.csv`](data/pricing.csv) | Source data, retained locally |
| [`resources/`](resources/) | Course sources, assignment requirements, and reference notes |
| [`CLEANUP_CURRENT.json`](CLEANUP_CURRENT.json) | Removed-file inventory |

Redundant trial checkpoints, cached splits, old deck backups, and temporary outputs were removed. Historical checkpoint paths in trial logs are provenance strings, not live files. The original cleanup retained only the frozen baseline and selected checkpoint. The follow-up now also retains its new trial checkpoints and cache for reproducibility. Selected-model retraining rebuilds its cache automatically; historical full-search resume state is incomplete after cleanup.

Course context: [`MODEL_TUNING_REFERENCE.md`](resources/MODEL_TUNING_REFERENCE.md) and [`CLASS_RESOURCES.md`](resources/CLASS_RESOURCES.md). The supplied *deep.pdf* covers RMSprop (pp. 32–33), clipping (pp. 77–78), checkpoints/stopping (pp. 82–87), L2 (pp. 89–91), and dropout (pp. 93–98). Exact ranges and whole-SKU masking are project choices. Sequence models were not fitted: the supplied material and episode-level benchmark did not establish a sequence-ready comparison.
