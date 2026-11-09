# Expanded neural-network tuning plan

## Objective and decision rules

Improve quantity prediction over the frozen sigmoid baseline (`../model/`): validation R2 0.6176738573, RMSE 30.7664597692; historical test R2 0.6195999975. Select exclusively using validation MSE (equivalently R2 on this fixed partition). The existing test partition has historical exposure and is not a new external holdout. Do not score new candidates on test until the configuration and inference checkpoint are frozen.

A noticeable improvement is proposed as **at least +0.01 mean validation R2** over the baseline under matched confirmation budgets, with improvement in at least two of three paired seeds. This is a practical decision rule, not a confidence interval or slide-prescribed threshold. Report small gains honestly even if the threshold is not met. Preserve the original model and presentation.

## Sources and newly released controls

Read all of `resources/deep.md`. It covers the same 105-page `resources/deep.pdf` inventory; the new opportunity is permission to use its controls beyond the earlier three-sigmoid-layer restriction. Relevant primary pages:

- Optimizers: deep.pdf pp. 12-37. Rate schedules: pp. 20-22.
- Initialization and saturation: pp. 38-58. Nonsaturating activations: p. 61; fundamentals.pdf pp. 116-162.
- Batch normalization before/after activation, learned scale/shift and training moving statistics: pp. 63-75.
- Gradient clipping: pp. 77-78. Early stopping/checkpoints: pp. 82-87.
- L1/L2 and dropout: pp. 89-99. Depth/width: pp. 103-105.

The notes contain instructional shorthand and algebra/API pitfalls. Use installed TensorFlow/Keras implementations, not handwritten Adam pseudo-code with conflicting moment labels and t=0 corrections. The second raw moment is not variance. He fan-in and lecture fan-average variants are distinct. Activation/optimizer rankings and claims that widening always helps are hypotheses, not guaranteed results.

## Fixed evaluation and preprocessing

Use all 349,975 training rows and 74,736 validation rows from the existing deterministic SKU/order split, split seed 42. Training-only numeric/target scaling and seen-ID vocabularies must match the packaged baseline. Retain raw/log1p/sqrt numeric features, SKU/category embedding dimensions 8/6, zero embedding initialization, linear standardized-quantity output and MSE loss. Numeric metrics use the supplied quantity scale. Evaluation MSE excludes regularization penalties.

SKU dropout remains **50%**, with a whole-vector mask, in every candidate. Hidden dropout starts at 5% and is refined separately. Inference always uses `training=False`, especially for BN moving statistics. No synthetic tabular augmentation without a defensible label-preserving mechanism. No sequence model: rows are episodes, there is no supplied sequence-ready design, and the decks do not establish detailed LSTM/GRU tuning.

## Bounded adaptive search: 42 fits maximum

All proposed values below are experiments, not slide-prescribed optima. Each fit restarts from scratch. Each stage is frozen before it runs; its winning parent is the lowest validation MSE observed so far. Carry parents forward if no child improves. Two independent processes, four CPU threads each, on eight available CPUs. Sum of fit times is aggregate work, not elapsed time.

### A. Activations and initialization: 9 fits

Full training set, seed 42, 20-epoch cap, stopping patience 12, plateau reduction after four unimproved epochs. Reproduce the baseline as a control. Compare sigmoid with Glorot; sigmoid with sigmoid-scaled fan-average variance scaling 16; tanh with Glorot; ReLU with Glorot; ReLU with He uniform fan-in; Leaky ReLU slope 0.01 with He; learned PReLU with zero initial slope and He; ELU alpha 1 with He; Swish with He. Keep all other controls fixed. Comparisons that alter activation and initializer together are labeled bundles; matched ReLU Glorot/He separates initialization.

### B. Optimizers/rates: 8 fits

Around the best A configuration: plain SGD rates 0.01/0.1, momentum 0.9 at 0.01, Nesterov momentum 0.9 at 0.01/0.1, AdaGrad at 0.01, RMSprop rho 0.9 at 0.001/0.0003. Adam remains the carried parent if better. Different rates avoid claiming the best optimizer from a single arbitrary shared rate. Twenty-epoch cap and seed 42.

### C. Depth, width, and BN: 8 fits

Around the best observed configuration: widths [128], [128,64], [128,128,128], [256,128,128,64,64], [512,256,128], [256,256,128,128,64,64,32]; also original parent widths with BN before or after hidden activation. BN uses momentum .99, epsilon .001, and no redundant dense bias before BN. Hidden dropout remains 5%, acknowledging that BN/dropout combinations can interact. Twenty-epoch cap and seed 42.

### D. Refinement: 8 fits

Around the best observed configuration: batches 256 and 4096; L2 0.00001 and 0.001; hidden dropout 0 and 0.10; initial learning rate 0.3 times its parent; exponential decay 0.95 per epoch, replacing plateau reductions. Convert epoch decay to optimizer steps using actual updates per epoch. Equal epochs imply different update counts for different batch sizes, so log both runtime and updates. Twenty-epoch cap and seed 42.

### E. Confirmation: 9 fits

Take the strongest two distinct non-baseline configurations from A-D and the baseline. Fresh full-training fits with a 40-epoch cap, patience 12, and seeds 42/101/202. Rank configuration by mean validation MSE, then choose its lowest-validation-MSE seed checkpoint. Compare against the paired baseline at the same 40-epoch budget. This separates an architectural gain from an extra-epochs gain. Record all individual scores, mean R2, ranges, paired differences, and the predeclared noticeable threshold.

If fewer than two distinct non-baseline candidates exist, use fewer fits. Diverged runs are logged as failed, without fabricated scores. Do not expand beyond the bound based on test scores.

## Final artifacts and verification

Every run records full configuration, parent, seed, split/source hashes, installed versions, parameter counts, selected epoch, epochs run, update count, train/validation R2/MSE/RMSE/MAE, per-epoch histories, learning rates, runtime, process memory, and a saved checkpoint. Keep failures and losing runs in the log. Freeze final selection before one new model's test scoring. Reload the checkpoint to verify metrics and ensure BN inference does not update statistics. Package any validated better model separately under `tuning/best_model/`, with preprocessing, configuration, and inference instructions. The existing active `model/` remains untouched.

Produce a concise presentation-ready improvement log/table. If the noticeable criterion is met, prepare a short Quarto slide insert with the old and new validation metrics and the measured improvement. Do not add a misleading chain of individual effects when a winning bundle changes several controls. Keep the current clear presentation intact until the measured result is available.

## User amendment during confirmation

The user explicitly accepted the approximately 0.6222 mean-validation-R2 lower-L2 RMSprop result and instructed that any measurable R2 increase is sufficient. This supersedes the assistant's proposed +0.01 presentation threshold. Accept a positive mean paired validation gain with improvement in at least two of three seeds. The observed lower-L2 candidate improves in all three. This amendment does not change configuration ranking, checkpoint choice, training budgets, or test policy. `USER_CRITERION.json` records the instruction, and `finalize_user_result.py` applies it transparently after the original runner finishes. Preserve the original threshold result as an audit field.
