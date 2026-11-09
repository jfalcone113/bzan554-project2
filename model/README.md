# Self-contained selected model

`model.keras`, `preprocessing.json`, and `seen.npz` form the prediction bundle. `pricing_model.py` supplies standalone training, prediction, evaluation, and verification commands. No old experiment folder is needed.

- `config.json`: exact selected settings and training budget.
- `metrics.json`: reference train/validation/test metrics in supplied quantity units.
- `epochs.json`: the original 20-epoch training history. Epoch 15 supplies the saved weights.
- `training.json`: checkpoint provenance, parameter count, timing, and checksum.
- `verification.json`: full-row prediction equivalence with the original checkpoint.
- `reproduction.json` and `reproduction.log`: independent from-scratch training verification.

See the project README for commands. The checkpoint does not include restorable optimizer state. For predictions, never use its raw standardized output without the bundled preprocessing and target inverse transformation.
