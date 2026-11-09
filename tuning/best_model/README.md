# Improved selected quantity model

This bundle is the winner of three-seed, equal-budget confirmation. The selected checkpoint uses seed **202**. `selection.json` records its exact epoch, scores, selection rule, and the user's acceptance criterion. `config.json` records the training architecture and hyperparameters.

It uses RMSprop at initial rate 0.001 and hidden-kernel L2 **0.00001**, retaining three sigmoid layers (256/128/64), hidden dropout 5%, SKU dropout 50%, and a linear output. The broader activation, initialization, depth/width, BN, and batch experiments did not improve enough to win confirmation; their full results remain in `../all_runs.csv`.

Keep `model.keras`, `preprocessing.json`, and `seen.npz` together. The generic prediction/verification code in the project's `model/pricing_model.py` supports this saved architecture without rebuilding the baseline.

From the project root:

```bash
../.venv/bin/python model/pricing_model.py verify --bundle tuning/best_model
../.venv/bin/python model/pricing_model.py predict --bundle tuning/best_model --input data/pricing.csv --output /tmp/improved_predictions.csv
../.venv/bin/python tuning/retrain_best.py --output tuning/retrained_best
```

The last command trains only this frozen configuration from scratch. It does not repeat the full search and never overwrites an existing fitted bundle. The custom checkpoints contain inference architecture/weights, not restorable optimizer state.

R2 is selected using validation, not test. Test scoring followed frozen configuration and seed choice, but this partition has historical exposure. Errors use supplied scaled quantity units. `verification.log` confirms that the packaged checkpoint reproduces its saved metrics through the normal prediction/evaluation code.
