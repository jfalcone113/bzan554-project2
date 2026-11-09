# Expanded tuning study

`PLAN.md` freezes the search objective, source references, range choices, staged budget, and model/test selection rules. `run_tuning.py` executes at most 42 full-training fits in five adaptive stages with two independent processes. It imports the existing standalone model's data and metric utilities, without modifying the active checkpoint.

Run from the project root with the existing environment:

```bash
../.venv/bin/python tuning/test_tuning.py
../.venv/bin/python tuning/run_tuning.py
../.venv/bin/python tuning/finalize_user_result.py
../.venv/bin/python tuning/report.py
```

Each `requests/` file is immutable once created. Each `runs/*/` retains the full signature, train/validation metrics, and per-epoch history. Redundant trial checkpoints and console logs were removed after selection; the winning checkpoint remains in `best_model/`. The historical full-search resume state is no longer complete because trial checkpoints were removed. To reproduce the selected model, use `retrain_best.py --output tuning/retrained_best`; it rebuilds the deleted data cache automatically. A full search should use a fresh study output location. After each adaptive stage, its plan and all results are saved before proceeding. Do not edit the runner in the middle of a study and mix new code with completed signatures.

`all_runs.csv` and `epoch_metrics.csv` are generated from those records by `report.py`. `RESULTS.md` gives measured findings. The selected configuration is frozen in `selection.json` before any new candidate test scoring. `best_model/` contains the selected checkpoint, exact configuration, common training-only preprocessing, and metrics. The existing `model/` remains the baseline. The accepted improvements are integrated in the current presentation; the superseded slide insert was removed.

The test partition has historical exposure. Numeric metrics are in supplied scaled quantity units. All data remain local. The study tests dense architectures; the supplied notes and episode-level records do not establish a sequence-ready benchmark. See `environment.json` for locally checked API signatures and `verification_summary.json` for invariant-test evidence.

The user explicitly superseded the assistant-proposed +0.01 threshold with a measurable-improvement criterion during confirmation. `USER_CRITERION.json` records the instruction. Run `finalize_user_result.py` after the study and before the report to apply that criterion while preserving the original threshold outcome as an audit field. Configuration ranking and test policy remain unchanged.

## Presentation follow-up at cap 40

`presentation_40.py` reruns the six early restoration steps and four displayed RMSprop regression tests with cap 40, preserving their other controls and selection policies. Each rerun must reproduce the original epoch-20 validation R² within 0.00001. Results are in `presentation_40/`; `presentation_40_report.py` exports full numeric logs and prepares current slide inputs, reusing three already completed seed-42 confirmation fits. No new test scoring or model replacement occurs. Historical 20-cap logs stay unchanged.
