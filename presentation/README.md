# Current presentation

`Presentation.pdf` is the current 16-slide Beamer deck, with slide numbers. The editable source is `Presentation.qmd`.

From the project root:

```bash
python presentation/build_slides.py
quarto render presentation/Presentation.qmd --to beamer
```

All nine displayed seed-42 configurations now use a **40-epoch cap**. Early restoration steps evaluate final weights; step 6 introduces best-validation checkpoint restoration. Measured regressions are retained. Slide 12 compares the same seed-42 confirmation scores with three-seed means; slide 13 distinguishes those means from the packaged checkpoint’s validation/test scores. Slide 14 contains four separate RMSprop branches sharing the same 40-cap parent (R² 0.6210).

- `results_40_epoch.json`: current step-table inputs, from ten new fits and three reused confirmation fits.
- `results.json` / `results_20_epoch.json`: frozen historical 20-cap inputs, retained for reproduction and provenance.
- `non_improving_trials_40.json`: current four RMSprop branch comparisons; `non_improving_trials.json` retains the original five 20-cap candidates.
- `tuning_update.json`, `selection.json`, and `final_test.json`: unchanged confirmation/selected-model evidence.
- `brief_sequence.csv` / `BRIEF_SEQUENCE.md`: generated current progression.
- `all_runs.csv`: original 30-fit trial log; full expanded 42-fit log is `../tuning/all_runs.csv`.
- `../tuning/presentation_40/all_runs.csv`: full-precision follow-up metrics, including the three reused fits; request protocol, results, epoch histories, and checkpoints are in that folder.
- `quality_check.json`: visual and numeric checks for the latest PDF.

The selected model is `../tuning/best_model/`; `../model/` is the frozen Adam baseline and shared prediction code. No new test scoring or checkpoint replacement occurred during the 40-cap follow-up. Actual team contributions remain placeholders. Footnotes and navigation symbols are removed; important notes appear in the slide body.
