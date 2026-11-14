# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Snapshots of a single Kaggle notebook for the **Playground Series S6E11** competition (binary target `is_fraud`, ~1.7% positive rate, 699,960 train rows / 299,984 test rows, metric ROC AUC). Each `version N.ipynb` is a saved Kaggle notebook version, not a separate project:

- `version 2.ipynb`, `version 3.ipynb` — Steps 1–5: baseline features, deep LightGBM/XGBoost/CatBoost, fixed-weight 3-model blend.
- `version 5.ipynb` — Steps 1–11: adds digit features, nested target encoding, shallow trees, spline GLM, PyTorch MLP, 9-model weight-searched blend.
- `version 6.ipynb` — Steps 1–14: the latest. Adds `X_dig3` pairwise keys, a periodic-embedding WideResNet, 10-fold models, and a second-level meta-stacker.

Later versions are supersets of earlier ones (cells are appended, earlier cells lightly edited). **Make changes in the highest-numbered version**, or copy it to a new `version N.ipynb`; the older files are history. File names contain a space, so quote them in shell commands.

## Running

There is no build, lint, or test setup, and no `requirements.txt`. The notebooks are written for the Kaggle kernel and will not run locally as-is:

- Data is read from `/kaggle/input/competitions/playground-series-s6e11/` (`train.csv`, `test.csv`); it is not in the repo.
- Prediction caches are read from and written to `/kaggle/working/*.npz`.
- XGBoost (`device="cuda"`) and CatBoost (`task_type="GPU"`) are hard-coded to GPU from version 5 on; the PyTorch cells fall back to CPU.
- Dependencies are whatever the Kaggle Python image provides: pandas, numpy, scikit-learn, scipy, lightgbm, xgboost, catboost, torch.

The output is `submission.csv` (`id`, `is_fraud` probability), written by the final cell.

## Notebook architecture (version 6)

The notebook is one long linear script; every cell depends on globals created by earlier cells, so cells must run top to bottom.

**Resumable-cell convention.** Expensive cells are wrapped in `if "<name>_oof" not in globals():` so re-running a cell in a live kernel skips training. Step 1 additionally restores cached predictions from `/kaggle/working/preds_*.npz` into `globals()` so a fresh kernel can skip them too. Consequences:

- After changing a model's code, its `*_oof` / `*_test` globals (and the matching `.npz`) must be removed, or the stale predictions are silently reused.
- A new expensive model should follow the same pattern: guard on its OOF variable, `np.savez` its arrays at the end of the cell, and add the file to the restore list in Step 1.
- `preds_backup.npz` is read but never written by the notebook; it was created out of band.

**`SKIP_SUPERSEDED` (Step 1).** When `True`, the early models from Steps 2–5, 7 and 8 are not trained and their arrays are filled with zeros so later references still resolve. Those zero arrays are placeholders, not predictions; Step 14 only stacks models listed in `cand_models`.

**Prediction naming.** Every model produces a pair `<name>_oof` (out-of-fold predictions over train) and `<name>_test` (fold-averaged test predictions). All blending and stacking operates on these pairs only.

**Feature-set lineage.** Each set extends the previous one, and the same transform is always applied to the test counterpart (`X_test*`):

- `X` — raw columns plus `make_features` (ratios, `risk_flag_count`, interaction).
- `X_dig` — `add_digit_features`: cents / ones / tens digits and `log1p` of the six float columns.
- `X_dig2` — `add_finer_digits` plus frequency encoding computed over train+test.
- `X_dig3` — exact-tenth integers, `flags_bitmask`, group z-scores, and pairwise keys.

`text_cols` are the five categorical columns; they are kept as pandas `category` with test categories aligned to train.

**Leakage rules the code follows.**

- Target encoding is always *nested*: inside each outer fold, training rows are encoded by inner folds (`inner_skf`), while validation and test rows use the full outer-training fold. Keys are listed in `te_keys` / `nte2_keys` / `nte3_keys`; Step 9 onward uses a pre-factorized `np.bincount` implementation instead of pandas groupby.
- Frequency encodings and group z-scores may use train+test combined because they never touch the label.

**Cross-validation splits.** `skf` (5-fold, seed 0) is the split for Steps 2–12. Step 13 uses `skf10` (10-fold, seed 77). The Step 14 meta-stacker uses its own `meta_skf` (10-fold, seed 2026). OOF arrays from different splits are stacked together; this is intentional.

**Final ensemble (Step 14).** A static blend (random Dirichlet weight search plus coordinate `polish_weights`, in both probability and logit space) is combined with two second-level models trained on base-model logits plus context features — a logistic regression and a shallow LightGBM — and the three are weighted by grid search on OOF AUC.

Model selection throughout is by OOF AUC printed in each cell; the notebook's saved outputs are the record of those scores.
