# Phase 3 Implementation Plan — Reviewer-Driven Hardening of the ECTFT VaR Pipeline

**Date:** 2026-09-08 (research cutoff for the frozen final run)
**Status: IMPLEMENTED (code complete + compiling 2026-09-08).** Phase 4 (GPU
regeneration + smoke run) remains pending on a torch/Colab environment.
**Scope:** Phase 1 (statistical/methodology), Phase 2 (6-arm tournament), Phase 3 (deployment hardening), plus reproducibility extras.

---

## 0. Design decisions (review-response mapping)

The reviewer's verdict and plan have been reconciled with the actual code. Decisions:

| # | Review finding | Decision | Where |
|---|----------------|----------|-------|
| 1 | `time_idx` present in canonical model but absent in ablation → different models | **Remove `time_idx` from the canonical feature lists.** Retain `relative_time_idx` + `add_encoder_length` (window-relative, not an absolute calendar counter). | `tft_model.py`, `config.py` (comment), `main.py` |
| 2 | Kupiec N=0 fix is not standard Kupiec | **Rewrite boundary to the standard LR test**: `LR = -2T ln(1-alpha)` with a numerically stable two-sided p-value. Note: standard Kupiec *rejects* an over-conservative zero-breach model (calibration too conservative), which is the correct statistical meaning. | `metrics.py:kupiec_pof_test` |
| 3 | Multiple testing (3 assets × variants) | **Add `holm_bonferroni()`** to `metrics.py`; pre-specified family: the direct pairwise DM comparisons (Full-vs-GARCH-TFT, Full-vs-GARCH, US-only-vs-GARCH-TFT, India-only-vs-GARCH-TFT) within the tournament, Holm-corrected per comparison family. | `metrics.py`, `ablation_runner.py` |
| 4 | Ensemble quantile crossing not audited | **Add `audit_quantile_monotonicity()`** to `metrics.py`; call on each seed AND on the averaged ensemble output. | `metrics.py`, `ablation_runner.py`, `tft_model.py` |
| 5 | GARCH "converged" counter actually counts only exceptions | **Inspect `res.convergence_flag == 0`**; maintain tri-state accounting: `converged / non_converged / exceptions`. | `build_data.py`, `proof.py` |
| 6 | Ablation only 3 TFT arms; macro vars are two different things | **Expand to 6-arm tournament** (see §2). | `ablation_runner.py` |
| 7 | Only ensemble results reported; reviewer wants seed sensitivity | **Emit per-seed result table** (breaches/Kupiec/DM for seeds 42/123/777) alongside ensemble. | `ablation_runner.py` |
| 8 | Deployment `tft.predict()` unpacking incompatible with PTF ≥0.9 `Prediction` namedtuple | **Replicate `unpack_predictions()`** (already in `ablation_runner.py`) into `production_engine.py` + any `deployment.py` call sites. | `production_engine.py`, `deployment.py` |
| 9 | Local checkpoint discovery misses `checkpoints/`; lexical `glob[0]` fragile | **Add `"checkpoints"` to search dirs** AND **write/read `experiment_manifest.json`** with seed, val_loss, epoch, git commit, dataset cutoff, feature spec, checkpoint path. | `tft_model.py`, `deployment.py` |
| 10 | Stale-artifact purge incomplete | **Add** `test_tft_predictions_panel*.csv`, `ablation_ens_panel_*.csv`, `ablation_tournament_results.csv`, `checkpoints/*.ckpt` to `purge_stale_artifacts()`. | `build_data.py` |
| 11 | Monkey-patching in ablation runner | **Refactor** `build_datasets(df, known_features=None, unknown_features=None, ...)` and `train_tft(..., known_features=None, unknown_features=None)` to take explicit feature lists. | `tft_model.py`, `main.py`, `ablation_runner.py` |
| 12 | `END_DATE = today()` non-reproducible | **Freeze** to `RESEARCH_END_DATE = "2026-09-07"`; default `END_DATE` = research cutoff for the final manuscript run. Keep a separate dynamic constant only for the production path. | `config.py` |

---

## 1. Phase 1 — Core architectural & statistical fixes

### 1.1 `config.py`
- Add `RESEARCH_END_DATE = "2026-09-07"` (frozen).
- Set `END_DATE` to the frozen research cutoff for research reproducibility; the deployment path can pass its own dynamic end date explicitly.
- No `time_idx` literal lives in config (candidates are in `tft_model.py`), but update any header comments that describe `time_idx` as a known feature.

### 1.2 `tft_model.py`
1. **`build_datasets(df, encoder_length=None, backtest_days=None, val_days=None, known_features=None, unknown_features=None)`**
   - Default `known_features=None` → canonical list **without `time_idx`**:
     `["GARCH_sigma", "US_VIX_Diff", "India_VIX_Diff"]` (only columns present in `df`).
   - Default `unknown_features=None` → `["Log_Ret_Feature", "GK_Vol"]`.
   - Drop the hard-coded `candidate_known`/`candidate_unknown` blocks; filter the passed lists against `df.columns`.
   - Keep `add_relative_time_idx=True`, `add_encoder_length=True` (window-relative).
   - Raise `ValueError` if a requested feature is absent from `df` (no silent drop) — except we must tolerate the India-VIX-proxy fallback case, so only warn for the known `India_VIX_Diff` absence path; document.
2. **`train_tft(df, ..., known_features=None, unknown_features=None)`** → forward to `build_datasets`.
3. **Checkpoint helpers**: extend search dirs to `("checkpoints", ".", config.OUTPUT_DIR)`.
4. **Manifest writer** `write_experiment_manifest(seed, ckpt_path, val_loss, epoch, known_features, unknown_features)` → writes `experiment_manifest.json` (append/overwrite entry per seed). Fields: `git_commit` (via `subprocess git rev-parse HEAD`, fallback `"unknown"`), `dataset_cutoff` (`config.END_DATE`), `feature_spec`, `seed`, `val_loss`, `best_epoch` (from checkpoint filename or `trainer.current_epoch`), `checkpoint_path`, `timestamp`.
5. Reuse for the per-seed quantile crossing audit within `generate_and_save_predictions` (already present for single models — keep; ensemble audit goes in callers).

### 1.3 `metrics.py`
1. **`kupiec_pof_test`** — replace the `N == 0` branch with the standard LR:
   - `lr_uc = -2.0 * T * np.log(1.0 - alpha)`
   - p-value via `1 - chi2.cdf(lr_uc, 1)`. For T=500, α=0.01 → LR≈10.05, p≈0.0015 (correctly flags over-conservatism). Keep the `N==T` (all-breach) symmetric boundary with a numerical guard, and document both.
2. **`holm_bonferroni(p_values)`** → returns list of adjusted p-values in the same order (sorted ascending, `p_adj_i = max((n-i+1)*p_i)`, enforce monotone non-decreasing, cap at 1.0). Accept list/array + return np.array.
3. **`audit_quantile_monotonicity(df_or_arrays, q_cols=("q01","q50","q99"))`** → returns `{total_rows, violations_lo, violations_hi}` and prints/asserts optionally. Used on seed + ensemble panels.

### 1.4 `main.py`
- Pass explicit `known_features=config_canonical` (import from `tft_model` defaults, or call `train_tft` with defaults → same). Minimal change: rely on new defaults (they now exclude `time_idx`), so canonical model == tournament Full ECTFT.

### 1.5 `build_data.py`
- **`rolling_gjr_garch_pit`**: replace the bare `except Exception: refit_failures += 1` block with tri-state:
  ```python
  try:
      current_res = am.fit(disp="off", show_warning=False)
      if int(getattr(current_res, "convergence_flag", 0)) != 0:
          refit_non_converged += 1      # parameters still updated? -> policy below
      else:
          refit_converged += 1
  except Exception:
      refit_exceptions += 1
      # carry forward previous params
  ```
  **Policy decision (needs your confirmation):** when `convergence_flag != 0` (optimizer did not converge) but parameters were returned, do we (a) **reject and carry forward** previous params (conservative; treats non-convergence as failure), or (b) **accept but count** (current behavior effectively)? I recommend (a) reject-and-carry-forward, since the whole point is PIT honesty — but I'll flag this as a decision point.
- Report line updated to: `attempted X, converged Y, non-converged Z, exceptions W (previous params retained on non-convergence/exception)`.
- **`purge_stale_artifacts()`**: add `test_tft_predictions_panel.csv`, `glob("test_tft_predictions_seed_*.csv")`, `glob("test_tft_predictions_panel_seed_*.csv")`, `ablation_tournament_results.csv`, `glob("ablation_ens_panel_*.csv")`, and `glob("checkpoints/*.ckpt")`.

### 1.6 `proof.py`
- Full-sample descriptive GJR-GARCH fits: check `res.convergence_flag == 0`, count and report; relabel output as **full-sample descriptive GJR-GARCH estimates** distinct from the PIT forecasts (reviewer §27). Read `proof.py` fully before editing (not yet read in full).

---

## 2. Phase 2 — The definitive 6-arm tournament

### 2.1 New `ABLATION_CONFIGS` (arm 0 is GJR-GARCH benchmark)

| Arm | known (time-varying known reals) | unknown | Label |
|-----|----------------------------------|---------|-------|
| 0 | — (GJR-GARCH benchmark, not a TFT) | — | Skew-t GJR-GARCH |
| 1 | `[]` | `Log_Ret_Feature, GK_Vol` | Raw TFT |
| 2 | `GARCH_sigma` | `Log_Ret_Feature, GK_Vol` | GARCH-TFT |
| 3 | `GARCH_sigma, US_VIX_Diff` | `Log_Ret_Feature, GK_Vol` | +US cross-border |
| 4 | `GARCH_sigma, India_VIX_Diff` | `Log_Ret_Feature, GK_Vol` | +India domestic |
| 5 | `GARCH_sigma, US_VIX_Diff, India_VIX_Diff` | `Log_Ret_Feature, GK_Vol` | Full ECTFT |

### 2.2 `ablation_runner.py` rewrite
- **Remove** `inject_custom_datasets`/`restore_build_datasets` entirely. Call `train_tft(df=master_df, seed=seed, known_features=cfg["known"], unknown_features=cfg["unknown"], enable_progress_bar=False)`.
- Keep `unpack_predictions()` (moved to a shared helper imported from `production_engine` or a small new `predict_utils.py` — decision: create **`predict_utils.py`** so `ablation_runner`, `production_engine`, and `deployment` import one canonical implementation; avoids three copies).
- **Per seed**: save q01 AND q50 AND q99 columns (currently only q01). Needed for the ensemble crossing audit. Store `TFT_VaR_99_Raw`, `TFT_Median`, `TFT_VaR_Upside` per seed.
- **Ensemble**: average all three quantile columns across seeds, then run `audit_quantile_monotonicity()` on the **averaged** output (reviewer §9).
- **Outputs**:
  1. `ablation_tournament_results.csv` — per-arm × per-asset ensemble breaches/Kupiec/DM vs GJR-GARCH.
  2. `ablation_seed_level_results.csv` — per-arm × per-asset × seed(42/123/777) breaches/Kupiec/DM.
  3. **Direct pairwise DM table** `ablation_pairwise_dm.csv` for the pre-specified comparisons: Full-vs-GARCH-TFT, Full-vs-GARCH, US-only-vs-GARCH-TFT, India-only-vs-GARCH-TFT, each per asset, with **Holm-corrected p-values** applied within the 3-asset family per comparison.
- Preserve the GJR-GARCH benchmark arm 0 evaluation on the same OOS window.

---

## 3. Phase 3 — Deployment hardening

### 3.1 Shared PTF unpacking
- New module **`predict_utils.py`**:
  ```python
  def unpack_predictions(result):
      if hasattr(result, "output") and hasattr(result, "index"): ...
      elif isinstance(result, (tuple,list)) and len(result) >= 2: ...
      else: raise TypeError(...)
  ```
- `production_engine.py`: replace `preds, index_df = tft.predict(...)` (2 sites: `run_live_daily_inference` and `run_live_ensemble_inference`) with the shared helper.

### 3.2 Checkpoint manifest
- `tft_model.py`: `write_experiment_manifest(...)` invoked inside `train_tft` after the best checkpoint is chosen (or in `main.py` after each seed). Manifest lives at repo root: `experiment_manifest.json`.
- `deployment.py` + `production_engine.py` `__main__` / `select_seed_checkpoints`: prefer manifest entries; fall back to glob only with an explicit warning.
- `select_seed_checkpoints(seeds)` and `select_median_checkpoint` in `tft_model.py`: search dirs `("checkpoints", ".", config.OUTPUT_DIR)`.

### 3.3 purge + cadence
- Extend `purge_stale_artifacts()` (see 1.5).
- Clarify docstring of `deployment.py`: it is a manual-trigger full-data-reconstruction with cadence bookkeeping, not a stateful GARCH production engine (reviewer §26) — comment-only change.

---

## 4. Phase 4 — Regenerate & smoke test (Colab / torch env)
- `python build_data.py` (rebuild master_df with frozen cutoff + tri-state GARCH accounting).
- `python proof.py`, then `python main.py` (canonical 3-seed, no-time_idx), then `python ablation_runner.py` (6 arms).
- Confirm: canonical Full ECTFT breaches ≈ ablation Full ECTFT (models now identical).
- Verify no quantile crossings, DM tables with Holm columns, seed-level CSVs exist, manifest written.

---

## 5. Phase 5 — Report/figure updates
- `generate_report_plots.py`: terminology (skew-t GJR-GARCH benchmark; binomial coverage zone); consume no-time_idx ensemble panel; regenerate final figures under frozen `RESEARCH_END_DATE`.
- `plans/nifty_var_review.md` and the draft docx: mark as historical provenance; append Phase-3 results once run.

---

## Resolved decisions (user-approved 2026-09-08 — recommended defaults adopted)
1. **GARCH non-convergence policy**: reject-and-carry-forward — when `res.convergence_flag != 0`, treat the refit as non-converged, RETAIN the previous parameters, and count it in a separate non-converged bucket.
2. **Feature-absence tolerance**: warn-and-continue ONLY for the documented `India_VIX_Diff` RV-proxy fallback path; hard-raise `ValueError` for any other requested feature missing from `df`.
3. **Manifest writer location**: `experiment_manifest.json` is written from INSIDE `train_tft()` (it alone knows the best val_loss + checkpoint path); all callers inherit it automatically.
4. **Holm-Bonferroni family scope**: correction is applied within each pre-specified comparison family of 3 assets, transparently reported as "Holm-adjusted within the 3-asset family."

Dependency order: config → metrics → tft_model/build_data/proof → main/ablation → production_engine/deployment → plots/docs.

---

## 6. Post-plan additions (approved during implementation, 2026-09-08)

### 6.1 Reproducibility environment
- Added **`requirements.txt`** pinning a mutually compatible matrix:
  `numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 matplotlib==3.8.4
  scikit-learn==1.4.2 torch==2.2.2 lightning==2.2.5
  pytorch-forecasting==1.0.0 arch==7.0.0 statsmodels==0.14.2 yfinance==0.2.41`.
  This is the exact stack under which the final manuscript run must execute.

### 6.2 Explainability figures (reviewer evidence)
- **`fig_vsn_feature_importance.png`** — horizontal bar of encoder VSN selection
  weights (mean +/- std across the 3 seeds) proving the network allocates
  material weight to `GARCH_sigma` and `US_VIX_Diff` (not pure overfit to the
  autoregressive return channel).
- **`fig_temporal_attention.png`** — mean attention weight per lookback lag
  with a RED reference line at uniform attention (1/21 = 4.76%) so any t-1/t-2
  recency concentration is judged against the flat baseline (Reviewer #20).
- Both figures + their aggregated CSVs + the text report are written to
  workspace AND `config.OUTPUT_DIR`.

### 6.3 Drive artifact persistence (GARCH_TFT_Results)
- `ablation_runner.py` now mirrors every output (`ablation_tournament_results.csv`,
  `ablation_seed_level_results.csv`, `ablation_pairwise_dm.csv`, each
  `ablation_ens_panel_*.csv`) into `config.OUTPUT_DIR` via `_mirror_to_output_dir()`.
- `explainability.py` writes figures/CSVs/report into `config.OUTPUT_DIR`
  (the existing `GARCH_TFT_Results` folder, Google Drive when mounted).

### 6.4 Final reproducibility commands (Colab / torch env)
1. `pip install -r requirements.txt`
2. `python build_data.py`   (frozen RESEARCH_END_DATE = 2026-09-07, tri-state GARCH accounting)
3. `python proof.py`        (full-sample descriptive fits + convergence flags)
4. `python main.py`         (canonical 3-seed no-time_idx ensemble + manifest)
5. `python ablation_runner.py`  (6-arm tournament; per-seed + ensemble + Holm pairwise tables)
6. `python explainability.py`   (VSN + attention figures)
7. `python generate_report_plots.py` (final publication figures + audit report)

All artifacts land both locally and under the mounted Drive `GARCH_TFT_Results`.
