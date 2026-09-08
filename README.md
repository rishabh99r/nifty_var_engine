# Econometrically-Conditioned TFT for Nifty 1% VaR Forecasting

A reproducible research pipeline that forecasts next-day 99% Value-at-Risk for
the NIFTY50, BANKNIFTY and NIFTYIT indices with a **Temporal Fusion
Transformer (TFT)** conditioned on an econometric **Skew-t GJR-GARCH(1,1)**
volatility prior plus cross-market/domestic volatility features, benchmarked
against the GJR-GARCH model itself.

---

## ⚠️ Security note (read first)

Local MCP/agent configs (`.roo/`) are git-ignored and must **never** be
committed: they can embed live API keys. If a key was ever committed, revoke
it in the provider dashboard and scrub history with
`git filter-repo --path .roo/mcp.json --invert-paths`.

---

## Research question

> Can economically conditioned temporal deep learning extract incremental
> information from external and domestic volatility indicators **beyond a
> strong asymmetric GJR-GARCH volatility prior** when forecasting extreme
> downside quantiles?

### Core finding (S10 — honest framing)
The ablation evidence does **not** support "TFT beats GARCH". The defensible
result is:

- The **Skew-t GJR-GARCH benchmark is highly competitive** — a raw TFT does not
  significantly outperform it.
- Conditioning the TFT on `GARCH_sigma` alone does not help (it slightly
  *increases* the loss disadvantage on NIFTY50).
- **The Full ECTFT underperforms the purely GARCH-conditioned TFT** in 1% VaR
  accuracy: adding the cross-border macro features (`US_VIX_Diff`,
  `India_VIX_Diff`) introduces noise that **degrades** out-of-sample tail
  calibration on this sample.
- Statistical-power caveat (S9): with a 500-day backtest at α=1% there are
  only ~5 expected exceptions, so Kupiec / Christoffersen / DQ tests have
  **low power**. Results are framed as *"no evidence of failure"* — never as a
  proof of accuracy.

This narrative is used throughout the reports and figures.

---

## Reproducibility (S6)

1. **Python 3.12**.
2. `pip install -r requirements.txt` (the validated, mutually-compatible matrix).
3. Freeze the exact environment you validated on (Google Colab) with
   `pip freeze > requirements.lock` and commit that lock file.
4. Run `python environment_check.py` **before** any training run — it asserts
   `torch`, `pytorch-forecasting`, `lightning`, `arch`, `pandas`, `numpy`,
   `scipy` match the pinned matrix and hard-fails on a mismatch.

Frozen research cut-off: `config.RESEARCH_END_DATE = "2026-09-07"`.

---

## Pipeline

| Step | Command | Output |
|------|---------|--------|
| 1. Rebuild master panel (frozen end date) | `python build_data.py` | `master_df.csv` (PIT GARCH + features; tri-state convergence accounting) |
| 2. Baseline econometric proofs | `python proof.py` | Granger predictive-precedence diagnostics on **co-trading dates only** |
| 3. Train canonical 3-seed ensemble | `python main.py` | `test_tft_predictions*.csv`, `experiment_manifest.json` |
| 4. 6-arm ablation tournament | `python ablation_runner.py` | `ablation_tournament_results.csv`, `ablation_seed_level_results.csv`, `ablation_pairwise_dm.csv` (Holm-adjusted) |
| 5. Explainability (VSN + attention figures) | `python explainability.py` | `fig_vsn_feature_importance.png`, `fig_temporal_attention.png` |
| 6. Publication figures + audit report | `python generate_report_plots.py` | `report_fig*.png`, `model_validation_master_report.txt` |

All artifacts are mirrored into `config.OUTPUT_DIR` (the `GARCH_TFT_Results`
folder — Google Drive when mounted) alongside the seed checkpoints.

### Methodology guarantees
- **No `time_idx` feature** in any arm (canonical model == ablation Full ECTFT).
  Only bounded, window-relative `relative_time_idx` / `encoder_length` are used.
- **Point-in-time discipline**: rolling GJR-GARCH parameters are estimated on
  data strictly before each forecast day; the VaR backtest never sees future
  information.
- **Ensemble governance**: exactly `len(config.VALIDATION_SEEDS)` = 3
  checkpoints must be found; a partial ensemble **hard-crashes** rather than
  silently serving a 1- or 2-model forecast.
- **Granger alignment**: causality tests run on **inner-joined shared trading
  dates** of native-calendar log-differences — never on forward-filled values
  that would fabricate zero-returns.
- **No median-seed path**: the system is ensemble-only.

---

## Terminology
- "Regulatory-Inspired Binomial Zone" (not "Basel Traffic Light") — a
  sample-size-adapted binomial classification, not FRTB compliance.
- "Tail Exceedance Depth Diagnostic" (not "Expected Shortfall backtest") — a
  descriptive breach-depth statistic.
- "US VIX exhibits predictive precedence for domestic volatility" (not "US VIX
  causes India tail risk").

## Disclaimer
Research prototype for academic/educational use. Not investment advice; not a
regulated VaR engine.
