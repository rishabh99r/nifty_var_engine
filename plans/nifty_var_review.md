# Rigorous Code Review: Nifty 50 VaR / Volatility Project

**Review scope:** All Python modules in the workspace, cross-verified against `results/` artifacts (two `model_validation_master_report*.txt`, two `regulatory_test_suite_results*.csv`, and dashboard/report PNGs).

**Verdict (executive):** The **temporal PIT design of the rolling GJR-GARCH filter is largely correct** — a genuine strength. However, the project has **several statistical, engineering, and reporting defects** that (a) overstate the fat-tail/leverage evidence, (b) create a **validation-vs-production mismatch**, (c) include **dead and inconsistent modules**, and (d) show **seed-dependent results presented without aggregation**. On its own statistical evidence (all Diebold–Mariano tests insignificant), **TFT does not outperform GJR-GARCH**, yet the project is framed as a hybrid model that does. **Not publishable in current form.**

---

## 1. Data Ingestion & Feature Engineering

### 1.1 [`build_data.py`](build_data.py) — the live data path
- Downloads `^NSEI`, `^NSEBANK`, `^CNXIT`, `^VIX` from 2015-01-01 to 2026-08-01. **This is the dataset actually used by the model.**
- Garman–Klass volatility proxy is computed correctly (log H/L, log C/O).
- **`India_VIX_Diff` is a misnomer.** It is NOT an India VIX; it is `Log_Ret.rolling(5).std().diff().shift(1)` — a lagged first-difference of a 5-day realized-volatility proxy. The report frames its Granger test as "US VIX → **India Volatility**", which is defensible only if the feature is honestly relabeled as *domestic realized-volatility proxy*. Using the string "VIX" in the variable name and the report is **scientific mislabeling**.
- **`US_VIX` is the US VIX (`^VIX`)**. The project never ingests an actual India VIX; the "cross-border" narrative is US VIX vs a domestic *realized-vol* proxy.

### 1.2 [`data_loader.py`](data_loader.py) — **dead, and dangerous if resurrected**
- Not imported by `main.py`, `tft_model.py`, or `generate_report_plots.py` (confirmed by search). It is orphaned.
- **`fetch_cpu_index()` fabricates a constant `Global_CPU = 100.0` series** when no `CPU_index.csv` exists, and `Global_CPU_Ret` ingests this constant directly. If this module were ever wired into the pipeline, it would inject a **meaningless constant feature** that downstream Granger/importance analyses could silently absorb. It must be deleted or made to fail loudly.

### 1.3 Configuration drift
- [`config.py`](config.py:20) declares `START_DATE = "2007-01-01"` and `TEST_START_DATE = "2025-01-01"`, but [`build_data.py`](build_data.py:132) hardcodes 2015 → 2026-08. The config is **not the source of truth** and is misleading to any reader/reviewer.

---

## 2. Rolling GJR-GARCH Point-in-Time Filter ([`build_data.py:68`](build_data.py:68))

**Assessment: the causal design is correct.** This is the strongest part of the codebase.

- Parameters re-estimated every `refit_freq=21` days on `train_slice = returns[t-lookback : t]` — **strictly F_{t-1}-measurable** (window ends at t-1). ✓
- The daily recursion uses only the t-1 shock and t-1 variance:
  `σ²_t = ω + (α + γ·1[ε_{t-1}<0])·ε²_{t-1} + β·σ²_{t-1}` — correct GJR-GARCH(1,1) asymmetric recursion. ✓
- `VaR_99[t] = μ + σ_t · Q_skewt(0.01)` with the skew-t quantile from the last refit. ✓ F_{t-1}-measurable.
- Fat-tail handling here is genuine: `dist='skewt'` captures both heavy tails and asymmetry. ✓

### Defects in this block
1. **`last_q_dist` is frozen between refits** — the quantile multiplier is only refreshed every 21 days. Acceptable, but worth noting the VaR boundary is step-wise constant in the skew parameter between refits.
2. **Quantile extraction `current_res.model.distribution.ppf(0.01, current_res.params[-2:])`** assumes the last two params are `(nu, lambda)` in that exact order. This is fragile and order-dependent; if the arch package changes the parameter layout it silently corrupts the VaR floor. Should be keyed by parameter name.
3. **GARCH_resid is a contemporaneous-return encoding:** `resid_arr[t] = (r_t − μ)/σ_t` embeds the *current* return. When fed to the TFT encoder at position t, this is effectively a scaled lagged return (causal, not a temporal leak), but it is **redundant** with `Log_Ret_Lag1` and inflates the perceived information available. See §3.2.

---

## 3. TFT Volatility Pipeline ([`tft_model.py`](tft_model.py))

### 3.1 Split construction ([`build_datasets`](tft_model.py:27))
- Temporal split: train `≤ val_cutoff`, val `(val_cutoff − enc, test_cutoff]`, test `> test_cutoff − enc`.
- The `−encoder_length` overlap on val/test encoder windows is **standard and correct** (encoder windows must be contiguous into the past).
- `stop_randomization=True`, `predict=False` on eval sets. ✓
- **No temporal leakage in the split design.** Credit where due.

### 3.2 The real concern: feature/target design
- `target = "Log_Ret"` **and** `"Log_Ret"` is also listed in `time_varying_unknown_reals`. Listing the target as a covariate is an anti-pattern that invites silent model misuse.
- `GARCH_resid` and `GK_Vol` are near-duplicate, return-derived encoder signals. Combined with `Log_Ret_Lag1/Lag2`, the encoder is heavily dominated by lagged-return information. The **VSN "importance"** analysis then partly measures redundancy, not economic signal.
- `GARCH_sigma` is classified as a **known** (future-known) feature. This is legitimate here because σ_t only needs t−1 data, but a reviewer must confirm it is genuinely observable at forecast time — it is, but only because of the discipline enforced in `build_data.py`. Any future change to the GARCH timing breaks this silently.

### 3.3 Checkpoint / "champion" handling
- The champion config is **hardcoded** (`hidden_size=64, dropout=0.30, lr=0.001552`) in [`main.py`](main.py:15) and [`tft_model.py`](tft_model.py:237) — it is not re-derived from a saved HPO study artifact. A reviewer cannot audit *how* these specific hyperparameters were chosen (search grid, iterations, budget).

---

## 4. Selection Bias / Seed Handling — **REPRODUCIBILITY RED FLAG**

- [`main.py`](main.py:40) trains **3 seeds** and averages only the **NIFTY50** breach/DM numbers.
- But the published report tables ([`results/regulatory_test_suite_results*.csv`](results/regulatory_test_suite_results.csv)) show **a single seed's** full-panel results.
- The **two result sets disagree materially**:
  - NIFTY50 breaches: **3** (report A) vs **5** (report B)
  - NIFTYIT breaches: **7** (GREEN) vs **9** (YELLOW)
  - NIFTYIT Basel zone flips **GREEN → YELLOW** across runs.
- **The report does not disclose that these are seed-dependent or which seed produced the shown table.** Presenting one seed's panel table as "the result" without reporting the across-seed distribution is **selection/cherry-picking** and is a top-blocker for publication.

---

## 5. Backtesting & Statistical Metrics ([`metrics.py`](metrics.py))

The metric implementations are mostly standard and correct:
- Kupiec POF, Christoffersen independence, LR-CC combination, Basel traffic light, Engle–Manganelli DQ, and Diebold–Mariano with Newey–West HAC — all correctly specified.

### Statistical weaknesses
1. **Underpowered backtest:** 500 test days at α=1% → expected ≈ **5 breaches**. All point estimates and Basel zones are computed on ~3–9 exceptions. The **Basel traffic light has negligible power** at this sample size; GREEN here is not strong evidence of model adequacy.
2. **Internal contradiction exposed by the data:** BANKNIFTY shows **Kupiec p = 0.0282** (rejects correct unconditional coverage at 5%) while being labeled **GREEN**. The traffic light (cumulative binomial) and the LR test tell different stories; the report surfaces only the GREEN label.
3. **DM tests are all insignificant.** Reported `DM p-values`: NIFTY50 0.093, BANKNIFTY 0.388, NIFTYIT 0.995 (report A). **TFT does NOT statistically beat GJR-GARCH** on the project's own loss comparison. The headline "hybrid TFT superior" is **not supported by the evidence**.
4. **Christoffersen returns p=1.0 when <2 hits** — honest low-power handling, but it means independence is *not testable* for BANKNIFTY (1 breach).
5. No **Expected Shortfall**, no backtest at 95%/97.5%/99.5% levels, no **PIT/CES density evaluation**. "Fat-tail handling" is asserted via a skew-t GARCH fit and quantile loss, but never rigorously validated as a *density*.

---

## 6. Fat-Tail Methodology Assessment

**What is done well:**
- Skew-t innovations in the GARCH filter — correct for leptokurtic, asymmetric equity returns.
- TFT trained with `QuantileLoss([0.01, 0.5, 0.99])` — distribution-free direct quantile regression, appropriate for fat tails.

**What invalidates the fat-tail evidence:**
1. **`df = 0.00` reported for ALL three series** in both validation reports (`results/model_validation_master_report.txt:19-21`). This comes from `res.params.get('nu', 0)` in [`generate_report_plots.py:134`](generate_report_plots.py:134). A skew-t degrees-of-freedom of 0.00 is **impossible** — the tail-df is not being read correctly. **The central "fat tail" claim is therefore unverifiable from the shipped artifacts**, and the extraction bug is trivially reproducible from the code.
2. The three reported GARCH parameter sets are **identical across the two runs** (`omega/alpha/gamma/beta` match to 5 decimals) even though the breach counts differ. This is because the news-impact fits are full-sample in-sample fits, independent of seed — so the "fat-tail parameter evidence" is **not seed-averaged and not linked to the validated model**.

---

## 7. Production vs. Validated Logic Mismatch — **CRITICAL**

- In [`generate_and_save_predictions`](tft_model.py:199), the GARCH circuit breaker is **commented out**:
  ```python
  # merged_panel["TFT_VaR_99"] = np.minimum(...GARCH floor...)
  merged_panel["TFT_VaR_99"] = merged_panel["TFT_VaR_99_Raw"]   # pure TFT
  ```
  So the **backtested/validated** VaR is the **raw, unconstrained TFT** quantile.
- But [`production_engine.py:83`](production_engine.py:83) **does** apply the floor:
  ```python
  final_var_99 = min(raw_tft_var_99, garch_floor_var)
  ```
- **The model that was backtested is NOT the model that is deployed.** The deployed "hybrid" (GARCH floor) behavior has **zero out-of-sample validation**. The report nonetheless labels the line "Hybrid TFT" — a misrepresentation of what was tested.
- The `circuit_breaker.py` module itself is **dead code** — it is imported nowhere in the project (confirmed by search: only self-references). Its rules (`|z|>4` override, positive-VaR clamping) are not part of any executed pipeline, yet the project name/framing implies a regulatory circuit breaker is active.

---

## 8. Granger Causality — **Spurious-Significance Risk** ([`generate_report_plots.py:146`](generate_report_plots.py:146), [`proof.py`](proof.py))

- The test uses `India_VIX_Diff = Log_Ret.rolling(5).std().diff()`, a **heavily overlapping rolling statistic**.
- Overlapping-window transforms induce strong serial correlation **in the feature itself**, which inflates Granger test rejection rates (near-zero p at 5-day lag, e.g. `p=0.0000`). **The near-zero p-values are largely an artifact of overlapping windows**, not evidence of true cross-border transmission.
- The test is **in-sample** (full sample), so it is descriptive, not predictive, and cannot be cited as out-of-sample support.
- NIFTYIT shows `1D lag p = 0.86, 2D lag p = 0.27` — the "spillover" is not even consistent across series, undermining the cross-border narrative.

---

## 9. Code Hygiene / Engineering Issues

- [`hpo.py:10`](hpo.py:10) imports `from metrics import quantile_loss`, but **`quantile_loss` does not exist** in [`metrics.py`](metrics.py) (only `pinball_loss`). The HPO module **crashes on import** → it is dead/broken code, and the "champion" hyperparameters cannot have come from it as written.
- `config.py` HIDDEN_SIZE=32 conflicts with champion 64; EPOCHS=100 vs 80; LEARNING_RATE=0.0018 vs 0.001552 — config and hardcoded values disagree.
- Data (`master_df.csv`), checkpoints (`*.ckpt`), and prediction CSVs are all **gitignored** (`results` also not committed as CSVs). The entire pipeline is non-reproducible from the repo because the artifacts and the data-builder output are excluded, and `yfinance` data is not versioned.

---

## 10. Publishability Assessment (as a finance research paper)

**Verdict: NOT publishable in current form.** Blocker-level issues:

| # | Issue | Severity |
|---|-------|----------|
| 1 | **DM tests insignificant** — no statistical evidence TFT beats GJR-GARCH; the central claim is unsupported | **Blocker** |
| 2 | **Seed-dependent results with no aggregated/uncertainty reporting**; two shipped result sets disagree; one seed cherry-picked into the table | **Blocker** |
| 3 | **Validation/deployment mismatch** — backtested pure-TFT vs deployed GARCH-floor "hybrid" | **Blocker** |
| 4 | **Tail-df reported as 0.00** — fat-tail evidence unverifiable from artifacts | **Blocker** |
| 5 | **"India VIX" is actually a realized-vol proxy** — feature mislabeling in a headline variable | **Blocker** |
| 6 | Granger significance inflated by overlapping rolling windows | Major |
| 7 | Dead/broken modules (`hpo.py` ImportError, `data_loader.py` constant-CPU fabrication, `circuit_breaker.py` orphaned) | Major |
| 8 | No Expected Shortfall, no multi-level VaR backtest, no density/PIT validation | Major |
| 9 | Underpowered 500-day backtest; Basel GREEN is weak evidence | Major |
| 10 | Non-reproducible repo (data/ckpt/predictions gitignored) | Major |

**What a revision must do to be credible:**
1. **Report across-seed distributions** (mean ± std of breaches, Kupiec, DM across all 3 seeds for the full panel), and stop picking a single favorable seed.
2. **Make the GARCH floor consistent everywhere** — either apply it in validation too, or drop the "hybrid" framing and report pure-TFT honestly.
3. **Fix the tail-df extraction** and show the actual estimated skew-t degrees of freedom with standard errors.
4. **Relabel the domestic feature** as a realized-volatility proxy and stop calling it "India VIX"; drop or fix the overlapping-window Granger test (use non-overlapping measures and PIT/exogenous-window causality).
5. **Add density validation** (PIT histogram, expected shortfall, backtests at 95/97.5/99.5%) and an honest **economic-significance** analysis given the insignificant DM tests.
6. **Repair or remove dead modules** (`hpo.py`, `data_loader.py`, `circuit_breaker.py`), reconcile `config.py` with the pipeline, and commit reproducible artifacts/seed config.
7. Given DM insignificance, reposition the paper honestly (e.g., "does a TFT add value over a well-specified skew-t GJR-GARCH?" with a **negative/qualified** result), which — if done rigorously — is itself publishable.

**Positive takeaways worth preserving:** the PIT-disciplined rolling GJR-GARCH filter, the correctly-specified temporal splits, the standard battery of VaR backtests, and the direct quantile-loss objective are all methodologically sound foundations.

---

# PART 2 — REFACTOR IMPLEMENTATION & POST-FIX PUBLISHABILITY

## 2.1 What was refactored (and where)

| Proposed fix | Implementation | File(s) |
|---|---|---|
| Narrative pivot (no superiority claim) | Repositioned as "Econometrically-Conditioned TFT"; added McNeil–Frey **Expected Shortfall** backtest as the tail-shape dimension | `metrics.py`, `tft_model.py`, `production_engine.py`, `generate_report_plots.py` |
| Validation == deployment | Removed GARCH floor from `production_engine.py`; backtested/exported VaR is the raw TFT quantile; GARCH is an **input prior** through the VSN, not an output override | `tft_model.py`, `production_engine.py` |
| Dead `circuit_breaker.py` | **Deleted** (orphaned, contradicted narrative) | deleted |
| India VIX mislabeling | Ingest actual `^INDIAVIX`; honest `Domestic_RV_Proxy` fallback gated by `MIN_INDIA_VIX_OBS`; provenance file written; **never mislabeled** | `config.py`, `build_data.py` |
| Spurious Granger | Uses **actual VIX daily log-differences** (`US_VIX_Level`/`India_VIX_Level`), no overlapping rolling windows | `build_data.py`, `proof.py`, `generate_report_plots.py` |
| Seed cherry-picking | All 3 seeds trained; **Mean ± Std** aggregation written to `multi_seed_validation_report.txt`; canonical table = **median seed**, explicitly captioned; per-seed artifacts retained | `main.py`, `metrics.py` (`aggregate_seed_metrics`) |
| df=0.00 extraction bug | **Keyed-by-name** `res.params['nu']`/`['lambda']` in all GARCH fits and rolling filter | `build_data.py`, `proof.py`, `generate_report_plots.py` |
| `Log_Ret` target leak | Removed `Log_Ret` from `time_varying_unknown_reals`; AR info only via `Log_Ret_Lag1/Lag2` | `tft_model.py`, `explainability.py` |
| Dead/broken modules | **Deleted** `hpo.py` (ImportError) and `data_loader.py` (fabricated constant CPU) | deleted |
| Config drift | Single source of truth in `config.py`; all modules import from it; torch import guarded | `config.py` |

## 2.2 Validation performed
- All `.py` files parse cleanly (`ast.parse`).
- `config.py` imports standalone on this machine (torch guard verified: `config OK: 64 [42,123,777] ^INDIAVIX torch=None`).
- Full runtime deps (torch/pandas/pytorch-forecasting/arch) are Colab-only and not present locally, so end-to-end execution must be confirmed on Colab after `python build_data.py && python main.py && python generate_report_plots.py`.

## 2.3 Would the project now be publishable?

**Yes — conditionally.** The blockers are removed, but publication is contingent on what the *re-run* shows:

1. **Expected: DM insignificance remains.** The paper must be written as a **parity/interpretability** contribution ("TFT achieves statistically indistinguishable 99% VaR coverage vs a skew-t GJR-GARCH, while adding interpretability and relaxing parametric constraints"). This is a legitimate, publishable framing **only if** the ES backtest and pinball-loss comparisons are reported transparently, including where GARCH wins.
2. **The ES backtest must be run and reported.** The refactor *enables* it but does not *guarantee* a favorable outcome. If ES t-tests are also insignificant, the honest contribution is even narrower (interpretability + no parametric constraints).
3. **Seed aggregation must be shown**, not just median-seed. The `multi_seed_validation_report.txt` now enforces this.
4. **Granger results will likely weaken** (overlapping-window artifact removed) — this is correct and must be presented as such, not "fixed" to force significance.

**Recommended target venues if the re-run supports parity + a meaningful ES result:** *Journal of Financial Econometrics*, *Quantitative Finance*, *International Journal of Forecasting* (forecast-diagnostics strand), or a good **arXiv quantitative-finance working paper** first. A purely null/parity result with rigorous methodology is viable for a methods-focused journal or a working paper; it is **not** viable for an "empirical edge" venue.

**Remaining non-code items before submission:**
- Re-run the full pipeline on Colab and capture the new seed-aggregated report + ES results.
- Reproducibility: commit `master_df.csv`/prediction CSVs or a versioned data snapshot (currently gitignored).
- Add multi-level VaR backtests (95/97.5/99.5) and a PIT/density evaluation to fully substantiate the fat-tail claim.
- Write the paper honestly around the parity-and-interpretability thesis.

---

# PART 3 — COLOMBA RUN REVIEW & SUBSEQUENT FIXES (3 ROUNDS OF FINDINGS)

## 3.1 What the first Colab re-run confirmed
- **DM parity is now solid and honestly reported:** NIFTY50 DM p=0.564, BANKNIFTY p=0.778, NIFTYIT p=0.706 — all comfortably > 0.30 across the panel.
- **Green Zone across the full panel** (3/500, 3/500, 7/500; Basel limit 9) — real, defensible.
- **Seed aggregation machinery works** (Mean ± Std, median-seed promotion, per-seed pinball ranking).
- **Outcomes are seed-invariant** (std ≈ 0 on Kupiec/Christoffersen/breaches): a stability result that must be framed as such, not as seed variation.

## 3.2 Problems the first re-run exposed (and their fixes)

### 3.2.1 Degenerate Expected Shortfall test (FIXED in metrics.py)
- NIFTY50 produced `es_t_stat = −42.77` from only **3 exceedances** — a numerical artifact of near-zero sample variance, not meaningful evidence. The `es_p_value=0.0005` was misleading.
- The sign was also adverse: `es_mean_resid = −3.27` means the 3 breach days averaged 3.3 conditional σ below the mean → the model **understates tail severity when it fails** (opposite of "safe in unconstrained production").
- **Fix:** `mcnell_frey_es_test` now (a) always reports **descriptive** ES + mean standardized residual when ≥1 exceedance, (b) only computes the t-test when exceedances ≥ `ES_MIN_BREACHES_TESTABLE` (5), and (c) surfaces an `es_testable` flag. The audit table adds "ES n", "ES mean resid", "ES testable" columns with a methodology note.

### 3.2.2 `df(nu) = nan` — fat-tail claim still unverifiable (FIXED)
- `lambda` extracted but `nu` did not, even with `params.get('nu')`.
- **Fix:** added `extract_garch_dist_params(res)` in metrics.py — keyed lookup on the fitted parameter names with aliases (`nu/df/v/shape/tail`, `lambda/skew/gamma`) then a fallback to the distribution's declared parameter-name order. `proof.py` and `generate_report_plots.py` now use it; `proof.py` prints the raw fitted parameter names for transparency.

### 3.2.3 Granger p=0.0000 at ALL 9 cells — mechanically suspect (FIXED at the source)
- Root cause in `build_macro_features` (build_data.py): `*_Level` were ffill()-ed onto the ticker calendar **then log-differenced**, creating artificial zero returns across US/India market closures → deflated variance → inflated significance.
- **Fix:** log-differences are now computed on the **native VIX calendar** (only observed days), then the *changes* are reindexed (ffill of changes is safe — it carries forward the last realized change, never invents a zero). A shared `granger_series_from_panel()` helper in metrics.py uses the clean `*_Diff` columns; both proof.py and generate_report_plots.py use it. The domestic series source (real India VIX vs `Domestic_RV_Proxy`) is now disclosed in the report.

## 3.3 Review of the proposed "New Hypothesis/Result" narrative
The repositioned thesis is materially better, but two claims must be toned down:
1. **"Exact regulatory calibration"** is not supported — 3 breaches vs 5 expected is *under*-breaching (conservative), not "exact." Use "Green-Zone 99% coverage and statistical parity."
2. **"Safely operate in unconstrained production"** is contradicted by the ES direction (tail understatement on breach days). The defensible claim is: *"Green-Zone 99% coverage and statistical parity with skew-t GJR-GARCH, with transparent VSN attribution."*
3. The **68/5/7 VSN split** is median-seed/single-run data and sums to 80% — the remaining ~20% must be identified, and across-seed stability reported.

## 3.4 Post-fix re-run checklist
- Re-run `build_data.py` (regenerates clean native-calendar `*_Diff` columns) → `main.py` → `generate_report_plots.py`.
- Confirm: `nu` is now populated; Granger p-values are no longer 0.0000 at every cell; ES t-stat is suppressed for n<5; the domestic-series label appears in the report.
- For the paper, still needed: multi-level VaR backtests (95/97.5/99.5), a PIT/density evaluation, and reproducible committed data artifacts.

---

# PART 4 — TIMEZONE LOOKAHEAD, LAG DUPLICATION, AND GRANGER ROBUSTNESS (ROUND 4 REVIEW)

## 4.1 Verdict on the five proposed fixes

| # | Fix proposed | Verdict | Action taken |
|---|---|---|---|
| 1 | US_VIX_Diff timezone lookahead (shift by 2 or move to unknown) | **CORRECT — genuine leak.** Confirmed by tracing alignment. | Shift US features by 2 (keeps them known reals; avoids teacher-forcing train/inference mismatch from unknown-reals). |
| 2 | Remove Log_Ret_Lag1/Lag2 duplication | **CORRECT.** TFT encoder processes the target sequence natively; explicit lags are literal copies. | Removed from data builder, feature list, and explainability categories. |
| 3 | NIFTY IT "Green Zone falsehood" | **DISCREPANCY — must not be fabricated.** Cited numbers (12 breaches, Kupiec p=0.0077) contradict the Colab output actually produced (7 breaches, Kupiec p=0.3966). Per-asset honest reporting is already built in. | No narrative fabricated; per-asset table retained. Confirm the correct numbers before writing. |
| 4 | ES tail understatement ("not safe for production") | **Already fixed** in the honest-ES refactor (descriptive-only below n=5, `es_mean_resid` sign exposed). | Retract the "safe for unconstrained production" claim; EVT overlay is the honest recommendation. |
| 5 | Granger p=0.0000 suspicious (ADF + artifact check) | **VALID** — a reviewer red flag. | Added `granger_diagnostics()` (ADF + zero%/dup% checks) and report it. |

## 4.2 Fix 1 — timezone lookahead (implemented)
- `US_VIX_SHIFT = 2`, `INDIA_VIX_SHIFT = 1` added to config with full timezone reasoning.
- `build_macro_features` now lags US level + US log-diff by 2 rows after native-calendar differencing; India VIX (same zone) lagged by 1. This guarantees the value at forecast step t+1 reflects only information known at India's close on t.

## 4.3 Fix 2 — lag-column duplication (implemented)
- Removed `Log_Ret_Lag1`/`Log_Ret_Lag2` from the data builder, from `time_varying_unknown_reals`, and from the VSN category map. The encoder now relies solely on its native target-sequence processing.

## 4.4 Fix 5 — Granger robustness (implemented)
- Added `granger_diagnostics()` in metrics.py: per-series **ADF stationarity p**, **zero-fraction**, and **duplicate-fraction**. Wired into `proof.py` (console report) and `generate_report_plots.py` (embedded in the master report with a "how to read" note). Near-zero p-values are now only interpretable if the series are stationary and artifact-free.

## 4.5 Required re-run
1. `python build_data.py` (regenerates timezone-corrected US features; drops lag columns)
2. `python main.py` (re-trains; expect VSN weights to de-fragment once lags are removed)
3. `python generate_report_plots.py`
4. Verify: Granger diagnostics printed; confirm whether NIFTY IT is truly at 7 or 12 breaches and use THAT number in the narrative.

---

# PART 5 — RESTORING AUTOREGRESSIVE CAPACITY & MULTI-SEED EXPLAINABILITY (ROUND 5)

## 5.1 Verdict on the review comments

| # | Review comment | Verdict | Action |
|---|---|---|---|
| 1 | "Removing Log_Ret from unknown reals made the TFT blind to price history" | **CORRECT — the earlier removal was a mistake.** PyTorch Forecasting's docs state the target *"should be included"* in `time_varying_unknown_reals` when real-valued; the encoder target is carried separately and is not re-injected as an encoder feature unless listed. | Re-added `Log_Ret` to `time_varying_unknown_reals`. |
| 2 | Explainability must aggregate across seeds | **VALID.** VaR metrics are mean±std across seeds but the explainability script used only `checkpoints[0]`. | Rewrote `explainability.py` to loop all seeds, reporting VSN + temporal attention Mean±Std. |
| 3 | US_VIX_SHIFT=2 timezone logic | **VINDICATED.** shift(1) at decoder row t+1 would expose Tuesday's US close which is unknown at Tuesday 15:30 IST; shift(2) exposes Monday's close, which IS known. Leave as-is. | No change. |

## 5.2 Fix 1 — Log_Ret restored as an unknown real (implemented)
- [`tft_model.py`](tft_model.py) `candidate_unknown` now includes `"Log_Ret"` first, with a comment explaining PyTorch Forecasting semantics: unknown reals feed the encoder (observed ≤ t) and are hidden from the decoder at t+1, restoring autoregressive momentum with no look-ahead. Explicit lag columns remain removed (encoder sees the full window through this mechanism).

## 5.3 Fix 2 — Multi-seed explainability (implemented)
- [`explainability.py`](explainability.py) rewritten:
  - Loops over every seed in `config.VALIDATION_SEEDS`, locating each `*seed<num>*.ckpt`.
  - Extracts per-seed VSN percentages and temporal attention weights.
  - Aggregates Mean ± Std across seeds; writes `vsn_feature_importance_seed_aggregated.csv`, `temporal_attention_distribution_seed_aggregated.csv`, and `tft_explainability_report.txt`.
  - NaN-safe std; per-seed detail retained in long-form records.
- [`plot_master_dashboard.py`](plot_master_dashboard.py) updated to read the aggregated CSVs (with legacy fallback) and render mean-across-seeds titles.

## 5.4 Required re-run
1. `python build_data.py`
2. `python main.py` (now the TFT sees its own price history — expect VSN weights to change materially, with Log_Ret regaining a dominant share)
3. `python explainability.py` (multi-seed VSN/attention aggregation)
4. `python generate_report_plots.py`
5. Re-check: VSN now shows Log_Ret as a top feature (not the 56.9% US_VIX artifact); explainability report shows mean±std across the 3 seeds.

---

# PART 6 — Log_Ret_Feature THROUGH THE VSN (ROUND 6)

## 6.1 Verdict
The proposal is **sound and implemented**. Rationale: in PyTorch Forecasting, listing the return history as a real-valued *unknown* feature (`Log_Ret_Feature`, a copy of `Log_Ret`) feeds it through the **encoder + VSN** (observed up to t) and hides it from the **decoder** at t+1 — leakage-safe. This forces the autoregressive sequence's VSN importance to be measured alongside the macro and econometric priors. `Log_Ret` itself stays reserved as the `target` (no duplicate column; no information loss — `Log_Ret_Feature` carries the identical series through the VSN).

## 6.2 Changes made
- [`build_data.py`](build_data.py): adds `df["Log_Ret_Feature"] = df["Log_Ret"]` immediately after the GARCH block (pre-dropna, so alignment is preserved), with a doc-comment explaining the VSN-attribution purpose and the leakage-safe unknown-real semantics.
- [`tft_model.py`](tft_model.py): `candidate_unknown = ["Log_Ret_Feature", "GK_Vol", "GARCH_resid"]` — replaces the raw `Log_Ret` entry so the encoder-only return history is scored by the VSN; header comment updated accordingly.
- [`explainability.py`](explainability.py): `CATEGORY_MAP` entry `"Log_Ret_Feature": "Autoregressive Target History"`.

## 6.3 Required re-run
1. `python build_data.py`
2. `python main.py`
3. `python explainability.py` (expect `Log_Ret_Feature` to appear in the VSN table with a meaningful, dominant share)
4. `python generate_report_plots.py`

---

# PART 7 — eta ALIAS + GRANGER UN-SHIFTING (ROUND 7)

## 7.1 Verdict
Both fixes are sound and implemented in [`metrics.py`](metrics.py):

1. **`eta` alias**: Some arch versions/configurations name the skew-t degrees-of-freedom parameter `"eta"` rather than `"nu"` — which explains the `N/A` on Colab. Added `"eta"` to `_DF_ALIASES`.
2. **Granger un-shifting**: Granger causality is a DESCRIPTIVE in-sample lead-lag test (regress Y_t on lags of X), not a forecast evaluation. The ML pipeline's timezone shifts (US_VIX_SHIFT=2, INDIA_VIX_SHIFT=1) are leakage-safe for forecasting but warp the *relative* chronology for the econometric test. Un-shifting each series by its own lag (`shift(-US_VIX_SHIFT)`, `shift(-INDIA_VIX_SHIFT)`, `shift(-1)` for the RV proxy) restores true calendar alignment so the standard Granger mapping (X_{t-k} -> Y_t) reflects genuine market chronology. The ML feature columns are unchanged; only the econometric-diagnostic read-out is un-shifted.

## 7.2 Changes
- `_DF_ALIASES = ("nu", "df", "v", "shape", "tail", "eta")`
- `granger_series_from_panel`: US and (real) India VIX diffs un-shifted by their config lags; `Domestic_RV_Proxy` un-shifted by −1 (it was built as `rv.diff().shift(1)`). Docstring explains the descriptive-vs-forecasting distinction.

## 7.3 Required verification
1. `python proof.py` — `Nu (Tail df)` should print a numeric value (not N/A).
2. `python generate_report_plots.py` — the Granger plot/report should reflect true chronological lead-lag; the ADF + zero/dup diagnostics remain printed for reviewer confidence.

---

# PART 8 — FULL-CODEBASE REVIEW (ROUND 8): SEVERITY-RANKED FINDINGS

## 8.1 FATAL: Granger test contamination — un-shifting samples the FUTURE (and shifts the y-axis too)

In Round 7 the `granger_series_from_panel` helper was changed to un-shift the stored `*_Diff` columns:
```python
us  = sub["US_VIX_Diff"].astype(float).shift(-config.US_VIX_SHIFT)      # -2
dom = sub["India_VIX_Diff"].astype(float).shift(-config.INDIA_VIX_SHIFT)  # -1
```
Two independent defects follow:

1. **Negative `shift()` is a LOOK-AHEAD operation.** `US_VIX_Diff` is stored in `master_df.csv` with its ML-timezone lag applied (row D holds `log(US[D-1]/US[D-2])`). `shift(-2)` moves row D+2's value into row D — i.e., the series now contains **future** information relative to the stored calendar. Granger is a *causal* lag test: regressing `Y_t` on `X_{t-k}` with `X` containing future-dated values silently destroys the very causal interpretation the test is meant to establish. The "descriptive in-sample" rationale does not rescue this: even descriptive lead-lag inference requires `X` values to be dated **before** `Y` in the same row. This is a textbook look-ahead, and it directly re-introduces the exact bias the earlier timezone fix was designed to remove.

2. **The shift direction is applied to the WRONG series.** To align an ML-lagged feature back to its true calendar date, the shift must be applied to the feature that was *shifted forward*, not the one being *tested*. `US_VIX_Diff` was shifted by `+US_VIX_SHIFT` at build time; reversing it with `shift(-US_VIX_SHIFT)` restores the US series — but `India_VIX_Diff` (shifted only by `+1`) and the `Domestic_RV_Proxy` (shifted by `+1`) are **also** un-shifted, and the *y-axis* series in the test is `dom` (the domestic one). If the intent was to restore only the US lead so that `US_{t-k}` correctly precedes `dom_t`, then un-shifting the domestic series too is wrong: it double-shifts the lead-lag relationship the other way. The result is a test whose relative timing is *still* misaligned, now in the opposite direction.

**Why this is the single most important finding:** it invalidates every Granger p-value produced after Round 7, and it was introduced to "fix" a Granger artifact that the ADF/zero/dup diagnostics (Round 4) already guarded against. The correct econometric design for a *cross-border lead-lag* test is either (a) test on the true calendar series WITHOUT any ML shift applied (i.e., build separate, unshifted `US_VIX_NativeDiff` columns used only for the econometric test), or (b) drop the un-shift entirely and interpret the shifted lags as the ML horizon (documenting that lag-k in the ML feature = lag-k+2 in true time). Option (a) is the clean fix.

## 8.2 HIGH: Granger diagnostics run on the un-shifted (future-contaminated) series

`granger_diagnostics` is called with `clean_df["us"]` / `clean_df["dom"]` *after* the un-shift in both `proof.py` and `generate_report_plots.py`. So the ADF p-value, zero%, dup% — the very checks meant to certify the series as artifact-free — are computed on the future-contaminated version. The diagnostic therefore certifies the wrong object. Fix (a) above also repairs this.

## 8.3 HIGH: `Log_Ret_Feature` placement in build_data is AFTER the return-dropna but the feature copy is aligned to the pre-dropna frame — verify alignment

In `build_data.py` the copy `df["Log_Ret_Feature"] = df["Log_Ret"]` is created *after* the initial `df = df.dropna()` (line 263) but *before* the final `df = df.dropna()` (line 303). Since `Log_Ret_Feature` is a pure copy of `Log_Ret` and both are dropped on the same rows, alignment is preserved. This is NOT a bug — but it is fragile: any future edit that introduces a NaN into `Log_Ret` after the copy (e.g., a forward-fill or a resample between the two dropna calls) would silently desynchronize the feature from the target. Recommend moving the copy to immediately after `df["Log_Ret"]` is created and before the first dropna, so the two series are definitionally identical for the whole frame.

## 8.4 MEDIUM: `extract_garch_dist_params` fallback ordering can return the WRONG parameter as `nu`

The keyed lookup (Strategy 1) iterates `_DF_ALIASES + _SKEW_ALIASES` in a fixed order and assigns to `out["nu"]` only while `np.isnan`. If the fitted parameter set contains BOTH `nu` (df) and a skew param literally named `lambda`, this is fine. But if arch names the df `eta` and the skew `lambda`, the alias loop will hit `eta` first (good) — however Strategy 2 falls back to positional mapping of `dist.name` order, which is only correct if the distribution's `name` list is ordered [df, skew]. If the order is reversed, `nu` and `lambda` get swapped silently. Since the earlier Colab run produced `nu = nan` while `lambda` extracted fine, there is direct evidence the fallback can misbehave. Recommend printing `res.params.index` (proof.py already does) and validating the mapping once per arch version before trusting the numbers.

## 8.5 MEDIUM: `proof.py` / `generate_report_plots.py` still label the test "US VIX -> India Volatility" even when the domestic side is the RV proxy

Both call sites pass `domestic_label` through, but the on-screen copy in `proof.py` (lines 80, 83) prints `"US VIX Granger-causes {domestic_label}"` — good — while `generate_report_plots.py`'s report section header still says `"CROSS-BORDER CAUSALITY (DAILY LOG-DIFFS on native VIX calendar)"` and the per-ticker row prints the p-values without consistently carrying the `domestic_label`. If the real India VIX was unavailable, the report can still read as "US VIX drives India VIX" when it actually drives a realized-vol proxy. Must be labelled everywhere.

## 8.6 MEDIUM: `production_engine.py` still uses positional `res.params[-2:]` for the quantile multiplier

`q01_multiplier = model.distribution.ppf(0.01, res.params[-2:])` (line 62) — the exact fragile positional indexing that was removed elsewhere. If the arch parameter order changes (as the eta/nu saga proved it can), the deployed VaR floor (even if not applied as an override, it is still *reported* as the GARCH reference) silently corrupts. Should call `extract_garch_dist_params` and pass `(nu, lambda)` explicitly.

## 8.7 LOW: schema drift risk in `generate_and_save_predictions` and `production_engine`

`tft_model.generate_and_save_predictions` merges `df[["time_idx","ticker","Date","Log_Ret","GARCH_VaR_99","GARCH_sigma"]]` into the prediction frame. After the Round 6 feature renames, the frame no longer carries `Log_Ret_Feature` (it is not needed downstream) — correct — but any script that assumed the old `Log_Ret_Lag1/2` columns (e.g., older `plot_*` or report code) would now KeyError. A quick grep confirmed no live reference remains, but the `production_engine` live path reconstructs `encoder_df` from `master_df` and relies on `tft.dataset_parameters` to know which columns to feed; if `master_df` lacks a column the checkpoint expects (e.g. `Log_Ret_Feature` was added to the training frame but the *live* `master_df.csv` used at inference time was regenerated without it), `from_parameters` will fail. The live buffer must be rebuilt from the SAME `build_data.py` that produced the training frame — worth an explicit guard.

## 8.8 Verdict on current state
- **Publishability gate: not yet clean.** The Granger un-shift (8.1) is a regression that must be reverted/fixed before any causal claim is made. Everything else in the ML/VaR path is structurally sound.
- The correct fix for 8.1/8.2 is to build **separate, unshifted native-calendar log-diff columns** (`US_VIX_NativeDiff`, `India_VIX_NativeDiff`) used *only* by the econometric Granger path, and leave the ML features timezone-shifted. This keeps the forecasting pipeline leakage-free while giving the econometric test a true chronological series. Recommend implementing that fix.

---

# PART 9 — OPTION-A FIREWALL IMPLEMENTATION (ROUND 9): FIX LOG

All findings 8.1-8.7 have been fixed by separating native (chronological) columns used by the econometric Granger path from the shifted ML columns. This is the "firewall" design.

## 9.1 build_data.py (8.1, 8.3)
- `Log_Ret_Feature` copy moved to **immediately after** `df["Log_Ret"]` creation and *before* the first `dropna()` — the two series can never desynchronize (8.3).
- `build_macro_features` now emits BOTH:
  - **ML (shifted):** `US_VIX_Diff`, `US_VIX_Level`, `India_VIX_Diff` (timezone-lagged, leakage-safe for the TFT).
  - **Native (unshifted):** `US_VIX_NativeDiff`, `India_VIX_NativeDiff` — chronological daily log-diffs used ONLY by the Granger test.
- The master frame additionally carries `Domestic_RV_NativeProxy` (unshifted rolling-RV diff, for Granger when no real India VIX) and `Domestic_RV_Proxy` (its `shift(1)` ML copy).

## 9.2 metrics.py (8.1, 8.2, 8.4)
- `granger_series_from_panel` now reads `US_VIX_NativeDiff` / `India_VIX_NativeDiff` / `Domestic_RV_NativeProxy` with **ZERO shifting** — the future-leak introduced by Round 7's `shift(-lag)` is removed. Docstring documents the firewall.
- `extract_garch_dist_params` Strategy 2 replaced with a **SkewStudent-specific positional fallback**: guarded on the distribution name, maps trailing `[..., nu, lambda]`. Keyed Strategy 1 retained.

## 9.3 generate_report_plots.py + proof.py (8.5)
- Report header now reads "NATIVE CALENDAR -- no ML timezone shift"; each asset line prints `US VIX -> {domestic_label}` so the RV-proxy vs real-India-VIX distinction is explicit. Plot titles carry the domestic source too.
- Diagnostic sub-labels renamed to `US VIX NativeDiff` / `Domestic Native`.

## 9.4 production_engine.py (8.6, 8.7)
- Positional `res.params[-2:]` replaced with `extract_garch_dist_params` + explicit `[nu, lambda]` passed to `distribution.ppf` (with safe fallbacks 5.0 / 0.0).
- Added a **schema-drift guard**: the live buffer is checked for every column the checkpoint expects (from `tft.dataset_parameters` known/unknown reals, e.g. `Log_Ret_Feature`), raising a clear error instructing to re-run `build_data.py`.

## 9.5 Re-run required
1. `python build_data.py` (emits native columns; `Log_Ret_Feature` pre-dropna)
2. `python main.py` (retrain with unchanged ML features)
3. `python proof.py` / `python generate_report_plots.py` — Granger now on true chronology; confirm the ADF/zero/dup diagnostics on the NATIVE columns.
4. `python production_engine.py` — live path guarded.
