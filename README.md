# Econometrically-Conditioned TFT for NIFTY 1% VaR Forecasting

## Research objective

This repository studies whether a Temporal Fusion Transformer (TFT) adds incremental value to one-day-ahead 1% Value at Risk (VaR) forecasting when the benchmark is a carefully specified point-in-time Skew-t GJR-GARCH(1,1) model.

The empirical assets are the NIFTY 50, NIFTY Bank, and NIFTY IT indices. The project deliberately treats the classical benchmark as a strong comparator rather than assuming that a higher-capacity neural model should win.

## Main result

The frozen experiment does **not** show a statistically significant 1% VaR forecasting improvement from any tested TFT specification over the Skew-t GJR-GARCH benchmark.

The final study contains:

- Raw TFT
- GARCH-conditioned TFT
- GARCH + US-VIX TFT
- GARCH + domestic-volatility TFT
- Full ECTFT using both volatility inputs
- A dedicated q=0.01 specialist TFT
- Additional Historical Simulation and asymmetric CAViaR benchmarks

The strongest defensible conclusion is therefore:

> In this daily-frequency, one-day-ahead extreme-quantile setting, the tested neural architectures do not demonstrate sufficient incremental predictive value to displace the strong Skew-t GJR-GARCH benchmark.

This is an empirical result for the tested sample and information set. It is **not** a claim that deep learning is universally ineffective for financial risk forecasting.

## Data and information set

- Raw research window: 2015-01-01 onward
- Frozen research cutoff: 2026-09-07
- Primary OOS horizon: 500 trading observations
- Validation window: 250 observations
- GARCH estimation window: 1,000 observations
- GARCH parameter refit frequency: 21 trading days
- Target: next-day 1% return quantile

### Timing discipline

The TFT decoder receives known covariates from the dataframe row corresponding to the prediction date. Therefore the dataframe must contain only information available at the forecast origin.

- `India_VIX_Diff`: shifted by 1 row because the India VIX close is available at the Indian market close on day t and is used for the forecast of day t+1.
- `US_VIX_Diff`: shifted by 2 Indian trading-day rows because the US market close corresponding to day t occurs after the Indian market close on day t.

These shifts are intentional anti-look-ahead controls. Do not replace them with mechanically smaller integer shifts without re-deriving the information set.

## Final model configuration

The final TFT specification was selected using a 20-trial Optuna tournament on the validation sample.

- Hidden size: 64
- Attention heads: 4
- Encoder length: 10 trading days
- Dropout: 0.3354115
- Learning rate: 0.00158829
- Batch size: 64
- Gradient clipping: 0.1
- Maximum epochs: 80
- Quantiles: 0.01, 0.50, 0.99
- Final seeds: 42, 123, 777, 2027, 31415
- Canonical forecast: arithmetic mean of the five seed-specific q=0.01 forecasts

The Optuna winner had validation loss 0.2870414555 under the tournament configuration (40-epoch development budget). The selected configuration was subsequently rerun across five seeds for the final study.

## Primary evaluation

The primary predictive criterion is 1% pinball loss. Competing forecasts are compared using paired Diebold–Mariano tests. Holm adjustment is applied within the pre-specified three-asset comparison family.

VaR diagnostics include:

- Kupiec unconditional coverage
- Christoffersen independence, where estimable
- Engle–Manganelli dynamic quantile (DQ) diagnostics
- descriptive tail-exceedance depth

With 500 OOS observations at alpha=0.01, only about five exceptions are expected. Coverage tests therefore have limited power and are interpreted as diagnostics, not proof of perfect calibration.

## Final six-arm result summary

### Raw TFT vs GJR-GARCH

No statistically significant improvement on any asset:

- NIFTY50: DM p = 0.0669
- BANKNIFTY: DM p = 0.8463
- NIFTYIT: DM p = 0.6671

### GARCH-TFT vs GJR-GARCH

No statistically significant improvement:

- NIFTY50: DM p = 0.2046
- BANKNIFTY: DM p = 0.1941
- NIFTYIT: DM p = 0.3168

### Full ECTFT vs GARCH-TFT

No statistically significant difference:

- NIFTY50: DM p = 0.7782
- BANKNIFTY: DM p = 0.6942
- NIFTYIT: DM p = 0.5003

The final results therefore do **not** support the stronger earlier claim that macro variables significantly damage the Full ECTFT. The correct conclusion is that they do not provide robust incremental 1% VaR forecasting value in the final experiment.

## Final Full ECTFT VaR diagnostics

| Asset | Breaches | Expected | Rate | Kupiec p | Christoffersen p | DQ p |
|---|---:|---:|---:|---:|---:|---:|
| NIFTY50 | 1 | 5 | 0.20% | 0.0282 | N/A | 0.7187 |
| BANKNIFTY | 2 | 5 | 0.40% | 0.1250 | 0.8990 | 0.9371 |
| NIFTYIT | 11 | 5 | 2.20% | 0.0199 | 0.4813 | 0.1283 |

The custom binomial zone in the code is **regulatory-inspired only**. It is not a Basel traffic-light certification or Basel III/FRTB compliance assessment.

## Cross-market result

US VIX innovations show strong predictive precedence for domestic India VIX innovations on shared trading dates at 1-, 2-, and 5-day lag specifications for all three indices.

This is interpreted as predictive precedence, not structural causality. The result is useful because it demonstrates that strong predictability of an intermediate volatility variable does not automatically imply incremental improvement in the downstream 1% VaR target.

## Interpretability results

Five-seed VSN allocation in the Full ECTFT:

| Feature | Mean weight |
|---|---:|
| GK_Vol | 35.62% |
| Log_Ret_Feature | 16.38% |
| relative_time_idx | 14.86% |
| US_VIX_Diff | 13.26% |
| India_VIX_Diff | 11.26% |
| GARCH_sigma | 8.61% |

These values are descriptive model-internal allocation weights. They are not causal feature importance measures.

## Repository structure

```text
nifty50_project/
├── README.md
├── config.py                 # single research configuration
├── build_data.py             # data retrieval + PIT GARCH feature construction
├── tft_model.py              # TFT datasets/training/checkpoint logic
├── main.py                   # final multi-seed TFT experiment
├── ablation_runner.py        # six-arm ablation suite
├── specialist_runner.py      # q=0.01 specialist robustness check
├── hpo_optuna.py             # validation-only Optuna tournament
├── classical_benchmarks.py   # HS, CAViaR, MCS, regime analysis
├── explainability.py         # VSN + temporal attention
├── learning_curves.py        # training diagnostics
├── metrics.py                # VaR + forecast comparison metrics
├── generate_report_plots.py  # publication figures
├── render_tables.py          # report tables
├── environment_check.py     # environment verification
├── results/                  # final result artifacts and figures
├── paper draft/              # manuscript development files
├── archive/                  # historical development material
└── ignore__other_results/   # nonessential raw/intermediate outputs
```

Legacy deployment files may be removed or archived in the publication repository because deployment is outside the final scientific scope.

## Reproduction order

1. Run `environment_check.py` and verify the pinned environment.
2. Run `build_data.py` using the frozen research cutoff.
3. Run `main.py` for the five-seed canonical TFT ensemble.
4. Run `ablation_runner.py` for the six-arm comparison.
5. Run `specialist_runner.py` for the q=0.01 specialist experiment.
6. Run `classical_benchmarks.py` for Historical Simulation, CAViaR, model-confidence-set, and regime diagnostics.
7. Run `explainability.py` for five-seed VSN and attention diagnostics.
8. Run `generate_report_plots.py` and `render_tables.py` for the publication figures/tables.

Do not use test-set performance to select a seed, hyperparameter, or model variant.

## Statistical interpretation rules

The paper uses the following interpretation discipline:

- Failure to reject a forecast-comparison null is not evidence of model equivalence.
- A significant Kupiec rejection can indicate either too many or too few exceptions; direction matters.
- A non-significant Christoffersen or DQ result means no evidence of the tested failure mode, not proof that the model is perfectly specified.
- Granger-style tests indicate predictive precedence, not structural causality.
- VSN and attention outputs are descriptive model diagnostics.
- The 500-day / 1% design has low expected exception counts, limiting the power of tail-coverage tests.

## Scope boundary

This repository is a research study, not a production risk engine. Deployment/MLOps code developed during earlier project stages is not part of the final empirical claim and can be frozen or removed.

Future extensions could examine higher-frequency data, option-implied distributions, richer cross-sectional information, multi-asset dependence, or longer multi-regime evaluation windows. Those are separate research questions and are deliberately outside the scope of the frozen study.

## Core references

- Bollerslev (1986), Generalized autoregressive conditional heteroskedasticity.
- Glosten, Jagannathan & Runkle (1993), asymmetric GARCH.
- Kupiec (1995), VaR backtesting.
- Diebold & Mariano (1995), forecast comparison.
- Christoffersen (1998), interval/conditional coverage.
- Diebold, Gunther & Tay (1998), density forecast evaluation.
- Engle & Manganelli (2004), CAViaR.
- Giacomini & Komunjer (2005), conditional quantile forecast evaluation.
- Gneiting & Raftery (2007), proper scoring rules.
- Hansen, Lunde & Nason (2011), Model Confidence Set.
- Lim et al. (2021), Temporal Fusion Transformer.
- Huang & Wang (2012), alternative GARCH distributions.
- Gurrola-Perez & Murphy (2015), filtered historical simulation VaR.
- Petrosino et al. (2025), GARCH-TFT volatility forecasting.
- Dai et al. (2025), Set-GARCH.

## License / data note

Add the project licence, data-provider terms, and exact software-version statement before public release. Market data availability and redistribution terms should be checked against the provider used for the final replication package.
