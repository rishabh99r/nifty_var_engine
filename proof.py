# proof.py
# =============================================================================
# Baseline econometric proofs & Granger PREDICTIVE-PRECEDENCE diagnostics.
#
# Fixes applied:
#   - Skew-T GJR-GARCH parameters extracted BY NAME (nu, lambda), never
#     positionally, so the reported tail-df is meaningful.
#   - Granger tests use the ACTUAL US VIX and (real) India VIX daily
#     log-differences INNER-JOINED on genuinely shared trading dates (S4),
#     avoiding forward-fill / overlapping-window artifacts.
#   - Terminology (S7): "predictive precedence", not "causation".
# =============================================================================
import os
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests

import config
from metrics import extract_garch_dist_params, granger_series_from_panel, granger_diagnostics, format_granger_diagnostics

warnings.filterwarnings("ignore")


def run_empirical_proofs(df_path="master_df.csv", max_lag=5):
    print("===== RUNNING BASELINE ECONOMETRIC & CAUSALITY PROOFS =====")
    if not os.path.exists(df_path):
        drive_path = os.path.join(config.OUTPUT_DIR, df_path)
        if os.path.exists(drive_path):
            df_path = drive_path
        else:
            raise FileNotFoundError(f"[ERROR] {df_path} missing. Run build_data.py first.")

    df = pd.read_csv(df_path)
    print(f"[LOAD] Loaded multi-series panel ({len(df)} rows across {df['ticker'].nunique()} tickers).")

    tickers = df["ticker"].unique()

    # 1. INDEPENDENT GJR-GARCH ESTIMATION (keyed-by-name parameter extraction)
    # NOTE (Phase-3 / Reviewer): these are FULL-SAMPLE descriptive estimates --
    # they characterize the data and are NOT the point-in-time forecasts used in
    # the backtest (those come from build_data.rolling_gjr_garch_pit).
    print("\n[STEP 1] Fitting Empirical Skew-T GJR-GARCH(1,1) Across Panel Series...")
    print("         (FULL-SAMPLE descriptive estimates, distinct from the PIT backtest fits)")
    for sym in tickers:
        sub = df[df["ticker"] == sym].sort_values(by="time_idx").dropna(subset=["Log_Ret"])
        returns = sub["Log_Ret"].values
        am = arch_model(returns, vol="Garch", p=1, o=1, q=1, dist="skewt")
        try:
            res = am.fit(disp="off", show_warning=False)
        except Exception as e:
            print(f"  [WARN] {sym}: full-sample GARCH fit raised: {e}")
            continue
        conv_flag = int(getattr(res, "convergence_flag", -1))

        shape = extract_garch_dist_params(res)
        nu = shape["nu"]
        lam = shape["lambda"]

        conv_tag = "CONVERGED" if conv_flag == 0 else f"NOT CONVERGED (flag={conv_flag})"
        print(f"\n--- Estimated Parameters: {sym}  [{conv_tag}] ---")
        print(f"  Omega (Baseline Variance):   {res.params['omega']:.6f}")
        print(f"  Alpha (Symmetric Shock):      {res.params['alpha[1]']:.6f}")
        print(f"  Gamma (Asymmetric Leverage):  {res.params['gamma[1]']:.6f}")
        print(f"  Beta (GARCH Persistence):      {res.params['beta[1]']:.6f}")
        print(f"  Nu (Tail Degrees of Freedom): {nu:.4f}" if not np.isnan(nu) else "  Nu (Tail df): N/A")
        print(f"  Lambda (Skew Parameter):      {lam:.4f}" if not np.isnan(lam) else "  Lambda (Skew): N/A")
        if conv_flag != 0:
            print("  [WARNING] Optimizer did not report convergence; treat these "
                  "descriptive parameters with caution.")
        # Report the raw fitted parameter names so any naming surprise is visible
        print(f"  [DEBUG] Fitted parameter names: {list(res.params.index)}")

    # 2. CROSS-BORDER GRANGER PREDICTIVE PRECEDENCE on native (unshifted) VIX
    #    log-differences, inner-joined on shared trading dates (S4/S7).
    print("\n[STEP 2] Granger Predictive-Precedence (US VIX <-> India Volatility) "
          "on SHARED-DATE native-calendar log-diffs...")
    nifty = df[df["ticker"] == "NIFTY50"].sort_values(by="time_idx").copy()

    # Build chronologically-true series from the *_NativeDiff columns built in
    # build_data.py (no ML timezone shift -- the econometric test must not use
    # the shifted *_Diff columns, which would warp the causal lag structure).
    us_logdiff, dom_logdiff, domestic_label = granger_series_from_panel(nifty)

    clean_df = pd.DataFrame({"us": us_logdiff, "dom": dom_logdiff}).dropna()
    lags = [1, 2, 3, 5]

    # Robustness diagnostics before trusting any near-zero p-value
    diag = granger_diagnostics({"US VIX NativeDiff": clean_df["us"], "Domestic Native": clean_df["dom"]})
    print("\n--- Granger Input Diagnostics (ADF stationarity + artifact check) ---")
    print("  " + format_granger_diagnostics(diag))
    print("  [NOTE] Granger p-values are only credible if ADF p<0.05 (stationary)")
    print("  and zero%/dup% are small (no calendar-misalignment artifact).\n")

    # NOTE (10.4): statsmodels' grangercausalitytests tests whether Col 1
    # has predictive precedence for Col 0. So [["dom","us"]] = "US -> dom"
    # (forward) and [["us","dom"]] = "dom -> US" (reverse). "Granger" here means
    # predictive precedence, NOT structural causation (S7).
    print(f"  -> Forward: US VIX predictive precedence for {domestic_label}")
    res_forward = grangercausalitytests(clean_df[["dom", "us"]], maxlag=max_lag, verbose=False)

    print(f"  -> Reverse: {domestic_label} predictive precedence for US VIX")
    res_reverse = grangercausalitytests(clean_df[["us", "dom"]], maxlag=max_lag, verbose=False)

    print("\n--- Granger Predictive-Precedence Matrix (p-values) ---")
    print("    [S9 caveat: unadjusted p-values, exploratory; family not Holm-corrected]")
    for l in lags:
        p_fwd = res_forward[l][0]["ssr_chi2test"][1]
        p_rev = res_reverse[l][0]["ssr_chi2test"][1]
        fwd_star = "***" if p_fwd < 0.01 else ("**" if p_fwd < 0.05 else "")
        rev_star = "***" if p_rev < 0.01 else ("**" if p_rev < 0.05 else "")
        print(f"  Lag {l} Day(s):")
        print(f"    - US VIX predictive precedence for India Volatility: p = {p_fwd:.5f} {fwd_star}")
        print(f"    - India Volatility predictive precedence for US VIX: p = {p_rev:.5f} {rev_star}")

    print("\n===== BASELINE DIAGNOSTICS COMPLETE =====")


if __name__ == "__main__":
    run_empirical_proofs()
