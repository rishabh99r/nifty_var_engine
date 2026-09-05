# proof.py
# =============================================================================
# Baseline econometric & causality proofs.
#
# Fixes applied:
#   - Skew-T GJR-GARCH parameters extracted BY NAME (nu, lambda), never
#     positionally, so the reported tail-df is meaningful.
#   - Granger causality uses the ACTUAL US VIX and (real) India VIX daily
#     log-differences, avoiding the serial-correlation artifacts of
#     overlapping rolling windows.
# =============================================================================
import os
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests

import config
from metrics import extract_garch_dist_params, granger_series_from_panel

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
    print("\n[STEP 1] Fitting Empirical Skew-T GJR-GARCH(1,1) Across Panel Series...")
    for sym in tickers:
        sub = df[df["ticker"] == sym].sort_values(by="time_idx").dropna(subset=["Log_Ret"])
        returns = sub["Log_Ret"].values
        am = arch_model(returns, vol="Garch", p=1, o=1, q=1, dist="skewt")
        res = am.fit(disp="off")

        shape = extract_garch_dist_params(res)
        nu = shape["nu"]
        lam = shape["lambda"]

        print(f"\n--- Estimated Parameters: {sym} ---")
        print(f"  Omega (Baseline Variance):   {res.params['omega']:.6f}")
        print(f"  Alpha (Symmetric Shock):      {res.params['alpha[1]']:.6f}")
        print(f"  Gamma (Asymmetric Leverage):  {res.params['gamma[1]']:.6f}")
        print(f"  Beta (GARCH Persistence):      {res.params['beta[1]']:.6f}")
        print(f"  Nu (Tail Degrees of Freedom): {nu:.4f}" if not np.isnan(nu) else "  Nu (Tail df): N/A")
        print(f"  Lambda (Skew Parameter):      {lam:.4f}" if not np.isnan(lam) else "  Lambda (Skew): N/A")
        # Report the raw fitted parameter names so any naming surprise is visible
        print(f"  [DEBUG] Fitted parameter names: {list(res.params.index)}")

    # 2. CROSS-BORDER GRANGER CAUSALITY on ACTUAL VIX log-differences
    print("\n[STEP 2] Executing Granger Causality (US VIX <-> India Volatility) on ACTUAL VIX log-diffs...")
    nifty = df[df["ticker"] == "NIFTY50"].sort_values(by="time_idx").copy()

    # Build clean series from the *_Diff columns (computed on native VIX
    # calendar in build_data.py -- never by differencing an ffill()-ed level).
    us_logdiff, dom_logdiff, domestic_label = granger_series_from_panel(nifty)

    clean_df = pd.DataFrame({"us": us_logdiff, "dom": dom_logdiff}).dropna()
    lags = [1, 2, 3, 5]

    print(f"  -> Forward: US VIX Granger-causes {domestic_label}")
    res_forward = grangercausalitytests(clean_df[["dom", "us"]], maxlag=max_lag, verbose=False)

    print(f"  -> Reverse: {domestic_label} Granger-causes US VIX")
    res_reverse = grangercausalitytests(clean_df[["us", "dom"]], maxlag=max_lag, verbose=False)

    print("\n--- Granger Causality Significance Matrix (p-values) ---")
    for l in lags:
        p_fwd = res_forward[l][0]["ssr_chi2test"][1]
        p_rev = res_reverse[l][0]["ssr_chi2test"][1]
        fwd_star = "***" if p_fwd < 0.01 else ("**" if p_fwd < 0.05 else "")
        rev_star = "***" if p_rev < 0.01 else ("**" if p_rev < 0.05 else "")
        print(f"  Lag {l} Day(s):")
        print(f"    - US VIX Granger-causes India Volatility: p = {p_fwd:.5f} {fwd_star}")
        print(f"    - India Volatility Granger-causes US VIX: p = {p_rev:.5f} {rev_star}")

    print("\n===== BASELINE DIAGNOSTICS COMPLETE =====")


if __name__ == "__main__":
    run_empirical_proofs()
