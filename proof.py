# proof.py
import os
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

DRIVE_DIR = "/content/drive/MyDrive/GARCH_TFT_Results"


def run_empirical_proofs(df_path="master_df.csv", max_lag=5):
    print("===== RUNNING BASELINE ECONOMETRIC & CAUSALITY PROOFS =====")
    if not os.path.exists(df_path):
        drive_path = os.path.join(DRIVE_DIR, df_path)
        if os.path.exists(drive_path):
            df_path = drive_path
        else:
            raise FileNotFoundError(f"[ERROR] {df_path} missing. Run build_data.py first.")

    # Load master dataframe cleanly without index coercion
    df = pd.read_csv(df_path)
    print(f"[LOAD] Loaded multi-series panel ({len(df)} rows across {df['ticker'].nunique()} tickers).")

    tickers = df['ticker'].unique()

    # 1. INDEPENDENT GJR-GARCH ESTIMATION ACROSS ALL PANEL ASSETS
    print("\n[STEP 1] Fitting Empirical Skew-T GJR-GARCH(1,1) Across Panel Series...")
    for sym in tickers:
        sub = df[df['ticker'] == sym].sort_values(by="time_idx").dropna(subset=['Log_Ret'])
        returns = sub['Log_Ret'].values
        am = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='skewt')
        res = am.fit(disp='off')

        print(f"\n--- Estimated Parameters: {sym} ---")
        print(f"  Omega (Baseline Variance):   {res.params['omega']:.6f}")
        print(f"  Alpha (Symmetric Shock):      {res.params['alpha[1]']:.6f}")
        print(f"  Gamma (Asymmetric Leverage):  {res.params['gamma[1]']:.6f}")
        print(f"  Beta (GARCH Persistence):      {res.params['beta[1]']:.6f}")
        if 'nu' in res.params:
            print(f"  Nu (Tail Degrees of Freedom): {res.params['nu']:.4f}")
        if 'lambda' in res.params:
            print(f"  Lambda (Skew Parameter):      {res.params['lambda']:.4f}")

    # 2. CROSS-BORDER GRANGER CAUSALITY MATRIX (WALL STREET -> DALAL STREET)
    print("\n[STEP 2] Executing Vector Autoregression Granger Causality (NIFTY 50)...")
    nifty = df[df['ticker'] == 'NIFTY50'].sort_values(by="time_idx").copy()

    # Resolve feature names
    vix_col = 'US_VIX_Diff' if 'US_VIX_Diff' in nifty.columns else ('VIX_Diff' if 'VIX_Diff' in nifty.columns else None)
    if vix_col is None and 'US_VIX' in nifty.columns:
        nifty['US_VIX_Diff'] = nifty['US_VIX'].diff()
        vix_col = 'US_VIX_Diff'

    if 'India_VIX_Diff' not in nifty.columns:
        nifty['India_VIX_Diff'] = nifty['Log_Ret'].rolling(5).std().diff()

    clean_df = nifty[[vix_col, 'India_VIX_Diff']].dropna()
    lags = [1, 2, 3, 5]

    print("  -> Computing Wall Street to Dalal Street transmission matrix...")
    res_forward = grangercausalitytests(clean_df[['India_VIX_Diff', vix_col]], maxlag=max_lag, verbose=False)

    print("  -> Computing Dalal Street to Wall Street transmission matrix...")
    res_reverse = grangercausalitytests(clean_df[[vix_col, 'India_VIX_Diff']], maxlag=max_lag, verbose=False)

    print("\n--- Granger Causality Significance Matrix (p-values) ---")
    for l in lags:
        p_fwd = res_forward[l][0]['ssr_chi2test'][1]
        p_rev = res_reverse[l][0]['ssr_chi2test'][1]
        fwd_star = "***" if p_fwd < 0.01 else ("**" if p_fwd < 0.05 else "")
        rev_star = "***" if p_rev < 0.01 else ("**" if p_rev < 0.05 else "")
        print(f"  Lag {l} Day(s):")
        print(f"    - US VIX Granger-causes India Volatility: p = {p_fwd:.5f} {fwd_star}")
        print(f"    - India Volatility Granger-causes US VIX: p = {p_rev:.5f} {rev_star}")

    print("\n===== BASELINE DIAGNOSTICS COMPLETE =====")


if __name__ == "__main__":
    run_empirical_proofs()
