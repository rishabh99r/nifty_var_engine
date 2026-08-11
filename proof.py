# proof.py
import pandas as pd
import numpy as np
import os
import warnings
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

def run_empirical_proofs(df_path="master_df.csv", max_lag=5):
    print("===== RUNNING BASELINE ECONOMETRIC & CAUSALITY PROOFS =====")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"[ERROR] {df_path} missing. Run data collection first.")

    df = pd.read_csv(df_path, parse_dates=True, index_col=0)
    print(f"[LOAD] Loaded {len(df)} days for baseline diagnostics.")

    # 1. PROGRAMMATIC GJR-GARCH COEFFICIENT ESTIMATION
    print("\n[STEP 1] Fitting Empirical Skew-T GJR-GARCH(1,1)...")
    returns = df['Log_Ret'].dropna().values
    am = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='skewt')
    res = am.fit(disp='off')

    print("\n--- Estimated Parameters ---")
    print(f"  Omega (Baseline Variance): {res.params['omega']:.6f}")
    print(f"  Alpha (Symmetric Shock):    {res.params['alpha[1]']:.6f}")
    print(f"  Gamma (Asymmetric Leverage): {res.params['gamma[1]']:.6f}")
    print(f"  Beta (GARCH Persistence):    {res.params['beta[1]']:.6f}")
    if 'nu' in res.params:
        print(f"  Nu (Tail Thickness):         {res.params['nu']:.4f}")
    if 'lambda' in res.params:
        print(f"  Lambda (Skewness):           {res.params['lambda']:.4f}")

    # 2. STATIONARY DYNAMIC GRANGER CAUSALITY MATRIX
    print("\n[STEP 2] Executing Vector Autoregression Granger Causality...")

    # Ensure volatility differential trackers are present and clean
    if 'US_VIX' in df.columns and 'VIX_Diff' not in df.columns:
        df['VIX_Diff'] = df['US_VIX'].diff()
    if 'India_VIX_Diff' not in df.columns:
        df['India_VIX_Diff'] = df['Log_Ret'].rolling(5).std().diff()

    clean_df = df[['VIX_Diff', 'India_VIX_Diff']].dropna()
    lags = [1, 2, 3, 5]

    print("  -> Computing Wall Street to Dalal Street transmission matrix...")
    res_forward = grangercausalitytests(clean_df[['India_VIX_Diff', 'VIX_Diff']], maxlag=max_lag, verbose=False)

    print("  -> Computing Dalal Street to Wall Street transmission matrix...")
    res_reverse = grangercausalitytests(clean_df[['VIX_Diff', 'India_VIX_Diff']], maxlag=max_lag, verbose=False)

    print("\n--- Granger Causality Significance Matrix (p-values) ---")
    for l in lags:
        p_fwd = res_forward[l][0]['ssr_chi2test'][1]
        p_rev = res_reverse[l][0]['ssr_chi2test'][1]
        print(f"  Lag {l} Day(s):")
        print(f"    - US VIX Granger-causes India Volatility: p = {p_fwd:.5f} " + ("*" if p_fwd < 0.05 else ""))
        print(f"    - India Volatility Granger-causes US VIX: p = {p_rev:.5f} " + ("*" if p_rev < 0.05 else ""))

    print("\n===== BASELINE DIAGNOSTICS COMPLETE =====")

if __name__ == "__main__":
    run_empirical_proofs()
