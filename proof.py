# proofs.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

def run_garch_supremacy_audit(csv_path="master_df.csv"):
    print("\n[PROOF] === Empirical Supremacy Audit: GARCH vs. GJR-GARCH vs. EGARCH ===")

    if os.path.exists(csv_path):
        print(f"[PROOF] Loading existing market returns from {csv_path}...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        returns = df['Log_Ret'].dropna()
    else:
        print("[PROOF] Local dataset not found. Fetching Nifty 50 from Yahoo Finance...")
        data = yf.download('^NSEI', start="2007-01-01", progress=False)['Close']
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        returns = (np.log(data / data.shift(1)) * 100).dropna()

    print(f"[PROOF] Sample Size: {len(returns)} trading days.")
    print("[PROOF] Fitting competing volatility specifications with Skewed Student-t errors...\n")

    # 1. Symmetric GARCH(1,1)
    garch_sym = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='skewt').fit(disp='off')

    # 2. Asymmetric GJR-GARCH(1,1)
    gjr_garch = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='skewt').fit(disp='off')

    # 3. Asymmetric EGARCH(1,1)
    egarch = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='skewt').fit(disp='off')

    # Likelihood Ratio Test: Symmetric vs GJR
    lr_stat = -2 * (garch_sym.loglikelihood - gjr_garch.loglikelihood)
    lr_pval = 1 - stats.chi2.cdf(lr_stat, df=1)

    # Leverage coefficient significance
    gjr_gamma = gjr_garch.params.get('gamma[1]', np.nan)
    gjr_gamma_pval = gjr_garch.pvalues.get('gamma[1]', np.nan)
    egarch_asym = egarch.params.get('gamma[1]', egarch.params.get('alpha[1]', np.nan))
    egarch_pval = egarch.pvalues.get('gamma[1]', egarch.pvalues.get('alpha[1]', np.nan))

    print(f"{'Model':<18} | {'Log-Likelihood':<15} | {'AIC':<12} | {'BIC':<12} | {'Asymmetry (p-val)'}")
    print("-" * 75)
    print(f"{'GARCH(1,1)':<18} | {garch_sym.loglikelihood:<15.2f} | {garch_sym.aic:<12.2f} | {garch_sym.bic:<12.2f} | None (Symmetric)")
    print(f"{'GJR-GARCH(1,1)':<18} | {gjr_garch.loglikelihood:<15.2f} | {gjr_garch.aic:<12.2f} | {gjr_garch.bic:<12.2f} | γ={gjr_gamma:.4f} (p={gjr_gamma_pval:.4e})")
    print(f"{'EGARCH(1,1)':<18} | {egarch.loglikelihood:<15.2f} | {egarch.aic:<12.2f} | {egarch.bic:<12.2f} | α/γ={egarch_asym:.4f} (p={egarch_pval:.4e})")
    print("-" * 75)

    print(f"\n[PROOF] Likelihood Ratio Test (GARCH vs. GJR-GARCH):")
    print(f"  -> LR Statistic: {lr_stat:.4f}")
    print(f"  -> p-value:      {lr_pval:.4e}")
    if lr_pval < 0.01:
        print("  -> CONCLUSION:   Symmetric GARCH is strongly rejected at 1% significance level.")
        print("                   Asymmetric leverage effect is empirically critical for Nifty 50.")
    else:
        print("  -> CONCLUSION:   No statistically significant leverage effect detected.")

def run_volatility_spillover_audit():
    print("\n[PROOF] === Volatility Spillover Audit: US VIX vs India VIX ===")

    # 1. Fetch Data
    vix_data = yf.download(['^VIX', '^INDIAVIX'], start="2008-03-02", progress=False)['Close']

    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    vix_data.columns = ['India_VIX', 'US_VIX']
    vix_data.ffill(inplace=True)
    vix_data.dropna(inplace=True)

    # 2. Stationarity Fix & Time-Zone Alignment
    df_test = pd.DataFrame()
    df_test['India_VIX_Diff'] = vix_data['India_VIX'].diff()
    df_test['US_VIX_Diff_Lag1'] = vix_data['US_VIX'].diff().shift(1)
    df_test.dropna(inplace=True)

    # 3. Granger Causality
    print("\n[PROOF] Direction 1: US VIX (t-1) -> India VIX (t) [Wall Street -> Dalal Street]")
    res_1 = grangercausalitytests(df_test[['India_VIX_Diff', 'US_VIX_Diff_Lag1']], maxlag=[1, 2, 3, 5], verbose=False)
    for lag in [1, 2, 3, 5]:
        p_val = res_1[lag][0]['ssr_ftest'][1]
        status = '✅ Causality Proven' if p_val < 0.05 else '❌ No Causality'
        print(f"Lag {lag} Days | P-Value: {p_val:.4f} | {status}")

    print("\n[PROOF] Direction 2: India VIX (t) -> US VIX (t-1) [Dalal Street -> Wall Street]")
    res_2 = grangercausalitytests(df_test[['US_VIX_Diff_Lag1', 'India_VIX_Diff']], maxlag=[1, 2, 3, 5], verbose=False)
    for lag in [1, 2, 3, 5]:
        p_val = res_2[lag][0]['ssr_ftest'][1]
        status = '✅ Causality Proven' if p_val < 0.05 else '❌ No Causality'
        print(f"Lag {lag} Days | P-Value: {p_val:.4f} | {status}")

if __name__ == "__main__":
    run_garch_supremacy_audit()
    run_volatility_spillover_audit()
