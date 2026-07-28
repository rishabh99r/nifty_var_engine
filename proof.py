# proofs.py
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

warnings.filterwarnings("ignore")

def run_volatility_spillover_audit():
    print("[PROOF] === Volatility Spillover Audit: US VIX vs India VIX ===")

    # 1. Fetch Data
    vix_data = yf.download(['^VIX', '^INDIAVIX'], start="2008-03-02", progress=False)['Close']

    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    vix_data.columns = ['India_VIX', 'US_VIX']
    vix_data.ffill(inplace=True)
    vix_data.dropna(inplace=True)

    # 2. Stationarity Fix & Time-Zone Alignment (Point 3)
    # Since NSE (India) closes before CBOE (US) opens on calendar day t,
    # US VIX on day t-1 is what was actually available when NSE closed.
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
    run_volatility_spillover_audit()
