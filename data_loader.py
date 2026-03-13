# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
import urllib3
import os
from statsmodels.tsa.stattools import adfuller
from config import START_DATE

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def stationarity_audit(df, columns, title):
    print(f"\n[AUDIT] --- {title} ---")
    print(f"{'Variable':<15} | {'ADF Stat':<15} | {'p-value':<10} | {'Status'}")
    print("-" * 65)

    skip_vars = ['Global_CPU']

    for col in columns:
        if col in skip_vars:
            print(f"{col:<15} | {'N/A':<15} | {'N/A':<10} | ABSOLUTE PROXY (SKIP)")
            continue

        series = df[col].dropna()
        if series.std() == 0:
            print(f"{col:<15} | {'N/A':<15} | {'N/A':<10} | CONSTANT (SKIP)")
            continue

        result = adfuller(series)
        status = "STATIONARY" if result[1] < 0.05 else "NON-STATIONARY"
        print(f"{col:<15} | {result[0]:<15.4f} | {result[1]:<10.4f} | {status}")

def fetch_cpu_index(cache_file="CPU_index.csv"):
    if os.path.exists(cache_file):
        print(f"[DATA] Loading Global CPU Index from {cache_file}...")
        cpu_df = pd.read_csv(cache_file, skiprows=4, usecols=[0, 1])
        cpu_df.columns = ['Date', 'Global_CPU']
        cpu_df['Date'] = pd.to_datetime(cpu_df['Date'], format='%b-%y', errors='coerce')
        cpu_df['Global_CPU'] = pd.to_numeric(cpu_df['Global_CPU'], errors='coerce')
        cpu_df.dropna(subset=['Date'], inplace=True)
        cpu_df.set_index('Date', inplace=True)
        return cpu_df.resample('D').ffill()

    print("[WARNING] CPU_index.csv not found. Ensure it is committed to the repo.")
    print("[WARNING] Falling back to placeholder value of 100.0 for all dates.")
    dates = pd.date_range(start='2007-01-01', end=pd.Timestamp.now())
    return pd.DataFrame({'Global_CPU': 100.0}, index=dates)

def fetch_and_clean_data():
    print(f"\n[PIPELINE] Initializing Data Ingestion from {START_DATE}...")

    assets = {'^NSEI': 'Nifty50', '^VIX': 'VIX', 'CL=F': 'Crude_Oil', '^TNX': 'US_10Y', 'DX=F': 'DXY'}
    print(f"[DATA] Downloading Yahoo Finance tickers: {list(assets.values())}...")
    raw_data = yf.download(list(assets.keys()), start=START_DATE, progress=False)['Close']

    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
    raw_data.rename(columns=assets, inplace=True)

    cpu_df = fetch_cpu_index()
    raw_data = raw_data.join(cpu_df, how='left')

    print("[DATA] Forward filling missing dates (holidays/weekends)...")
    raw_data.ffill(inplace=True)
    raw_data.dropna(inplace=True)

    stationarity_audit(raw_data, raw_data.columns, title="PRE-TRANSFORMATION (Raw Prices/Levels)")

    print("\n[TRANSFORM] Executing Asset-Specific Transformations...")
    df = pd.DataFrame(index=raw_data.index)

    print("  -> Log Returns:       Nifty50, Crude_Oil, DXY")
    df['Log_Ret'] = np.log(raw_data['Nifty50'] / raw_data['Nifty50'].shift(1)) * 100
    df['Crude_Oil_Ret'] = np.log(raw_data['Crude_Oil'] / raw_data['Crude_Oil'].shift(1)) * 100
    df['DXY_Ret'] = np.log(raw_data['DXY'] / raw_data['DXY'].shift(1)) * 100

    print("  -> First Differences: US_10Y, VIX")
    df['US_10Y_Diff'] = raw_data['US_10Y'] - raw_data['US_10Y'].shift(1)
    df['VIX_Diff'] = raw_data['VIX'] - raw_data['VIX'].shift(1)

    # Using raw index level (not pct_change) for Global_CPU.
    # The CPU data is monthly, forward-filled to daily frequency.
    # pct_change() on forward-filled data produces ~95% zeros (21 flat days per month),
    # making the feature near-useless to the TFT. The raw level is the meaningful signal.
    # Column name kept as Global_CPU_Ret for downstream compatibility.
    print("  -> Raw Level:         Global_CPU (pct_change avoided — see comment in code)")
    df['Global_CPU_Ret'] = raw_data['Global_CPU']

    print("[TRANSFORM] Dropping initial NaN row from shift()...")
    df.dropna(inplace=True)

    missing_count = df.isna().sum().sum()
    print(f"\n[CHECK] Missing values in final dataset: {missing_count}")
    if missing_count > 0:
        print("  WARNING: NaNs detected after transformations. Investigate before proceeding.")

    stationarity_audit(df, df.columns, title="POST-TRANSFORMATION")

    print(f"\n[PIPELINE] Data Module Complete. Final Trading Days: {len(df)}")
    return df

if __name__ == "__main__":
    fetch_and_clean_data()
