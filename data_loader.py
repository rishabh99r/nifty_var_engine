import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

def fetch_cpu_index(cache_file="CPU_index.csv"):
    if os.path.exists(cache_file):
        cpu_df = pd.read_csv(cache_file, skiprows=4, usecols=[0, 1])
        cpu_df.columns = ['Date', 'Global_CPU']
        cpu_df['Date'] = pd.to_datetime(cpu_df['Date'], format='%b-%y', errors='coerce')
        cpu_df['Global_CPU'] = pd.to_numeric(cpu_df['Global_CPU'], errors='coerce')
        cpu_df.dropna(subset=['Date'], inplace=True)
        cpu_df.set_index('Date', inplace=True)
        return cpu_df.resample('D').ffill()

    dates = pd.date_range(start='2007-01-01', end=pd.Timestamp.now())
    return pd.DataFrame({'Global_CPU': 100.0}, index=dates)

def fetch_and_clean_data(start_date="2007-01-01"):
    print("[LOADER] Fetching Nifty 50 and macroeconomic drivers from Yahoo Finance...")
    tickers = {"Nifty50": "^NSEI", "US_VIX": "^VIX", "US_10Y": "^TNX", "DXY": "DX-Y.NYB", "Crude_Oil": "CL=F"}

    data = pd.DataFrame()
    for name, ticker in tickers.items():
        raw = yf.download(ticker, start=start_date, progress=False)['Close']
        if isinstance(raw, pd.DataFrame): raw = raw.iloc[:, 0]
        data[name] = raw

    cpu_df = fetch_cpu_index()
    data = data.join(cpu_df, how='left')
    data.ffill(inplace=True)
    data.dropna(inplace=True)

    df = pd.DataFrame(index=data.index)
    df["Log_Ret"] = (np.log(data["Nifty50"] / data["Nifty50"].shift(1)) * 100)

    df["Log_Ret_Lag1"] = df["Log_Ret"].shift(1)
    df["Log_Ret_Lag2"] = df["Log_Ret"].shift(2)

    df["VIX_Diff"] = data["US_VIX"].diff()
    df["US_10Y_Diff"] = data["US_10Y"].diff()
    df["DXY_Ret"] = (np.log(data["DXY"] / data["DXY"].shift(1)) * 100)
    df["Crude_Oil_Ret"] = (np.log(data["Crude_Oil"] / data["Crude_Oil"].shift(1)) * 100)
    df["Global_CPU_Ret"] = data["Global_CPU"]

    df.dropna(inplace=True)
    print(f"[LOADER] Feature engineering complete. Dataset shape: {df.shape}")
    return df
