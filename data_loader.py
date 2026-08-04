# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def fetch_and_clean_data(start_date="2007-01-01"):
    print("[LOADER] Fetching Nifty 50 and macroeconomic drivers from Yahoo Finance...")

    tickers = {
        "Nifty50": "^NSEI",
        "US_VIX": "^VIX",
        "US_10Y": "^TNX",
        "DXY": "DX-Y.NYB",
        "Crude_Oil": "CL=F"
    }

    data = pd.DataFrame()
    for name, ticker in tickers.items():
        raw = yf.download(ticker, start=start_date, progress=False)['Close']
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        data[name] = raw

    # Forward fill missing calendar dates across international exchanges
    data.ffill(inplace=True)
    data.dropna(inplace=True)

    df = pd.DataFrame(index=data.index)

    # Target Variable: Log Returns
    df["Log_Ret"] = (np.log(data["Nifty50"] / data["Nifty50"].shift(1)) * 100)

    # INJECTED: Autoregressive lags to solve attention-head laziness
    df["Log_Ret_Lag1"] = df["Log_Ret"].shift(1)
    df["Log_Ret_Lag2"] = df["Log_Ret"].shift(2)

    # Exogenous Macro Drivers
    df["VIX_Diff"] = data["US_VIX"].diff()
    df["US_10Y_Diff"] = data["US_10Y"].diff()
    df["DXY_Ret"] = (np.log(data["DXY"] / data["DXY"].shift(1)) * 100)
    df["Crude_Oil_Ret"] = (np.log(data["Crude_Oil"] / data["Crude_Oil"].shift(1)) * 100)

    # Simulate / Proxy Global Economic Policy Uncertainty (Z-score normalized level)
    # In live deployment, replace this with your actual imported EPU/CPU index
    df["Global_CPU_Ret"] = np.random.normal(0, 1, size=len(df))

    # Clean up NaN values introduced by shift() and diff()
    df.dropna(inplace=True)

    print(f"[LOADER] Feature engineering complete. Dataset shape: {df.shape}")
    return df
