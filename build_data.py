# build_data.py
import os
import glob
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore")


def compute_garman_klass(df):
    """
    Computes Garman-Klass historical volatility proxy from daily OHLC prices.
    Requires columns: 'Open', 'High', 'Low', 'Close'.
    Up to 8x more statistically efficient than close-to-close squared returns.
    """
    df = df[(df['High'] > 0) & (df['Low'] > 0) & (df['Open'] > 0) & (df['Close'] > 0)].copy()

    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])

    # Garman-Klass Variance Formula
    df['GK_Variance'] = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

    # Convert to annualized percentage standard deviation
    df['GK_Vol'] = np.sqrt(np.maximum(df['GK_Variance'], 1e-8) * 252) * 100
    return df


def purge_stale_artifacts():
    """
    Scans the runtime directory and aggressively wipes out stale data cache,
    temporary validation checkpoints, and incomplete model optimization tracking databases.
    """
    print("\n=== SYSTEM SANITIZATION: PURGING STALE ARTIFACTS ===")

    stale_files = [
        "master_df.csv",
        "test_tft_predictions.csv",
        "tft_nifty_optimization.db",
        "model_validation_master_report.txt"
    ]

    for file in stale_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  -> Successfully deleted stale file: {file}")
            except Exception as e:
                print(f"  [WARNING] Could not purge {file}: {str(e)}")

    # Wipe out residual PyTorch Lightning checkpoints from aborted training sessions
    stale_checkpoints = glob.glob("lightning_logs/*/checkpoints/*.ckpt")
    for ckpt in stale_checkpoints:
        try:
            os.remove(ckpt)
            print(f"  -> Purged abandoned model checkpoint: {ckpt}")
        except Exception:
            pass

    print("=== WORKSPACE CLEAR: LAUNCHING FRESH DATA ENGINEERING PIPELINE ===\n")


def clean_yf_columns(raw_df):
    """Flattens potential MultiIndex columns from recent yfinance updates."""
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    return raw_df.copy()


def fit_gjr_garch_per_series(returns_series):
    """
    Fits Skew-T GJR-GARCH(1,1) model on log returns to extract conditional volatility,
    scale-free standardized innovations (z_t), and the parametric 99% VaR floor.
    """
    am = arch_model(returns_series, mean='Constant', vol='Garch', p=1, o=1, q=1, dist='skewt')
    res = am.fit(disp='off')

    vol = res.conditional_volatility
    resid = res.std_resid  # z_t = (r_t - mu) / sigma_t (standardized innovation)

    # Parametric 99% quantile boundary (Student-t inverse CDF)
    q_dist = res.model.distribution.ppf(0.01, res.params[-2:])
    mu = res.params.get('mu', 0.0)
    var_99 = mu + vol * q_dist

    return vol, resid, var_99


def generate_clean_production_data():
    # 1. Trigger the isolation purge
    purge_stale_artifacts()

    print("[ETL] Fetching historical index panel and cross-border volatility...")
    start_date = "2015-01-01"
    end_date = "2026-08-01"

    # Define the domestic multi-series panel tickers
    tickers = {
        'NIFTY50': '^NSEI',
        'BANKNIFTY': '^NSEBANK',
        'NIFTYIT': '^CNXIT'
    }

    # 2. Download and prepare exogenous macro volatility (US VIX)
    raw_vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if raw_vix.empty:
        raise ValueError("[FATAL] Yahoo Finance API failure: Unable to download ^VIX.")

    raw_vix = clean_yf_columns(raw_vix)
    vix_close = raw_vix['Close'].dropna()

    # Strict Point-in-Time (PIT) lag enforcement: shift by 1 to prevent lookahead bias
    macro_df = pd.DataFrame(index=vix_close.index)
    macro_df['US_VIX'] = vix_close.shift(1)
    macro_df['US_VIX_Diff'] = vix_close.diff().shift(1)
    macro_df['VIX_Diff'] = macro_df['US_VIX_Diff']  # Dual-alias for backward compatibility
    macro_df = macro_df.dropna()

    # 3. Process each ticker in the multi-series panel
    ticker_dfs = []

    for label, symbol in tickers.items():
        print(f"[ETL] Downloading and processing series: {label} ({symbol})...")
        raw_ticker = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if raw_ticker.empty:
            raise ValueError(f"[FATAL] Failed downloading data for {symbol}.")

        raw_ticker = clean_yf_columns(raw_ticker)
        df = raw_ticker[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()

        # Compute percentage log returns
        df['Log_Ret'] = 100 * np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()

        # Compute Garman-Klass range-based volatility proxy
        df = compute_garman_klass(df)

        # Fit Stage 1 Skew-T GJR-GARCH filter
        vol, resid, var_99 = fit_gjr_garch_per_series(df['Log_Ret'])
        df['GARCH_Vol'] = vol
        df['GARCH_sigma'] = vol       # Dual-alias for TFT known_reals
        df['GARCH_Resid'] = resid
        df['GARCH_resid'] = resid     # Dual-alias for TFT unknown_reals
        df['GARCH_VaR_99'] = var_99

        # Lag features (PIT-safe, observable at t-1)
        df['Log_Ret_Lag1'] = df['Log_Ret'].shift(1)
        df['Log_Ret_Lag2'] = df['Log_Ret'].shift(2)
        df['India_VIX_Diff'] = df['Log_Ret'].rolling(5).std().diff().shift(1)

        # Join cross-border macro features
        df = df.join(macro_df, how='inner')
        df = df.dropna()

        df['ticker'] = label
        df['Date'] = df.index.strftime('%Y-%m-%d')
        ticker_dfs.append(df)

    # 4. Synchronize trading dates across all panel tickers
    common_dates = sorted(list(set(ticker_dfs[0]['Date']).intersection(*[set(d['Date']) for d in ticker_dfs[1:]])))
    date_to_time_idx = {d: i for i, d in enumerate(common_dates)}

    aligned_dfs = []
    for df in ticker_dfs:
        df_aligned = df[df['Date'].isin(common_dates)].copy()
        df_aligned['time_idx'] = df_aligned['Date'].map(date_to_time_idx)
        aligned_dfs.append(df_aligned)

    # 5. Concatenate and sort panel long matrix
    master_df = pd.concat(aligned_dfs, ignore_index=True)
    master_df = master_df.sort_values(by=['time_idx', 'ticker']).reset_index(drop=True)

    output_path = "master_df.csv"
    master_df.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Reconstructed clean multi-series panel at: {output_path}")
    print(f"Total Observations: {len(master_df)} rows across {len(tickers)} tickers.")
    print(f"Common Lookback Steps (time_idx): 0 to {master_df['time_idx'].max()} ({len(common_dates)} days).")


if __name__ == "__main__":
    generate_clean_production_data()
