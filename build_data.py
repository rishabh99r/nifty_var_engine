# build_data.py
import os
import glob
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore")

LOOKBACK_DAYS = 1000
REFIT_FREQ = 21  # Monthly parameter re-estimation


def compute_garman_klass(df):
    """
    Computes Garman-Klass historical volatility proxy from daily OHLC prices.
    Requires columns: 'Open', 'High', 'Low', 'Close'.
    """
    df = df[(df['High'] > 0) & (df['Low'] > 0) & (df['Open'] > 0) & (df['Close'] > 0)].copy()

    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])

    df['GK_Variance'] = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    df['GK_Vol'] = np.sqrt(np.maximum(df['GK_Variance'], 1e-8) * 252) * 100
    return df


def purge_stale_artifacts():
    """Wipes out stale data cache, validation checkpoints, and tracking files."""
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


def rolling_gjr_garch_pit(returns_series, lookback=LOOKBACK_DAYS, refit_freq=REFIT_FREQ):
    """
    Point-in-Time Rolling Skew-T GJR-GARCH(1,1) filter.
    Parameters re-estimated every `refit_freq` steps over historical slice [t-lookback : t-1].
    Conditional variance sigma_t and VaR_99 are strictly F_{t-1}-measurable.
    """
    T = len(returns_series)
    vol_arr = np.full(T, np.nan)
    resid_arr = np.full(T, np.nan)
    var99_arr = np.full(T, np.nan)

    current_res = None
    last_params = {}
    last_q_dist = -2.326

    print(f"  -> Running PIT rolling GJR-GARCH across {T} periods (warm-up: {lookback} days)...")

    for t in range(lookback, T):
        # 1. Periodic Parameter Re-estimation
        if (t - lookback) % refit_freq == 0 or current_res is None:
            train_slice = returns_series.iloc[t - lookback : t]
            am = arch_model(train_slice, mean='Constant', vol='Garch', p=1, o=1, q=1, dist='skewt')
            try:
                current_res = am.fit(disp='off', show_warning=False)
                last_params = {
                    'mu': current_res.params.get('mu', 0.0),
                    'omega': current_res.params['omega'],
                    'alpha': current_res.params['alpha[1]'],
                    'gamma': current_res.params['gamma[1]'],
                    'beta': current_res.params['beta[1]']
                }
                # Parametric 99% quantile boundary (Skew-T inverse CDF)
                last_q_dist = current_res.model.distribution.ppf(0.01, current_res.params[-2:])
            except Exception:
                # Retain previous parameters if numerical MLE fails
                pass

        # 2. Daily Variance Recursion at t using shock from t-1
        prev_r = returns_series.iloc[t - 1]
        prev_vol = vol_arr[t - 1] if not np.isnan(vol_arr[t - 1]) else current_res.conditional_volatility.iloc[-1]

        eps_prev = prev_r - last_params['mu']
        leverage_ind = 1.0 if eps_prev < 0.0 else 0.0

        sigma2_t = (
            last_params['omega']
            + (last_params['alpha'] + last_params['gamma'] * leverage_ind) * (eps_prev ** 2)
            + last_params['beta'] * (prev_vol ** 2)
        )
        sigma_t = np.sqrt(max(sigma2_t, 1e-6))

        vol_arr[t] = sigma_t
        resid_arr[t] = (returns_series.iloc[t] - last_params['mu']) / sigma_t
        var99_arr[t] = last_params['mu'] + sigma_t * last_q_dist

    return pd.Series(vol_arr, index=returns_series.index), \
           pd.Series(resid_arr, index=returns_series.index), \
           pd.Series(var99_arr, index=returns_series.index)


def generate_clean_production_data():
    purge_stale_artifacts()

    print("[ETL] Fetching historical index panel and cross-border volatility...")
    start_date = "2015-01-01"
    end_date = "2026-08-01"

    tickers = {
        'NIFTY50': '^NSEI',
        'BANKNIFTY': '^NSEBANK',
        'NIFTYIT': '^CNXIT'
    }

    # 1. Macro exogenous feature alignment (strictly lagged to t-1)
    raw_vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if raw_vix.empty:
        raise ValueError("[FATAL] Yahoo Finance API failure: Unable to download ^VIX.")

    raw_vix = clean_yf_columns(raw_vix)
    vix_close = raw_vix['Close'].dropna()

    macro_df = pd.DataFrame(index=vix_close.index)
    macro_df['US_VIX'] = vix_close.shift(1)
    macro_df['US_VIX_Diff'] = vix_close.diff().shift(1)
    macro_df['VIX_Diff'] = macro_df['US_VIX_Diff']
    macro_df = macro_df.dropna()

    # 2. Process domestic index panel
    ticker_dfs = []

    for label, symbol in tickers.items():
        print(f"[ETL] Downloading and processing series: {label} ({symbol})...")
        raw_ticker = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if raw_ticker.empty:
            raise ValueError(f"[FATAL] Failed downloading data for {symbol}.")

        raw_ticker = clean_yf_columns(raw_ticker)
        df = raw_ticker[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()

        df['Log_Ret'] = 100 * np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()

        df = compute_garman_klass(df)

        # Apply Point-in-Time rolling GJR-GARCH
        vol, resid, var_99 = rolling_gjr_garch_pit(df['Log_Ret'], lookback=LOOKBACK_DAYS, refit_freq=REFIT_FREQ)

        # Preserve all exact schema aliases expected by downstream modules
        df['GARCH_Vol'] = vol
        df['GARCH_sigma'] = vol
        df['GARCH_Resid'] = resid
        df['GARCH_resid'] = resid
        df['GARCH_VaR_99'] = var_99

        df['Log_Ret_Lag1'] = df['Log_Ret'].shift(1)
        df['Log_Ret_Lag2'] = df['Log_Ret'].shift(2)
        df['India_VIX_Diff'] = df['Log_Ret'].rolling(5).std().diff().shift(1)

        df = df.join(macro_df, how='inner')
        df = df.dropna()  # Purges the initial 1000-day warm-up period cleanly

        df['ticker'] = label
        df['Date'] = df.index.strftime('%Y-%m-%d')
        ticker_dfs.append(df)

    # 3. Synchronize trading dates across all panel tickers
    common_dates = sorted(list(set(ticker_dfs[0]['Date']).intersection(*[set(d['Date']) for d in ticker_dfs[1:]])))
    date_to_time_idx = {d: i for i, d in enumerate(common_dates)}

    aligned_dfs = []
    for df in ticker_dfs:
        df_aligned = df[df['Date'].isin(common_dates)].copy()
        df_aligned['time_idx'] = df_aligned['Date'].map(date_to_time_idx)
        aligned_dfs.append(df_aligned)

    master_df = pd.concat(aligned_dfs, ignore_index=True)
    master_df = master_df.sort_values(by=['time_idx', 'ticker']).reset_index(drop=True)

    output_path = "master_df.csv"
    master_df.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Reconstructed clean multi-series panel at: {output_path}")
    print(f"Total Observations: {len(master_df)} rows across {len(tickers)} tickers.")
    print(f"Common Lookback Steps (time_idx): 0 to {master_df['time_idx'].max()} ({len(common_dates)} days).")


if __name__ == "__main__":
    generate_clean_production_data()
