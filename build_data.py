# build_data.py
# =============================================================================
# Data engineering pipeline producing the multi-series master panel.
#  - Ingests US VIX and attempts REAL India VIX (^INDIAVIX).
#  - Falls back to an honestly-labelled Domestic_RV_Proxy if India VIX history
#    is insufficient. The feature is NEVER mislabelled as "India VIX".
#  - Applies a Point-in-Time rolling Skew-T GJR-GARCH(1,1) filter.
# All configuration is imported from config.py (single source of truth).
# =============================================================================
import glob
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

import config

warnings.filterwarnings("ignore")


def compute_garman_klass(df):
    """Garman-Klass historical volatility proxy from daily OHLC prices."""
    df = df[(df["High"] > 0) & (df["Low"] > 0) & (df["Open"] > 0) & (df["Close"] > 0)].copy()

    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])

    df["GK_Variance"] = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    df["GK_Vol"] = np.sqrt(np.maximum(df["GK_Variance"], 1e-8) * 252) * 100
    return df


def purge_stale_artifacts():
    """Wipes stale data cache, checkpoints, and tracking files."""
    print("\n=== SYSTEM SANITIZATION: PURGING STALE ARTIFACTS ===")

    stale_files = [
        "master_df.csv",
        "test_tft_predictions.csv",
        "tft_nifty_optimization.db",
        "model_validation_master_report.txt",
    ]
    for file in stale_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  -> Deleted stale file: {file}")
            except Exception as e:
                print(f"  [WARNING] Could not purge {file}: {str(e)}")

    for ckpt in glob.glob("lightning_logs/*/checkpoints/*.ckpt"):
        try:
            os.remove(ckpt)
            print(f"  -> Purged abandoned checkpoint: {ckpt}")
        except Exception:
            pass

    print("=== WORKSPACE CLEAR: LAUNCHING FRESH DATA ENGINEERING PIPELINE ===\n")


def clean_yf_columns(raw_df):
    """Flattens potential MultiIndex columns from recent yfinance updates."""
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    return raw_df.copy()


def fetch_vix_pair():
    """
    Fetches US VIX (^VIX) and, if available, REAL India VIX (^INDIAVIX).

    Returns (us_vix_close, india_vix_close, used_real_india: bool).
    If the real India VIX has too few observations, returns None for it and the
    caller falls back to a Domestic_RV_Proxy computed from index returns.
    """
    raw_us = yf.download("^VIX", start=config.START_DATE, end=config.END_DATE, progress=False)
    if raw_us.empty:
        raise ValueError("[FATAL] Yahoo Finance API failure: Unable to download ^VIX.")

    raw_us = clean_yf_columns(raw_us)
    us_vix_close = raw_us["Close"].dropna()

    india_vix_close = None
    try:
        raw_in = yf.download(config.INDIA_VIX_SYMBOL, start=config.START_DATE, end=config.END_DATE, progress=False)
        raw_in = clean_yf_columns(raw_in)
        if not raw_in.empty and "Close" in raw_in.columns:
            india_vix_close = raw_in["Close"].dropna()
    except Exception as e:
        print(f"  [INFO] India VIX download failed ({e}); falling back to RV proxy.")

    used_real = False
    if india_vix_close is not None and len(india_vix_close) >= config.MIN_INDIA_VIX_OBS:
        used_real = True
    else:
        india_vix_close = None

    return us_vix_close, india_vix_close, used_real


def build_macro_features(us_vix_close, india_vix_close, ticker_index):
    """
    Builds macro feature columns ALIGNED to the ticker trading calendar
    (`ticker_index`), with timezone-correct lags (anti look-ahead bias).

    TWO rules are enforced:

    1. ANTI-ARTIFACT (Granger): log-differences are computed on the NATIVE
       VIX calendar (only genuinely observed days), then the resulting daily
       changes are reindexed onto the ticker calendar with ffill. Differencing
       an ffill()-ed LEVEL on the ticker calendar would create artificial zero
       returns on any market-calendar mismatch day, deflating variance and
       inflating Granger significance. We avoid that entirely here.

    2. ANTI-LOOKAHEAD (timezone): The US VIX closes at 01:30 IST on the NEXT
       calendar day. The Indian market closes at 15:30 IST on day D, BEFORE
       the US session for day D even opens. So the value a model may use to
       forecast India's day-D+1 VaR (generated at India's close on D) must
       reflect only information known by end-of-day D. We therefore lag the
       US features by config.US_VIX_SHIFT (=2) calendar rows after reindexing:
       US_VIX at row D holds US[D-1], US_VIX_Diff at row D holds
       log(US[D-1]/US[D-2]) -- both fully known by India's close on D.
       India VIX is same-zone (closes 15:30 IST on D) so config.INDIA_VIX_SHIFT
       (=1) is sufficient.
    """
    us_sh = config.US_VIX_SHIFT
    in_sh = config.INDIA_VIX_SHIFT

    # Native-calendar daily log-changes (non-overlapping, no ffill zeros)
    us_log_change = np.log(us_vix_close).diff().dropna()
    # Reindex the CHANGES onto the ticker calendar (ffill of changes is safe:
    # carry forward the last realized change, never invent a zero change).
    us_change_reindexed = us_log_change.reindex(ticker_index).ffill()
    us_level_reindexed = us_vix_close.reindex(ticker_index).ffill()

    macro_df = pd.DataFrame(index=ticker_index)
    macro_df["US_VIX"] = us_level_reindexed.shift(us_sh)
    macro_df["US_VIX_Diff"] = us_change_reindexed.shift(us_sh)
    # Native level, timezone-shifted as well (reference/level feature).
    macro_df["US_VIX_Level"] = us_level_reindexed.shift(us_sh)

    if india_vix_close is not None:
        in_log_change = np.log(india_vix_close).diff().dropna()
        in_change_reindexed = in_log_change.reindex(ticker_index).ffill()
        in_level_reindexed = india_vix_close.reindex(ticker_index).ffill()

        macro_df["India_VIX"] = in_level_reindexed.shift(in_sh)
        macro_df["India_VIX_Diff"] = in_change_reindexed.shift(in_sh)
        macro_df["India_VIX_Level"] = in_level_reindexed.shift(in_sh)
        macro_df["has_real_india_vix"] = True
    else:
        macro_df["has_real_india_vix"] = False

    return macro_df


def compute_domestic_rv_proxy(returns_series):
    """
    Honest realized-volatility proxy: lagged first-difference of the 5-day
    rolling std of returns. Explicitly NOT called "India VIX".
    """
    rv = returns_series.rolling(5).std()
    return rv.diff().shift(1)


def rolling_gjr_garch_pit(returns_series):
    """
    Point-in-Time rolling Skew-T GJR-GARCH(1,1) filter.
    Parameters re-estimated every config.REFIT_FREQ steps on the slice
    [t-lookback : t-1]. Conditional variance and VaR_99 are strictly
    F_{t-1}-measurable.
    """
    lookback = config.LOOKBACK_DAYS
    refit_freq = config.REFIT_FREQ
    T = len(returns_series)

    vol_arr = np.full(T, np.nan)
    resid_arr = np.full(T, np.nan)
    var99_arr = np.full(T, np.nan)

    current_res = None
    last_params = {}
    # Default to standard normal quantile until first successful skew-t refit
    last_q_dist = -2.326

    print(f"  -> Running PIT rolling GJR-GARCH across {T} periods (warm-up: {lookback} days)...")

    for t in range(lookback, T):
        # 1. Periodic parameter re-estimation
        if (t - lookback) % refit_freq == 0 or current_res is None:
            train_slice = returns_series.iloc[t - lookback : t]
            am = arch_model(train_slice, mean="Constant", vol="Garch", p=1, o=1, q=1, dist="skewt")
            try:
                current_res = am.fit(disp="off", show_warning=False)
                params = current_res.params
                last_params = {
                    "mu": float(params.get("mu", 0.0)),
                    "omega": float(params["omega"]),
                    "alpha": float(params["alpha[1]"]),
                    "gamma": float(params["gamma[1]"]),
                    "beta": float(params["beta[1]"]),
                    "nu": float(params.get("nu", 5.0)),
                    "lambda": float(params.get("lambda", 0.0)),
                }
                # Parametric 99% quantile boundary (Skew-T inverse CDF)
                last_q_dist = float(current_res.model.distribution.ppf(0.01, params[-2:]))
            except Exception:
                # Retain previous parameters if numerical MLE fails
                pass

        # 2. Daily variance recursion at t using shock from t-1
        prev_r = returns_series.iloc[t - 1]
        prev_vol = vol_arr[t - 1] if not np.isnan(vol_arr[t - 1]) else current_res.conditional_volatility.iloc[-1]

        eps_prev = prev_r - last_params["mu"]
        leverage_ind = 1.0 if eps_prev < 0.0 else 0.0

        sigma2_t = (
            last_params["omega"]
            + (last_params["alpha"] + last_params["gamma"] * leverage_ind) * (eps_prev ** 2)
            + last_params["beta"] * (prev_vol ** 2)
        )
        sigma_t = np.sqrt(max(sigma2_t, config.GARCH_MIN_VARIANCE))

        vol_arr[t] = sigma_t
        resid_arr[t] = (returns_series.iloc[t] - last_params["mu"]) / sigma_t
        var99_arr[t] = last_params["mu"] + sigma_t * last_q_dist

    return (
        pd.Series(vol_arr, index=returns_series.index),
        pd.Series(resid_arr, index=returns_series.index),
        pd.Series(var99_arr, index=returns_series.index),
    )


def generate_clean_production_data():
    purge_stale_artifacts()

    print("[ETL] Fetching historical index panel and cross-border volatility...")

    # 1. US + (attempted) India VIX
    us_vix_close, india_vix_close, used_real_india = fetch_vix_pair()
    if used_real_india:
        print(f"  [ETL] Using REAL India VIX (^INDIAVIX) with {len(india_vix_close)} observations.")
    else:
        print("  [ETL] India VIX history insufficient -> using Domestic_RV_Proxy (realized-vol proxy).")

    ticker_dfs = []

    for label, symbol in config.TICKERS.items():
        print(f"[ETL] Downloading and processing series: {label} ({symbol})...")
        raw_ticker = yf.download(symbol, start=config.START_DATE, end=config.END_DATE, progress=False)
        if raw_ticker.empty:
            raise ValueError(f"[FATAL] Failed downloading data for {symbol}.")

        raw_ticker = clean_yf_columns(raw_ticker)
        df = raw_ticker[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

        df["Log_Ret"] = 100 * np.log(df["Close"] / df["Close"].shift(1))
        df = df.dropna()

        df = compute_garman_klass(df)

        # Point-in-Time rolling GJR-GARCH
        vol, resid, var_99 = rolling_gjr_garch_pit(df["Log_Ret"])

        df["GARCH_Vol"] = vol
        df["GARCH_sigma"] = vol
        df["GARCH_Resid"] = resid
        df["GARCH_resid"] = resid
        df["GARCH_VaR_99"] = var_99

        # NOTE: Explicit Log_Ret_Lag1/Lag2 columns are DELIBERATELY NOT created.
        # The TFT encoder natively receives the observed target sequence
        # (Log_Ret up to time t) via encoder_target, so manual lag columns would
        # duplicate data the network already sees through its temporal encoder.

        # Macro features aligned to the ticker trading calendar
        macro_df = build_macro_features(us_vix_close, india_vix_close, df.index)

        df["US_VIX"] = macro_df["US_VIX"]
        df["US_VIX_Diff"] = macro_df["US_VIX_Diff"]
        df["US_VIX_Level"] = macro_df["US_VIX_Level"]

        if used_real_india:
            df["India_VIX"] = macro_df["India_VIX"]
            df["India_VIX_Diff"] = macro_df["India_VIX_Diff"]
            df["India_VIX_Level"] = macro_df["India_VIX_Level"]
            df["Domestic_RV_Proxy"] = compute_domestic_rv_proxy(df["Log_Ret"])
        else:
            # Honest fallback: realize-vol proxy, clearly labelled.
            df["India_VIX"] = np.nan
            df["India_VIX_Diff"] = compute_domestic_rv_proxy(df["Log_Ret"])
            df["India_VIX_Level"] = np.nan
            df["Domestic_RV_Proxy"] = df["India_VIX_Diff"]

        df = df.dropna()  # purges warm-up period

        df["ticker"] = label
        df["Date"] = df.index.strftime("%Y-%m-%d")
        ticker_dfs.append(df)

    # 3. Synchronize trading dates across all panel tickers
    common_dates = sorted(list(set(ticker_dfs[0]["Date"]).intersection(*[set(d["Date"]) for d in ticker_dfs[1:]])))
    date_to_time_idx = {d: i for i, d in enumerate(common_dates)}

    aligned_dfs = []
    for df in ticker_dfs:
        df_aligned = df[df["Date"].isin(common_dates)].copy()
        df_aligned["time_idx"] = df_aligned["Date"].map(date_to_time_idx)
        aligned_dfs.append(df_aligned)

    master_df = pd.concat(aligned_dfs, ignore_index=True)
    master_df = master_df.sort_values(by=["time_idx", "ticker"]).reset_index(drop=True)

    output_path = "master_df.csv"
    master_df.to_csv(output_path, index=False)

    # Persist provenance metadata about the domestic volatility source
    provenance = pd.DataFrame(
        [{
            "used_real_india_vix": used_real_india,
            "india_vix_obs": 0 if india_vix_close is None else len(india_vix_close),
        }]
    )
    provenance.to_csv("volatility_provenance.csv", index=False)

    print(f"\n[SUCCESS] Reconstructed clean multi-series panel at: {output_path}")
    print(f"Total Observations: {len(master_df)} rows across {len(config.TICKERS)} tickers.")
    print(f"Domestic Volatility Source: {'Real India VIX' if used_real_india else 'Domestic_RV_Proxy'}")


if __name__ == "__main__":
    generate_clean_production_data()
