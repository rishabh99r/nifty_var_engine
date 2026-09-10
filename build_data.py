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
from metrics import extract_garch_dist_params

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
        "test_tft_predictions_panel.csv",
        "ablation_tournament_results.csv",
        "tft_nifty_optimization.db",
        "model_validation_master_report.txt",
        "experiment_manifest.json",  # stale until fresh training rewrites it
    ]
    for file in stale_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  -> Deleted stale file: {file}")
            except Exception as e:
                print(f"  [WARNING] Could not purge {file}: {str(e)}")

    # Wildcard artifact groups (Phase-3: fully sanitize the workspace between
    # fresh runs so stale seed panels / ablation panels never leak forward).
    stale_globs = [
        "test_tft_predictions_seed_*.csv",
        "test_tft_predictions_panel_seed_*.csv",
        "ablation_ens_panel_*.csv",
    ]
    for pattern in stale_globs:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                print(f"  -> Deleted stale file: {f}")
            except Exception as e:
                print(f"  [WARNING] Could not purge {f}: {str(e)}")

    for ckpt in (glob.glob("lightning_logs/*/checkpoints/*.ckpt")
                 + glob.glob("checkpoints/*.ckpt")):
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


def fetch_vix_pair(start_date=None, end_date=None):
    """
    Fetches US VIX (^VIX) and, if available, REAL India VIX (^INDIAVIX).

    Returns (us_vix_close, india_vix_close, used_real_india: bool).
    If the real India VIX has too few observations, returns None for it and the
    caller falls back to a Domestic_RV_Proxy computed from index returns.
    """
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = config.END_DATE

    raw_us = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if raw_us.empty:
        raise ValueError("[FATAL] Yahoo Finance API failure: Unable to download ^VIX.")

    raw_us = clean_yf_columns(raw_us)
    us_vix_close = raw_us["Close"].dropna()

    india_vix_close = None
    try:
        raw_in = yf.download(config.INDIA_VIX_SYMBOL, start=start_date, end=end_date, progress=False)
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

    # Native-calendar daily log-changes (non-overlapping, no ffill zeros).
    us_log_change = np.log(us_vix_close).diff().dropna()

    # ---- ML channel (ffill is SAFE: carries the last realized change, never
    # invents a zero change). These shifted features feed the TFT and must be
    # gap-free on the ticker calendar for the model to train/predict.
    us_change_ml = us_log_change.reindex(ticker_index).ffill()
    us_level_reindexed = us_vix_close.reindex(ticker_index).ffill()

    macro_df = pd.DataFrame(index=ticker_index)
    macro_df["US_VIX"] = us_level_reindexed.shift(us_sh)
    macro_df["US_VIX_Diff"] = us_change_ml.shift(us_sh)
    macro_df["US_VIX_Level"] = us_level_reindexed.shift(us_sh)

    # ---- Econometric channel (STRICT, S4): the *_NativeDiff columns are used
    # ONLY by the Granger-causality tests. They are reindexed onto the ticker
    # calendar WITHOUT ffill, so a ticker-trading day on which the US market
    # was CLOSED carries NaN. The shared helper (metrics.granger_series_from_
    # panel) drops NaN after joining, which yields a genuine INNER JOIN on
    # co-trading dates -- no carried-forward values masquerading as real
    # observations, no artificial zeros/autocorrelation inflating significance.
    # These columns are attached to master_df AFTER the ML dropna() (see
    # generate_clean_production_data) so their NaNs never shrink the model panel.
    macro_df["US_VIX_NativeDiff"] = us_log_change.reindex(ticker_index)

    if india_vix_close is not None:
        in_log_change = np.log(india_vix_close).diff().dropna()
        in_change_ml = in_log_change.reindex(ticker_index).ffill()
        in_level_reindexed = india_vix_close.reindex(ticker_index).ffill()

        macro_df["India_VIX"] = in_level_reindexed.shift(in_sh)
        macro_df["India_VIX_Diff"] = in_change_ml.shift(in_sh)
        macro_df["India_VIX_Level"] = in_level_reindexed.shift(in_sh)
        # STRICT native calendar (no ffill): NaN where India VIX did not trade.
        macro_df["India_VIX_NativeDiff"] = in_log_change.reindex(ticker_index)
    else:
        # NOTE: when the real India VIX is unavailable, India_VIX_NativeDiff is
        # left NaN here and the caller (generate_clean_production_data) fills it
        # with the non-overlapping domestic proxy under the has_real_india_vix=0
        # provenance flag.
        macro_df["India_VIX_NativeDiff"] = np.nan

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

    # Convergence accounting (Round 23 + Phase-3 fix): never silently swallow
    # MLE failures AND never assume "no exception == converged". The arch
    # optimizer exposes a convergence_flag; only flag==0 means the optimizer
    # actually converged. Tri-state accounting:
    #   converged       -> flag == 0, parameters updated
    #   non_converged   -> fit returned but flag != 0, PREVIOUS params retained
    #   exceptions      -> fit raised, previous params retained
    refit_attempts = 0
    refit_converged = 0
    refit_non_converged = 0
    refit_exceptions = 0

    print(f"  -> Running PIT rolling GJR-GARCH across {T} periods (warm-up: {lookback} days)...")

    for t in range(lookback, T):
        # 1. Periodic parameter re-estimation
        refit_just_happened = False  # state-continuity flag (F1)
        if (t - lookback) % refit_freq == 0 or current_res is None:
            refit_attempts += 1
            train_slice = returns_series.iloc[t - lookback : t]
            am = arch_model(train_slice, mean="Constant", vol="Garch", p=1, o=1, q=1, dist="skewt")
            try:
                candidate_res = am.fit(disp="off", show_warning=False)
                conv_flag = int(getattr(candidate_res, "convergence_flag", 0))
                if conv_flag != 0:
                    # Optimizer did NOT converge. Reject the refit: retain the
                    # previous parameters (strict point-in-time honesty).
                    refit_non_converged += 1
                    continue
                current_res = candidate_res
                refit_converged += 1
                refit_just_happened = True
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
                # FIX 10.1: Robust Skew-T shape extraction for the VaR floor.
                # Pass [nu, lambda] explicitly instead of fragile positional
                # params[-2:] (which can silently corrupt the quantile if arch
                # reorders its parameters).
                shape = extract_garch_dist_params(current_res)
                nu = shape["nu"] if not np.isnan(shape["nu"]) else 5.0
                lam = shape["lambda"] if not np.isnan(shape["lambda"]) else 0.0
                last_q_dist = float(current_res.model.distribution.ppf(0.01, [nu, lam]))
            except Exception:
                # Numerical MLE exception. Retain previous parameters and count.
                refit_exceptions += 1

        # FIX 12.2: If the VERY FIRST fit never succeeded (current_res is None),
        # there are no parameters to recurse with -- skip this day rather than
        # crashing on current_res.conditional_volatility. A failed MID-STREAM
        # refit is fine: the previous day's parameters remain valid and are
        # carried forward (that is the documented intent of except: pass).
        if current_res is None:
            continue

        # 2. Daily variance recursion at t using shock from t-1
        prev_r = returns_series.iloc[t - 1]
        # F1 (state-continuity fix): if a converged refit JUST happened, the
        # newly estimated parameter set was fit on the window [t-lookback, t-1],
        # so its FINAL fitted conditional volatility (the t-1 state under the NEW
        # parameters) is the correct recursion seed. Using vol_arr[t-1] -- which
        # was computed under the OLD parameters -- would inject a discontinuous
        # variance state into the first new-parameter recursion.
        if refit_just_happened:
            prev_vol = float(current_res.conditional_volatility.iloc[-1])
        else:
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

    # Convergence summary (Phase 3): tri-state, never silent.
    print(f"  -> PIT GARCH refits: {refit_attempts} attempted, "
          f"{refit_converged} converged, {refit_non_converged} non-converged "
          f"(flag!=0, params retained), {refit_exceptions} exceptions "
          f"(params retained).")

    return (
        pd.Series(vol_arr, index=returns_series.index),
        pd.Series(resid_arr, index=returns_series.index),
        pd.Series(var99_arr, index=returns_series.index),
    )


def generate_clean_production_data(start_date=None, end_date=None, purge=True):
    """
    Rebuilds the full master panel.

    - start_date/end_date default to config.START_DATE / config.END_DATE.
      config.END_DATE is the FROZEN research cutoff; pass an explicit end_date
      for a reproducible research window.
    - purge=True (default, research path): wipes stale model checkpoints and
      prediction artifacts first via purge_stale_artifacts().
    - purge=False: rebuilds the buffer WITHOUT touching any model artifacts.
      This is the SAFE path used by deployment (S2) so a routine market-data
      refresh can NEVER destroy trained checkpoints.
    """
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = config.END_DATE

    if purge:
        purge_stale_artifacts()
    else:
        print("[ETL] purge=False: leaving all checkpoints / prediction artifacts intact.")

    print("[ETL] Fetching historical index panel and cross-border volatility...")
    print(f"[ETL] Window: {start_date} -> {end_date}")

    # 1. US + (attempted) India VIX
    us_vix_close, india_vix_close, used_real_india = fetch_vix_pair(start_date, end_date)
    if used_real_india:
        print(f"  [ETL] Using REAL India VIX (^INDIAVIX) with {len(india_vix_close)} observations.")
    else:
        print("  [ETL] India VIX history insufficient -> using Domestic_RV_Proxy (realized-vol proxy).")

    ticker_dfs = []

    for label, symbol in config.TICKERS.items():
        print(f"[ETL] Downloading and processing series: {label} ({symbol})...")
        raw_ticker = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if raw_ticker.empty:
            raise ValueError(f"[FATAL] Failed downloading data for {symbol}.")

        raw_ticker = clean_yf_columns(raw_ticker)
        df = raw_ticker[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

        df["Log_Ret"] = 100 * np.log(df["Close"] / df["Close"].shift(1))

        # FIX 8.3: Copy Log_Ret -> Log_Ret_Feature IMMEDIATELY after creation,
        # BEFORE any dropna(), so the two series can never desynchronize.
        # Log_Ret_Feature lets the autoregressive sequence pass through the TFT's
        # Variable Selection Network (it is listed as an unknown real in
        # tft_model.py -- encoder-only, hidden from the decoder at t+1).
        df["Log_Ret_Feature"] = df["Log_Ret"]

        df = df.dropna(subset=["Log_Ret"]).copy()

        df = compute_garman_klass(df)

        # Point-in-Time rolling GJR-GARCH
        vol, resid, var_99 = rolling_gjr_garch_pit(df["Log_Ret"])

        df["GARCH_Vol"] = vol
        df["GARCH_sigma"] = vol
        df["GARCH_Resid"] = resid
        df["GARCH_resid"] = resid
        df["GARCH_VaR_99"] = var_99

        # Macro features aligned to the ticker trading calendar.
        macro_df = build_macro_features(us_vix_close, india_vix_close, df.index)

        df["US_VIX"] = macro_df["US_VIX"]
        df["US_VIX_Diff"] = macro_df["US_VIX_Diff"]
        df["US_VIX_Level"] = macro_df["US_VIX_Level"]
        # S4: STRICT *_NativeDiff columns (NaN on non-co-trading days) are
        # attached AFTER df.dropna() below so their NaNs NEVER shrink the ML
        # training panel. Econometric consumers dropna() themselves to obtain a
        # genuine inner join on shared trading dates.
        # (US_VIX_NativeDiff / India_VIX_NativeDiff assigned post-drop.)

        # FIX 14.3: Domestic realized-vol proxy for the Granger path. The
        # rolling(5).std() transform is OVERLAPPING and inflates serial
        # correlation in Granger tests, so the econometric fallback series is
        # now a NON-overlapping daily realized-vol proxy (|daily log-return|).
        # The overlapping rolling proxy is retained only as a legacy column.
        df["Domestic_RV_AbsRet"] = df["Log_Ret"].abs()
        df["Domestic_RV_NativeNonOverlap"] = df["Domestic_RV_AbsRet"]
        df["Domestic_RV_NativeProxy"] = df["Log_Ret"].rolling(5).std().diff()
        df["Domestic_RV_Proxy"] = df["Domestic_RV_NativeProxy"].shift(1)

        # Persist whether the REAL India VIX was used (1) or the proxy fallback
        # (0). master_df does NOT otherwise retain this, so downstream Granger
        # labeling cannot infer it from NaN counts (the fallback is NaN-free).
        df["has_real_india_vix"] = int(used_real_india)

        if used_real_india:
            df["India_VIX"] = macro_df["India_VIX"]
            df["India_VIX_Diff"] = macro_df["India_VIX_Diff"]
            df["India_VIX_Level"] = macro_df["India_VIX_Level"]
        else:
            # Honest fallback: realize-vol proxy, clearly labelled. Uses the
            # NON-overlapping series for the native (econometric) column so the
            # Granger test is not contaminated by rolling-window overlap.
            # NOTE (16.2): India_VIX and India_VIX_Level are intentionally
            # all-NaN SENTINEL columns in the fallback (schema stability). The
            # model never consumes them (not in candidate lists); consumers must
            # check has_real_india_vix before using them.
            df["India_VIX"] = np.nan
            df["India_VIX_Diff"] = df["Domestic_RV_Proxy"]
            df["India_VIX_Level"] = np.nan

        # ML dropna() -- purges the GARCH warm-up and any ML-channel NaN while
        # the strict econometric native columns are NOT yet attached (S4).
        df = df.dropna()

        # S4: attach the STRICT *_NativeDiff columns AFTER the ML dropna() so a
        # US/India market holiday (a ticker trading day with no VIX print) does
        # NOT remove that row from the training panel. Econometric consumers
        # (metrics.granger_series_from_panel) dropna() to inner-join only the
        # genuinely shared co-trading dates.
        df["US_VIX_NativeDiff"] = macro_df.loc[df.index, "US_VIX_NativeDiff"]
        if used_real_india:
            # Real India VIX: strict native calendar (NaN on non-India days).
            df["India_VIX_NativeDiff"] = macro_df.loc[df.index, "India_VIX_NativeDiff"]
        else:
            # Proxy fallback: the non-overlapping |daily return| proxy is defined
            # on the TICKER's own calendar, so it is fully observed (NaN-free).
            df["India_VIX_NativeDiff"] = df["Domestic_RV_NativeNonOverlap"]

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


def refresh_production_data_only(start_date=None, end_date=None):
    """
    S2 (security/engineering): SAFELY rebuilds the master buffer for live
    inference WITHOUT running purge_stale_artifacts().

    A routine market-data refresh must NEVER destroy trained checkpoints or
    validated prediction artifacts. The deployment orchestrator calls THIS
    function (not generate_clean_production_data with its purge), so a stale
    buffer can be refreshed without silently invalidating the ensemble.
    """
    print("[ETL] refresh_production_data_only: safe market-data refresh (NO purge).")
    generate_clean_production_data(start_date=start_date, end_date=end_date, purge=False)


if __name__ == "__main__":
    generate_clean_production_data()
