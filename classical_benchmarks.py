# classical_benchmarks.py
# =============================================================================
# Computes Classical Benchmarks (Historical Simulation, Asymmetric CAViaR),
# runs a Model Confidence Set (MCS) heuristic, and performs Regime Analysis
# on the already-frozen 500-day out-of-sample predictions.
#
# Requires: master_df.csv and ablation_ens_panel_*.csv to be present.
# =============================================================================
import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

import config
from metrics import calculate_metrics, pinball_loss

OUTPUT_DIR = getattr(config, "OUTPUT_DIR", ".") + "/"
Q = 0.01

def asymmetric_caviar(beta, returns, q=0.01):
    """Asymmetric CAViaR recursion: VaR_t = b0 + b1*VaR_{t-1} + b2*(r_{t-1})^+ - b3*(r_{t-1})^-"""
    var = np.zeros_like(returns)
    var[0] = np.quantile(returns[:300], q) # Initialize with empirical quantile
    for t in range(1, len(returns)):
        r_prev = returns[t-1]
        var[t] = beta[0] + beta[1]*var[t-1] + beta[2]*max(r_prev, 0) - beta[3]*min(r_prev, 0)
    return var

def caviar_objective(beta, returns, q=0.01):
    var = asymmetric_caviar(beta, returns, q)
    return np.sum(pinball_loss(returns, var, q))

def fit_caviar(train_returns, test_returns):
    """Fits CAViaR on train, projects onto test."""
    # Bounds: b0 <= 0 (negative VaR), 0 <= b1 < 1 (persistence), b2 >= 0, b3 >= 0
    bounds = [(None, 0), (0, 0.999), (0, None), (0, None)]
    init_guess = [-0.1, 0.90, 0.1, 0.1]
    res = opt.minimize(caviar_objective, init_guess, args=(train_returns, Q), bounds=bounds, method='SLSQP')

    # Project OOS
    full_returns = np.concatenate([train_returns, test_returns])
    full_var = asymmetric_caviar(res.x, full_returns, Q)
    return full_var[-len(test_returns):], res.x

def compute_formal_hln_mcs(loss_df, alpha=0.10, B=1000, block_size=10):
    """
    Formal Hansen-Lunde-Nason (2011) Model Confidence Set.

    Uses a Stationary Block Bootstrap to control the Family-Wise Error Rate
    (FWER) while preserving the time-series autocorrelation in the VaR losses.
    The T_max statistic is used: at each elimination round the model with the
    largest (worst) t-statistic is removed if the bootstrap p-value < alpha;
    the procedure stops when no further model can be rejected, yielding the
    MCS "superior set" of statistically indistinguishable models.

    Returns the list of surviving model names.
    """
    active_models = list(loss_df.columns)
    losses = loss_df.values
    T = len(losses)

    # Degenerate guard: with too few observations for a block bootstrap, the
    # MCS is not identifiable -- return all models (no elimination).
    if T <= block_size:
        return active_models

    # 1. Generate Stationary Block Bootstrap indices.
    #    This preserves the time-series autocorrelation in the VaR losses.
    boot_indices = np.random.randint(0, T - block_size + 1, size=(B, T // block_size + 1))
    boot_indices = (boot_indices[:, :, None] + np.arange(block_size)).reshape(B, -1)[:, :T]

    while len(active_models) > 1:
        active_losses = loss_df[active_models].values

        # 2. Compute differentials from the mean of active models.
        mean_active = active_losses.mean(axis=1, keepdims=True)
        d_i = active_losses - mean_active
        d_i_mean = d_i.mean(axis=0)

        # 3. Bootstrap the differentials.
        boot_d_i = np.zeros((B, len(active_models)))
        for b in range(B):
            boot_d_i[b] = d_i[boot_indices[b]].mean(axis=0)

        var_d_i = boot_d_i.var(axis=0, ddof=1)
        var_d_i = np.maximum(var_d_i, 1e-10)  # Guard against zero variance

        # 4. T_max statistic.
        t_stats = d_i_mean / np.sqrt(var_d_i / T)
        t_max = t_stats.max()
        worst_idx = int(t_stats.argmax())

        # 5. Bootstrap p-value under the null (centered).
        boot_centered = boot_d_i - d_i_mean
        boot_t_stats = boot_centered / np.sqrt(var_d_i / T)
        boot_t_max = boot_t_stats.max(axis=1)

        p_val = np.mean(boot_t_max >= t_max)

        # 6. Elimination rule.
        if p_val < alpha:
            active_models.pop(worst_idx)
        else:
            break

    return active_models

def get_tft_col(df):
    """
    Robustly locates the TFT VaR column in an ablation ensemble panel.

    The ablation runner writes the ensemble q0.01 as 'TFT_q01' (PyTorch
    Forecasting's quantile naming); older panels used 'TFT_VaR_99_Ensemble' /
    'TFT_VaR_99' / 'TFT_VaR_99_Raw' / 'TFT_Downside_99'. This helper tries the
    known names in order and raises a clear KeyError listing the actual columns
    if none match -- so a schema drift fails loudly instead of a silent NaN.
    """
    candidates = ['TFT_q01', 'TFT_VaR_99_Ensemble', 'TFT_VaR_99', 'TFT_VaR_99_Raw', 'TFT_Downside_99']
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find TFT VaR column in {list(df.columns)}")


def main():
    print("=== RUNNING CLASSICAL BENCHMARKS & MCS ===")
    df = pd.read_csv("master_df.csv")

    # Load all ablation panels to gather OOS predictions
    models_data = {}
    arms = ["1_Raw_TFT", "2_GARCH_TFT", "5_Full_ECTFT"]
    for arm in arms:
        path = f"ablation_ens_panel_{arm}.csv"
        if os.path.exists(path):
            models_data[arm] = pd.read_csv(path)
        else:
            print(f"[WARN] {path} missing. Run ablation_runner.py first.")
            return

    results = []
    regime_results = []

    for ticker in config.TICKERS.keys():
        print(f"\nProcessing {ticker}...")
        sub_df = df[df['ticker'] == ticker].copy()

        # Define Train/Test splits (matches config.BACKTEST_DAYS)
        test_cutoff_idx = len(sub_df) - config.BACKTEST_DAYS
        train_sub = sub_df.iloc[:test_cutoff_idx]
        test_sub = sub_df.iloc[test_cutoff_idx:]

        train_ret = train_sub['Log_Ret'].values
        test_ret = test_sub['Log_Ret'].values

        # 1. Historical Simulation (250-day rolling)
        sub_df['HS_VaR_99'] = sub_df['Log_Ret'].shift(1).rolling(250).quantile(Q)
        hs_oos = sub_df['HS_VaR_99'].iloc[test_cutoff_idx:].values

        # 2. Asymmetric CAViaR
        caviar_oos, caviar_beta = fit_caviar(train_ret, test_ret)
        print(f"  CAViaR Beta: {np.round(caviar_beta, 4)}")

        # Extract existing forecasts (robust column lookup -- the ablation
        # panels name the ensemble q0.01 'TFT_q01', not 'TFT_VaR_99_Ensemble').
        garch_oos = test_sub['GARCH_VaR_99'].values

        df_raw = models_data["1_Raw_TFT"]
        df_garch = models_data["2_GARCH_TFT"]
        df_full = models_data["5_Full_ECTFT"]

        tft_raw = df_raw[df_raw['ticker'] == ticker][get_tft_col(df_raw)].values
        tft_garch = df_garch[df_garch['ticker'] == ticker][get_tft_col(df_garch)].values
        tft_full = df_full[df_full['ticker'] == ticker][get_tft_col(df_full)].values

        # Construct Loss DataFrame for MCS
        loss_df = pd.DataFrame({
            "HS_250": pinball_loss(test_ret, hs_oos, Q),
            "CAViaR": pinball_loss(test_ret, caviar_oos, Q),
            "GJR-GARCH": pinball_loss(test_ret, garch_oos, Q),
            "Raw_TFT": pinball_loss(test_ret, tft_raw, Q),
            "GARCH_TFT": pinball_loss(test_ret, tft_garch, Q),
            "Full_ECTFT": pinball_loss(test_ret, tft_full, Q)
        })

        # Run the formal Hansen-Lunde-Nason (2011) MCS with block bootstrap.
        superior_set = compute_formal_hln_mcs(loss_df)
        print(f"  Superior Set (alpha=0.10): {superior_set}")

        # Calculate standard metrics for new benchmarks
        for name, var_pred in [("HS_250", hs_oos), ("CAViaR", caviar_oos)]:
            m = calculate_metrics(test_ret, garch_var=garch_oos, tft_var=var_pred)
            results.append({
                "Asset": ticker, "Model": name, "Breaches": m["breaches"],
                "Kupiec p": m["kupiec_p_value"], "DQ p": m["dq_p_value"],
                "Mean Loss": loss_df[name].mean(),
                "In Superior Set": "Yes" if name in superior_set else "No"
            })

        # Append existing models for completeness in report
        for name in ["GJR-GARCH", "Raw_TFT", "GARCH_TFT", "Full_ECTFT"]:
            results.append({
                "Asset": ticker, "Model": name, "Breaches": "N/A (See Main Table)",
                "Kupiec p": "N/A", "DQ p": "N/A",
                "Mean Loss": loss_df[name].mean(),
                "In Superior Set": "Yes" if name in superior_set else "No"
            })

        # --- Regime Analysis (High vs Low Volatility) ---
        # High vol regime = GARCH sigma > 75th percentile of historical train window
        vol_75th = np.percentile(train_sub['GARCH_sigma'].dropna(), 75)
        high_vol_mask = test_sub['GARCH_sigma'].values > vol_75th

        loss_garch = loss_df["GJR-GARCH"].values
        loss_tft = loss_df["Full_ECTFT"].values

        regime_results.append({
            "Asset": ticker,
            "Regime": "High Volatility",
            "Obs": sum(high_vol_mask),
            "GARCH Mean Loss": loss_garch[high_vol_mask].mean() if sum(high_vol_mask)>0 else np.nan,
            "TFT Mean Loss": loss_tft[high_vol_mask].mean() if sum(high_vol_mask)>0 else np.nan
        })
        regime_results.append({
            "Asset": ticker,
            "Regime": "Normal Volatility",
            "Obs": sum(~high_vol_mask),
            "GARCH Mean Loss": loss_garch[~high_vol_mask].mean() if sum(~high_vol_mask)>0 else np.nan,
            "TFT Mean Loss": loss_tft[~high_vol_mask].mean() if sum(~high_vol_mask)>0 else np.nan
        })

    # Export Reports
    df_res = pd.DataFrame(results)
    df_regime = pd.DataFrame(regime_results)

    with open(os.path.join(OUTPUT_DIR, "classical_benchmarks_report.txt"), "w") as f:
        f.write("=== CLASSICAL BENCHMARKS & MODEL CONFIDENCE SET ===\n")
        f.write(df_res.to_string(index=False))
        f.write("\n\n=== REGIME ANALYSIS (High vs Normal Volatility) ===\n")
        f.write(df_regime.to_string(index=False))

    print(f"\n[SUCCESS] Saved to {OUTPUT_DIR}classical_benchmarks_report.txt")

if __name__ == "__main__":
    main()
