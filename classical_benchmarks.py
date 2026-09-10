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

def compute_mcs_heuristic(loss_df, alpha=0.10):
    """
    Simplified Model Confidence Set (MCS) heuristic based on pairwise DM tests.
    Iteratively eliminates the worst model until all remaining models are
    statistically indistinguishable (p > alpha) from the best remaining model.
    """
    active_models = list(loss_df.columns)

    while len(active_models) > 1:
        # Calculate mean losses
        mean_losses = loss_df[active_models].mean()
        best_model = mean_losses.idxmin()
        worst_model = mean_losses.idxmax()

        # DM Test: Worst vs Best
        d_t = loss_df[worst_model] - loss_df[best_model]
        d_bar = d_t.mean()
        var_d = d_t.var() / len(d_t) # Simplified variance (no HAC for speed)

        if var_d == 0: break

        t_stat = d_bar / np.sqrt(var_d)
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        if p_val < alpha:
            active_models.remove(worst_model) # Eliminate worst
        else:
            break # Superior set found

    return active_models

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

        # Extract existing forecasts
        garch_oos = test_sub['GARCH_VaR_99'].values
        tft_raw = models_data["1_Raw_TFT"][models_data["1_Raw_TFT"]['ticker'] == ticker]['TFT_VaR_99_Ensemble'].values
        tft_garch = models_data["2_GARCH_TFT"][models_data["2_GARCH_TFT"]['ticker'] == ticker]['TFT_VaR_99_Ensemble'].values
        tft_full = models_data["5_Full_ECTFT"][models_data["5_Full_ECTFT"]['ticker'] == ticker]['TFT_VaR_99_Ensemble'].values

        # Construct Loss DataFrame for MCS
        loss_df = pd.DataFrame({
            "HS_250": pinball_loss(test_ret, hs_oos, Q),
            "CAViaR": pinball_loss(test_ret, caviar_oos, Q),
            "GJR-GARCH": pinball_loss(test_ret, garch_oos, Q),
            "Raw_TFT": pinball_loss(test_ret, tft_raw, Q),
            "GARCH_TFT": pinball_loss(test_ret, tft_garch, Q),
            "Full_ECTFT": pinball_loss(test_ret, tft_full, Q)
        })

        # Run MCS Heuristic
        superior_set = compute_mcs_heuristic(loss_df)
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
