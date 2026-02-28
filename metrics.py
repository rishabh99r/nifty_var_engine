# metrics.py
import numpy as np
import scipy.stats as stats

def quantile_loss(actual, forecast, q=0.01):
    """Calculates the asymmetric tick loss for VaR forecasts."""
    error = actual - forecast
    return np.where(error < 0, (1 - q) * np.abs(error), q * np.abs(error))

def calculate_metrics(results_df):
    """
    Takes a DataFrame with 'Actual', 'TFT_VaR_99', and 'GARCH_VaR_99'.
    """
    total_days = len(results_df)

    # --- 1. TFT Failures & Basel Limit ---
    tft_failures = (results_df['Actual'] < results_df['TFT_VaR_99']).sum()
    tft_failure_rate = tft_failures / total_days

    green_limit = 0
    while stats.binom.cdf(green_limit, total_days, 0.01) < 0.95:
        green_limit += 1
    green_limit -= 1

    passed_basel = tft_failures <= green_limit

    # --- 2. Kupiec POF Test (TFT) ---
    p_target = 0.01
    if tft_failures == 0:
        p_value = 0.0
    else:
        lr_null = (1 - p_target)**(total_days - tft_failures) * (p_target**tft_failures)
        lr_alt = (1 - tft_failure_rate)**(total_days - tft_failures) * (tft_failure_rate**tft_failures)
        lr_stat = -2 * np.log(lr_null / lr_alt)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    # --- 3. Diebold-Mariano Test (Tick Loss) ---
    results_df['TFT_Tick_Loss'] = quantile_loss(results_df['Actual'], results_df['TFT_VaR_99'], q=0.01)
    results_df['GARCH_Tick_Loss'] = quantile_loss(results_df['Actual'], results_df['GARCH_VaR_99'], q=0.01)

    tft_mean_loss = results_df['TFT_Tick_Loss'].mean()
    garch_mean_loss = results_df['GARCH_Tick_Loss'].mean()

    # DM Statistic Calculation (1-step ahead)
    loss_differential = results_df['GARCH_Tick_Loss'] - results_df['TFT_Tick_Loss']
    mean_diff = loss_differential.mean()
    variance_diff = loss_differential.var(ddof=1)

    if variance_diff == 0:
        dm_stat, dm_p_value = 0.0, 1.0
    else:
        dm_stat = mean_diff / np.sqrt(variance_diff / total_days)
        # Two-tailed test
        dm_p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "tft_failures": tft_failures,
        "tft_failure_rate": tft_failure_rate,
        "basel_limit": green_limit,
        "passed_basel": passed_basel,
        "kupiec_p_value": p_value,
        "tft_mean_loss": tft_mean_loss,
        "garch_mean_loss": garch_mean_loss,
        "dm_statistic": dm_stat,
        "dm_p_value": dm_p_value
    }
