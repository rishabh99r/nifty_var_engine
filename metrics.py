import numpy as np
import scipy.stats as stats

def quantile_loss(actual, forecast, q=0.01):
    error = actual - forecast
    return np.where(error < 0, (1 - q) * np.abs(error), q * np.abs(error))

def calculate_metrics(results_df):
    total_days = len(results_df)
    tft_failures = (results_df['Actual'] < results_df['TFT_VaR_99']).sum()
    tft_failure_rate = tft_failures / total_days

    green_limit = 0
    while stats.binom.cdf(green_limit, total_days, 0.01) < 0.95: green_limit += 1
    green_limit -= 1

    if tft_failures == 0:
        p_value = 0.0
    else:
        lr_null = (1 - 0.01)**(total_days - tft_failures) * (0.01**tft_failures)
        lr_alt = (1 - tft_failure_rate)**(total_days - tft_failures) * (tft_failure_rate**tft_failures)
        lr_stat = -2 * np.log(lr_null / lr_alt)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    results_df['TFT_Tick_Loss'] = quantile_loss(results_df['Actual'], results_df['TFT_VaR_99'], q=0.01)
    results_df['GARCH_Tick_Loss'] = quantile_loss(results_df['Actual'], results_df['GARCH_VaR_99'], q=0.01)

    d_t = (results_df['GARCH_Tick_Loss'] - results_df['TFT_Tick_Loss']).values
    mean_diff = np.mean(d_t)

    max_lag = int(np.floor(4 * (total_days / 100)**(2/9)))
    gamma_0 = np.sum((d_t - mean_diff)**2) / total_days
    hac_variance = gamma_0

    for j in range(1, max_lag + 1):
        gamma_j = np.sum((d_t[:-j] - mean_diff) * (d_t[j:] - mean_diff)) / total_days
        kernel_weight = 1 - (j / (max_lag + 1))
        hac_variance += 2 * kernel_weight * gamma_j

    if hac_variance <= 0:
        dm_stat, dm_p_value = 0.0, 1.0
    else:
        dm_stat = mean_diff / np.sqrt(hac_variance / total_days)
        dm_p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "tft_failures": tft_failures,
        "basel_limit": green_limit,
        "kupiec_p_value": p_value,
        "dm_statistic": dm_stat,
        "dm_p_value": dm_p_value
    }
