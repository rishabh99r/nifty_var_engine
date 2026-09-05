# metrics.py
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def pinball_loss(y_true, y_pred, q=0.01):
    """Vectorized Asymmetric Pinball (Quantile) Loss."""
    diff = y_true - y_pred
    return np.where(diff < 0, (1.0 - q) * (-diff), q * diff)


def kupiec_pof_test(actual, var_pred, alpha=0.01):
    """
    Kupiec Unconditional Coverage (POF) Likelihood Ratio Test.
    Includes exact binomial tail probability fallback for N=0 edge cases.
    """
    hits = (actual < var_pred).astype(int)
    N = int(np.sum(hits))
    T = len(hits)

    if T == 0:
        return {"N": 0, "T": 0, "p_hat": 0.0, "stat": np.nan, "p_value": np.nan}

    p_hat = N / T

    if N == 0:
        p_val_exact = (1.0 - alpha) ** T
        lr_uc = -2.0 * np.log((1.0 - alpha) ** T)
        return {"N": 0, "T": T, "p_hat": 0.0, "stat": float(lr_uc), "p_value": float(p_val_exact)}

    num = ((1.0 - alpha) ** (T - N)) * (alpha ** N)
    den = ((1.0 - p_hat) ** (T - N)) * (p_hat ** N)

    if den <= 0 or num <= 0:
        lr_uc = 0.0
    else:
        lr_uc = -2.0 * np.log(num / den)

    p_value = 1.0 - stats.chi2.cdf(lr_uc, df=1)
    return {"N": N, "T": T, "p_hat": float(p_hat), "stat": float(lr_uc), "p_value": float(p_value)}


def christoffersen_independence_test(actual, var_pred):
    """
    Christoffersen Markov Interval Independence Test (LR_ind).
    Tests whether breaches cluster by modeling hits as a first-order Markov chain.
    """
    hits = (actual < var_pred).astype(int)
    T = len(hits)

    if T < 2 or np.sum(hits) < 2:
        return {"stat": 0.0, "p_value": 1.0, "n00": 0, "n01": 0, "n10": 0, "n11": 0}

    h_lag = hits[:-1]
    h_curr = hits[1:]

    n00 = int(np.sum((h_lag == 0) & (h_curr == 0)))
    n01 = int(np.sum((h_lag == 0) & (h_curr == 1)))
    n10 = int(np.sum((h_lag == 1) & (h_curr == 0)))
    n11 = int(np.sum((h_lag == 1) & (h_curr == 1)))

    pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    l_null = ((1.0 - pi) ** (n00 + n10)) * (pi ** (n01 + n11)) if 0 < pi < 1 else 1e-12
    l_alt = 1.0
    l_alt *= ((1.0 - pi_0) ** n00) * (pi_0 ** n01) if (n00 + n01) > 0 and 0 < pi_0 < 1 else 1.0
    l_alt *= ((1.0 - pi_1) ** n10) * (pi_1 ** n11) if (n10 + n11) > 0 and 0 < pi_1 < 1 else 1.0

    if l_null <= 0 or l_alt <= 0:
        lr_ind = 0.0
    else:
        lr_ind = -2.0 * np.log(l_null / l_alt)

    lr_ind = max(0.0, lr_ind)
    p_value = 1.0 - stats.chi2.cdf(lr_ind, df=1)

    return {"stat": float(lr_ind), "p_value": float(p_value), "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def engle_manganelli_dq_test(actual, var_pred, alpha=0.01, lags=4):
    """
    Engle-Manganelli Dynamic Quantile (DQ) Test.
    Regresses Hit_t = I_t - alpha on lagged hits and forecasted VaR.
    """
    hits = (actual < var_pred).astype(float)
    T = len(hits)
    if T <= lags + 2 or np.sum(hits) == 0:
        return {"stat": np.nan, "p_value": np.nan}

    hit_dem = hits - alpha
    X = np.ones((T - lags, lags + 2))
    for l in range(1, lags + 1):
        X[:, l] = hit_dem[(lags - l):(T - l)]
    X[:, -1] = var_pred[lags:]

    y = hit_dem[lags:]

    try:
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
        beta = np.linalg.solve(XtX, Xty)
        dq_stat = np.dot(beta.T, np.dot(XtX, beta)) / (alpha * (1.0 - alpha))
        df = lags + 2
        p_val = 1.0 - stats.chi2.cdf(dq_stat, df=df)
        return {"stat": float(dq_stat), "p_value": float(p_val), "df": df}
    except np.linalg.LinAlgError:
        return {"stat": np.nan, "p_value": np.nan, "df": lags + 2}


def diebold_mariano_test(y_true, y_pred1, y_pred2, q=0.01):
    """
    Diebold-Mariano test comparing pinball losses with Newey-West HAC standard errors.
    """
    d_t = pinball_loss(y_true, y_pred1, q) - pinball_loss(y_true, y_pred2, q)
    T = len(d_t)
    if T < 5:
        return {"dm_stat": 0.0, "dm_p_value": 1.0, "mean_diff": 0.0}

    d_bar = np.mean(d_t)
    max_lag = int(np.floor(4.0 * ((T / 100.0) ** (2.0 / 9.0))))
    gamma_0 = np.var(d_t, ddof=0)

    gamma_sum = 0.0
    for l in range(1, max_lag + 1):
        weight = 1.0 - (l / (max_lag + 1.0))
        cov_l = np.cov(d_t[:-l], d_t[l:], ddof=0)[0, 1]
        gamma_sum += 2.0 * weight * cov_l

    lr_var = max(gamma_0 + gamma_sum, 1e-10)
    dm_stat = d_bar / np.sqrt(lr_var / T)
    dm_p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "dm_p_value": float(dm_p_val), "mean_diff": float(d_bar)}


def multivariate_co_breach_test(actual_dict, var_dict, alpha=0.01):
    """
    Evaluates simultaneous tail exceedances across panel indices (NIFTY50, BANKNIFTY, NIFTYIT).
    """
    tickers = list(actual_dict.keys())
    K = len(tickers)
    T = len(actual_dict[tickers[0]])

    hits = np.zeros((T, K))
    for i, t in enumerate(tickers):
        hits[:, i] = (actual_dict[t] < var_dict[t]).astype(int)

    co_breaches = (np.sum(hits, axis=1) == K).astype(int)
    observed_co_breaches = int(np.sum(co_breaches))
    p_joint = alpha ** K
    expected_co_breaches = T * p_joint

    p_value = 1.0 - stats.poisson.cdf(observed_co_breaches - 1, expected_co_breaches) if observed_co_breaches > 0 else 1.0

    return {
        "panel_size": K,
        "observed_co_breaches": observed_co_breaches,
        "expected_co_breaches": float(expected_co_breaches),
        "poisson_p_value": float(p_value)
    }


def get_basel_traffic_light(failures, total_obs, alpha=0.01):
    """Calculates Basel III / FRTB Traffic Light status based on binomial CDF."""
    p_cum = stats.binom.cdf(failures, total_obs, alpha)
    # Regulatory bounds for 99% VaR: Green (cumulative prob < 95%), Yellow (< 99.99%), Red (>= 99.99%)
    green_limit = stats.binom.ppf(0.95, total_obs, alpha)
    if p_cum < 0.95:
        zone = "GREEN"
    elif p_cum < 0.9999:
        zone = "YELLOW"
    else:
        zone = "RED"
    return int(green_limit), zone


def calculate_metrics(actual_or_df, garch_var=None, tft_var=None, alpha=0.01):
    """
    Universal dispatcher supporting both DataFrame and explicit array parameters.
    """
    if isinstance(actual_or_df, pd.DataFrame):
        df = actual_or_df.copy()
        act_col = 'Actual' if 'Actual' in df.columns else 'Log_Ret'
        garch_col = 'GARCH_VaR_99'
        tft_col = 'TFT_VaR_99' if 'TFT_VaR_99' in df.columns else 'TFT_Downside_99'

        actual = df[act_col].values
        garch_var = df[garch_col].values
        tft_var = df[tft_col].values
    else:
        actual = np.asarray(actual_or_df)
        garch_var = np.asarray(garch_var)
        tft_var = np.asarray(tft_var)

    kupiec = kupiec_pof_test(actual, tft_var, alpha=alpha)
    christ = christoffersen_independence_test(actual, tft_var)
    dq = engle_manganelli_dq_test(actual, tft_var, alpha=alpha)
    dm = diebold_mariano_test(actual, garch_var, tft_var, q=alpha)

    lr_cc = kupiec["stat"] + christ["stat"]
    p_cc = 1.0 - stats.chi2.cdf(lr_cc, df=2)

    limit, zone = get_basel_traffic_light(kupiec["N"], kupiec["T"], alpha=alpha)

    return {
        "breaches": kupiec["N"],
        "tft_failures": kupiec["N"],
        "total_obs": kupiec["T"],
        "basel_limit": limit,
        "basel_zone": zone,
        "kupiec_stat": kupiec["stat"],
        "kupiec_p_value": kupiec["p_value"],
        "christ_stat": christ["stat"],
        "christ_p_value": christ["p_value"],
        "cc_stat": float(lr_cc),
        "cc_p_value": float(p_cc),
        "dq_stat": dq["stat"],
        "dq_p_value": dq["p_value"],
        "dm_stat": dm["dm_stat"],
        "dm_statistic": dm["dm_stat"],
        "dm_p_value": dm["dm_p_value"],
        "mean_loss_diff": dm["mean_diff"]
    }


def evaluate_panel_metrics(panel_df, alpha=0.01):
    """Evaluates metrics across every index in the panel, plus joint co-breaches."""
    tickers = panel_df['ticker'].unique()
    per_ticker = {}
    actual_dict = {}
    var_dict = {}

    for t in tickers:
        sub = panel_df[panel_df['ticker'] == t].sort_values(by='Date')
        m = calculate_metrics(sub, alpha=alpha)
        per_ticker[t] = m
        actual_dict[t] = sub['Log_Ret'].values if 'Log_Ret' in sub.columns else sub['Actual'].values
        var_dict[t] = sub['TFT_VaR_99'].values

    # Determine minimum length for multi-asset alignment
    min_len = min(len(v) for v in actual_dict.values())
    for t in tickers:
        actual_dict[t] = actual_dict[t][-min_len:]
        var_dict[t] = var_dict[t][-min_len:]

    co_breach = multivariate_co_breach_test(actual_dict, var_dict, alpha=alpha)
    return {"per_ticker": per_ticker, "co_breach": co_breach}
