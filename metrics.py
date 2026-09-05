# metrics.py
# =============================================================================
# Statistical backtesting and validation metrics for the VaR pipeline.
# Includes standard VaR backtests (Kupiec, Christoffersen, LR-CC, Engle-
# Manganelli DQ, Basel traffic light, Diebold-Mariano) plus a McNeil-Frey
# Expected Shortfall backtest, and helpers to aggregate metrics across
# multiple random seeds (Mean +/- Std) for honest statistical reporting.
# =============================================================================
import numpy as np
import pandas as pd
import scipy.stats as stats

import config


def pinball_loss(y_true, y_pred, q=0.01):
    """Vectorized Asymmetric Pinball (Quantile) Loss."""
    diff = y_true - y_pred
    return np.where(diff < 0, (1.0 - q) * (-diff), q * diff)


def quantile_loss(y_true, y_pred, q=0.01):
    """Alias kept for backward compatibility with HPO scripts."""
    return pinball_loss(y_true, y_pred, q)


def kupiec_pof_test(actual, var_pred, alpha=0.01):
    """Kupiec Unconditional Coverage (POF) Likelihood Ratio Test."""
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
    """Christoffersen Markov Interval Independence Test (LR_ind)."""
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
    """Engle-Manganelli Dynamic Quantile (DQ) Test."""
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
    """Diebold-Mariano test comparing pinball losses with Newey-West HAC SEs."""
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


def mcnell_frey_es_test(actual, var_pred, sigma, alpha=0.01, mu=0.0):
    """
    McNeil-Frey Expected Shortfall backtest.

    For every day where the realized return falls below the VaR forecast
    (an exceedance), standardize the exceedance by the forecast volatility:
        z_i = (r_i - mu) / sigma_i
    Under a correctly-specified model, the mean of these standardized
    exceedances should be consistent with the model's tail expectation. We
    report the empirical mean tail loss (ES) and a one-sample t-test on the
    standardized exceedances (H0: zero-mean), which detects whether the
    model systematically under- or over-states tail severity beyond the VaR.

    This gives the TFT a SECOND dimension (tail SHAPE) to demonstrate value
    even when VaR breach counts are statistically indistinguishable.

    Returns a dict with the ES estimate, the t-statistic and p-value, and the
    number of exceedances used.
    """
    hits = actual < var_pred
    n_exceed = int(np.sum(hits))

    if n_exceed < config.ES_MIN_BREACHES:
        return {
            "n_exceed": n_exceed,
            "es_empirical": np.nan,
            "es_t_stat": np.nan,
            "es_p_value": np.nan,
            "es_mean_resid": np.nan,
        }

    sigma_vals = np.asarray(sigma)[hits]
    actual_vals = np.asarray(actual)[hits]

    # Empirical ES (average loss beyond the VaR boundary)
    es_empirical = float(np.mean(actual_vals))

    # Standardized exceedances
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (actual_vals - mu) / sigma_vals
    z = z[np.isfinite(z)]

    if len(z) < 2:
        return {
            "n_exceed": n_exceed,
            "es_empirical": es_empirical,
            "es_t_stat": np.nan,
            "es_p_value": np.nan,
            "es_mean_resid": np.nan,
        }

    # One-sample t-test on standardized exceedances (H0: mean == 0)
    t_stat, p_val = stats.ttest_1samp(z, 0.0)

    return {
        "n_exceed": n_exceed,
        "es_empirical": es_empirical,
        "es_t_stat": float(t_stat),
        "es_p_value": float(p_val),
        "es_mean_resid": float(np.mean(z)),
    }


def multivariate_co_breach_test(actual_dict, var_dict, alpha=0.01):
    """Evaluates simultaneous tail exceedances across panel indices."""
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
        "poisson_p_value": float(p_value),
    }


def get_basel_traffic_light(failures, total_obs, alpha=0.01):
    """Basel III / FRTB Traffic Light status based on binomial CDF."""
    p_cum = stats.binom.cdf(failures, total_obs, alpha)
    green_limit = stats.binom.ppf(config.BASEL_GREEN_CUM, total_obs, alpha)
    if p_cum < config.BASEL_GREEN_CUM:
        zone = "GREEN"
    elif p_cum < config.BASEL_YELLOW_CUM:
        zone = "YELLOW"
    else:
        zone = "RED"
    return int(green_limit), zone


def calculate_metrics(actual_or_df, garch_var=None, tft_var=None, garch_sigma=None, alpha=0.01):
    """
    Universal dispatcher supporting both DataFrame and explicit array parameters.
    If a DataFrame is passed, columns are expected to include:
        Actual or Log_Ret, GARCH_VaR_99, TFT_VaR_99 (or TFT_Downside_99),
        and optionally GARCH_sigma (for the McNeil-Frey ES backtest).
    """
    if isinstance(actual_or_df, pd.DataFrame):
        df = actual_or_df.copy()
        act_col = "Actual" if "Actual" in df.columns else "Log_Ret"
        garch_col = "GARCH_VaR_99"
        tft_col = "TFT_VaR_99" if "TFT_VaR_99" in df.columns else "TFT_Downside_99"

        actual = df[act_col].values
        garch_var = df[garch_col].values
        tft_var = df[tft_col].values
        garch_sigma = df["GARCH_sigma"].values if "GARCH_sigma" in df.columns else None
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

    # McNeil-Frey Expected Shortfall backtest (tail-shape dimension)
    if garch_sigma is not None:
        es = mcnell_frey_es_test(actual, tft_var, garch_sigma, alpha=alpha)
    else:
        es = {"n_exceed": np.nan, "es_empirical": np.nan, "es_t_stat": np.nan,
              "es_p_value": np.nan, "es_mean_resid": np.nan}

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
        "mean_loss_diff": dm["mean_diff"],
        "es_n_exceed": es["n_exceed"],
        "es_empirical": es["es_empirical"],
        "es_t_stat": es["es_t_stat"],
        "es_p_value": es["es_p_value"],
        "es_mean_resid": es["es_mean_resid"],
    }


def evaluate_panel_metrics(panel_df, alpha=0.01):
    """Evaluates metrics across every index in the panel, plus joint co-breaches."""
    tickers = panel_df["ticker"].unique()
    per_ticker = {}
    actual_dict = {}
    var_dict = {}

    for t in tickers:
        sub = panel_df[panel_df["ticker"] == t].sort_values(by="Date")
        m = calculate_metrics(sub, alpha=alpha)
        per_ticker[t] = m
        actual_dict[t] = sub["Log_Ret"].values if "Log_Ret" in sub.columns else sub["Actual"].values
        var_dict[t] = sub["TFT_VaR_99"].values

    min_len = min(len(v) for v in actual_dict.values())
    for t in tickers:
        actual_dict[t] = actual_dict[t][-min_len:]
        var_dict[t] = var_dict[t][-min_len:]

    co_breach = multivariate_co_breach_test(actual_dict, var_dict, alpha=alpha)
    return {"per_ticker": per_ticker, "co_breach": co_breach}


def aggregate_seed_metrics(metrics_list):
    """
    Aggregates a list of per-seed metric dicts (from calculate_metrics) into
    Mean +/- Std summary rows, with explicit count of seeds. This enforces
    honest statistical disclosure across random seeds instead of reporting a
    single favorable seed.

    Returns a list of dicts, one per distinct metric key, each with:
        metric, mean, std, values (list), n_seeds
    """
    if not metrics_list:
        return []

    # Collect all unique metric keys across seeds
    keys = sorted({k for m in metrics_list for k in m.keys()})
    agg_rows = []

    for k in keys:
        vals = []
        for m in metrics_list:
            v = m.get(k)
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                vals.append(float(v))

        if not vals:
            continue

        vals_arr = np.array(vals)
        agg_rows.append({
            "metric": k,
            "mean": float(np.nanmean(vals_arr)),
            "std": float(np.nanstd(vals_arr, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [round(v, 6) for v in vals],
            "n_seeds": len(vals),
        })

    return agg_rows


def format_mean_std(agg_rows, metric_key, decimals=4):
    """Returns 'mean +/- std' string for a metric, or 'N/A'."""
    for row in agg_rows:
        if row["metric"] == metric_key:
            if np.isnan(row["mean"]):
                return "N/A"
            return f"{row['mean']:.{decimals}f} +/- {row['std']:.{decimals}f}"
    return "N/A"
