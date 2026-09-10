# metrics.py
# =============================================================================
# Statistical backtesting and validation metrics for the VaR pipeline.
# Includes standard VaR backtests (Kupiec, Christoffersen, LR-CC, Engle-
# Manganelli DQ, Diebold-Mariano) plus a REGULATORY-INSPIRED binomial
# coverage zone (NOT the formal Basel traffic-light table / FRTB) and a
# descriptive "Tail Exceedance Depth" diagnostic (NOT a formal McNeil-Frey
# Expected Shortfall backtest), plus helpers to aggregate metrics across
# multiple random seeds for honest statistical reporting.
# =============================================================================
import numpy as np
import pandas as pd
import scipy.stats as stats

import config


def pinball_loss(y_true, y_pred, q=0.01):
    """Vectorized Asymmetric Pinball (Quantile) Loss."""
    diff = y_true - y_pred
    return np.where(diff < 0, (1.0 - q) * (-diff), q * diff)


# Aliases used by arch-family distributions for the tail degrees of freedom.
# 'eta' is included because some arch versions/configurations name the skew-t
# degrees-of-freedom parameter 'eta' rather than 'nu'.
_DF_ALIASES = ("nu", "df", "v", "shape", "tail", "eta")
_SKEW_ALIASES = ("lambda", "skew", "gamma")


def granger_series_from_panel(sub):
    """
    Builds clean, chronologically-true (us, domestic) series for Granger
    causality from one ticker's panel slice.

    IMPORTANT (native-vs-shifted FIREWALL): the ML forecasting pipeline uses
    timezone-shifted *_Diff columns (US_VIX_SHIFT=2, INDIA_VIX_SHIFT=1) to stay
    free of look-ahead bias. Those shifted columns MUST NOT be used for the
    Granger test -- applying a negative shift() to "undo" them would pull
    FUTURE values into the present and destroy the causal arrow. Instead,
    build_data.py emits separate, UN-shifted, STRICT *_NativeDiff columns
    (NaN on non-co-trading days -- see S4) that are used ONLY by this
    econometric test. They are read here with ZERO shifting.

    S4 (artificial-zero removal): BOTH returned series are inner-aligned on
    genuinely shared observations -- rows where EITHER series is NaN (e.g. a
    US market holiday with no VIX print, or a real-India-VIX day with no US
    print) are dropped. This is a true inner join on co-trading dates and
    eliminates forward-filled values / artificial zero-returns that would
    deflate variance and falsely inflate Granger significance. The two series
    are returned sharing an identical DatetimeIndex of length
    min(len(us), len(dom)) after the drop.

    Returns (us_series, dom_series, domestic_label).
    """
    us = sub["US_VIX_NativeDiff"].astype(float)

    # Branch on the persisted provenance flag, NOT a NaN count: the proxy
    # fallback in build_data fills India_VIX_NativeDiff with the (NaN-free)
    # abs-return proxy, so a "notna().sum() > 50" test would ALWAYS pass and
    # mislabel the proxy series as "Real India VIX".
    if "has_real_india_vix" in sub.columns and bool(sub["has_real_india_vix"].iloc[0]):
        dom = sub["India_VIX_NativeDiff"].astype(float)
        domestic_label = "Real India VIX (native calendar)"
    else:
        # FIX 10.3 + 14.3: label the RV proxy with the asset's OWN ticker name
        # (each ticker's proxy is that index's own vol, not a shared India-vol
        # series), and use the NON-overlapping |daily return| proxy so the
        # Granger test is not contaminated by rolling-window overlap.
        if "Domestic_RV_NativeNonOverlap" in sub.columns:
            dom = sub["Domestic_RV_NativeNonOverlap"].astype(float)
            proxy_desc = "realized-vol proxy (|daily return|, non-overlapping)"
        else:
            dom = sub["Domestic_RV_NativeProxy"].astype(float)
            proxy_desc = "rolling-vol proxy (overlapping)"
        ticker_name = sub["ticker"].iloc[0] if "ticker" in sub.columns else "Asset"
        domestic_label = f"{ticker_name} own {proxy_desc}"

    # S4: inner-align on shared observations (drop rows where EITHER side is
    # NaN). us/dom may carry NaN on non-co-trading calendar days.
    if not us.index.equals(dom.index):
        us = us.reindex(us.index.union(dom.index))
        dom = dom.reindex(us.index)
    mask = us.notna() & dom.notna()
    us = us[mask]
    dom = dom[mask]

    return us, dom, domestic_label


def granger_diagnostics(series_dict):
    """
    Runs robustness diagnostics on the series used in Granger-causality tests.
    These are the checks a skeptical reviewer will demand before accepting
    p-values near 0.0000 as real spillover rather than an alignment artifact:

      1. ADF stationarity test on each series (Granger requires stationary
         inputs; differencing US VIX log-levels is intended to deliver this).
      2. Zero-fraction: share of exact-0 values. A high zero-fraction from
         calendar misalignment would deflate variance and inflate significance.
      3. Duplicate-fraction: share of values equal to their predecessor
         (repeated carries / unchanged days), the residue of ffill artifacts.

    `series_dict` maps a label -> pd.Series (already cleaned, NaNs dropped).
    Returns {label: {'adf_stat','adf_p','zero_frac','dup_frac','n'}}.
    """
    from statsmodels.tsa.stattools import adfuller  # lazy import (Colab has it)

    out = {}
    for label, s in series_dict.items():
        s = pd.Series(s).dropna().astype(float)
        n = len(s)
        zero_frac = float(np.mean(s == 0.0)) if n else np.nan
        dup_frac = float(np.mean(s.iloc[1:].values == s.iloc[:-1].values)) if n > 1 else np.nan

        adf_stat = adf_p = np.nan
        if n > 10:
            try:
                adf_res = adfuller(s, autolag="AIC")
                adf_stat = float(adf_res[0])
                adf_p = float(adf_res[1])
            except Exception:
                pass

        out[label] = {
            "n": n,
            "adf_stat": adf_stat,
            "adf_p": adf_p,
            "zero_frac": zero_frac,
            "dup_frac": dup_frac,
        }
    return out


def _fmt_pct(v, mult=100.0, decimals=2):
    """NaN-safe percentage formatter."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(v):
        return "N/A"
    return f"{mult * v:.{decimals}f}%"


def _fmt_p(v, decimals=4):
    """NaN-safe p-value formatter."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def format_granger_diagnostics(diag):
    """Human-readable one-liner for a diagnostics dict."""
    lines = []
    for label, d in diag.items():
        lines.append(
            f"{label}: n={d['n']}, ADF p={_fmt_p(d['adf_p'])}, "
            f"zero%={_fmt_pct(d['zero_frac'])}, dup%={_fmt_pct(d['dup_frac'])}"
        )
    return " | ".join(lines)


def extract_garch_dist_params(res):
    """
    Robustly extracts the shape parameters (tail df, skew) from a fitted arch
    model RESULT by reading the distribution's OWN parameter names.

    The arch package can name the skew-t degrees-of-freedom differently across
    versions / mean specifications. This helper first does a KEYED lookup on the
    fitted parameter names, then falls back to a SkewStudent-specific positional
    mapping (the arch SkewStudent always positions [..., nu, lambda] last).

    Returns {'nu': float-or-NaN, 'lambda': float-or-NaN}.
    """
    out = {"nu": np.nan, "lambda": np.nan}
    if res is None:
        return out

    params = getattr(res, "params", None)
    if params is None:
        return out

    try:
        names = list(params.index)
    except Exception:
        names = []

    # Strategy 1: keyed lookup on fitted parameter index
    for alias in _DF_ALIASES + _SKEW_ALIASES:
        if alias in names:
            try:
                val = float(params[alias])
            except (TypeError, ValueError):
                continue
            if alias in _DF_ALIASES and np.isnan(out["nu"]):
                out["nu"] = val
            elif alias in _SKEW_ALIASES and np.isnan(out["lambda"]):
                out["lambda"] = val

    # Strategy 2: SkewStudent-specific positional fallback. The arch SkewStudent
    # distribution positions its two shape parameters LAST in the fitted vector,
    # in the order [..., nu, lambda]. A generic name-order mapping is fragile
    # across arch versions, so we guard on the distribution name first.
    if (np.isnan(out["nu"]) or np.isnan(out["lambda"])) and hasattr(res.model, "distribution"):
        dist_name = str(getattr(res.model.distribution, "name", "")).lower()
        if "skew" in dist_name and "student" in dist_name:
            try:
                if np.isnan(out["nu"]):
                    out["nu"] = float(params.iloc[-2])
                if np.isnan(out["lambda"]):
                    out["lambda"] = float(params.iloc[-1])
            except Exception:
                pass

    return out


def kupiec_pof_test(actual, var_pred, alpha=0.01):
    """
    Kupiec Unconditional Coverage (POF) Likelihood Ratio Test.

    STANDARD LR FORM (Phase-3 fix, Reviewer): the boundary case N=0 (zero
    exceptions) uses the exact likelihood-ratio limit
        LR_UC = -2*T*ln(1-alpha)
    which for T=500, alpha=0.01 gives LR≈10.05, p≈0.0015 -- i.e. standard
    Kupiec correctly REJECTS a hyper-conservative zero-breach model at 5%
    (it is too conservative to be calibrated at the nominal 1% coverage rate).
    This is the statistically correct meaning, distinct from practical
    risk-management judgement. Symmetrically, N=T (all breaches) yields
    LR_UC = -2*T*ln(alpha), also handled by the general formula with a
    numerical guard.
    """
    hits = (actual < var_pred).astype(int)
    N = int(np.sum(hits))
    T = len(hits)

    if T == 0:
        return {"N": 0, "T": 0, "p_hat": 0.0, "stat": np.nan, "p_value": np.nan}

    p_hat = N / T

    if N == 0:
        # Boundary: zero failures. LR -> -2*T*ln(1-alpha) (exact limit).
        lr_uc = -2.0 * T * np.log(1.0 - alpha)
        lr_uc = max(lr_uc, 0.0)
        p_value = 1.0 - stats.chi2.cdf(lr_uc, df=1)
        return {"N": 0, "T": T, "p_hat": 0.0, "stat": float(lr_uc), "p_value": float(p_value)}
    if N == T:
        # Symmetric boundary: all breaches. LR -> -2*T*ln(alpha).
        lr_uc = -2.0 * T * np.log(alpha)
        lr_uc = max(lr_uc, 0.0)
        p_value = 1.0 - stats.chi2.cdf(lr_uc, df=1)
        return {"N": T, "T": T, "p_hat": 1.0, "stat": float(lr_uc), "p_value": float(p_value)}

    num = ((1.0 - alpha) ** (T - N)) * (alpha ** N)
    den = ((1.0 - p_hat) ** (T - N)) * (p_hat ** N)

    if den <= 0 or num <= 0:
        lr_uc = 0.0
    else:
        lr_uc = -2.0 * np.log(num / den)

    p_value = 1.0 - stats.chi2.cdf(lr_uc, df=1)
    return {"N": N, "T": T, "p_hat": float(p_hat), "stat": float(lr_uc), "p_value": float(p_value)}


def holm_bonferroni(p_values):
    """
    Holm-Bonferroni step-down multiple-testing correction (family-wise error
    rate control). Operates on a family of raw p-values IN ANY ORDER and
    returns adjusted p-values aligned to the input order.

    Method: sort raw p ascending; adjusted_i = max over j<=i of
    (n - j + 1) * p_(j); enforce monotone non-decreasing; cap at 1.0.
    """
    p_raw = np.asarray(p_values, dtype=float).ravel()
    n = len(p_raw)
    if n == 0:
        return np.array([])

    # NaN-safe: NaN p-values are passed through unchanged (never "significant").
    finite_mask = ~np.isnan(p_raw)
    adjusted = np.full(n, np.nan)
    if finite_mask.sum() == 0:
        return adjusted

    idx = np.where(finite_mask)[0]
    vals = p_raw[idx]
    order = np.argsort(vals, kind="mergesort")
    sorted_p = vals[order]
    m = len(sorted_p)

    # Step-down Holm: min(1, (m-j+1) * p_(j)), then enforce monotonicity.
    step = np.minimum(1.0, (m - np.arange(1, m + 1) + 1) * sorted_p)
    step = np.maximum.accumulate(step)
    step = np.minimum(step, 1.0)

    adjusted_idx = np.empty(m)
    adjusted_idx[order] = step
    adjusted[idx] = adjusted_idx
    return adjusted


def audit_quantile_monotonicity(q01, q50, q99, raise_on_violation=False):
    """
    Audits that quantile forecasts obey the monotonicity boundary
        q0.01 <= q0.50 <= q0.99
    for every row (quantile regression does NOT enforce this by construction).

    Accepts arrays/Series/DataFrames (aligned by position). Returns
        {"total": n, "violations_lo": n(q01>q50), "violations_hi": n(q50>q99)}
    and optionally raises if any violation is found.
    """
    a = np.asarray(q01, dtype=float)
    b = np.asarray(q50, dtype=float)
    c = np.asarray(q99, dtype=float)

    if not (a.ndim == b.ndim == c.ndim == 1 and len(a) == len(b) == len(c)):
        raise ValueError("q01/q50/q99 must be 1-D arrays of equal length.")

    total = len(a)
    viol_lo = int(np.sum(a > b))
    viol_hi = int(np.sum(b > c))
    if raise_on_violation and (viol_lo + viol_hi) > 0:
        raise RuntimeError(
            f"Quantile crossing detected: {viol_lo + viol_hi}/{total} rows "
            f"(q01>q50: {viol_lo}, q50>q99: {viol_hi})."
        )
    return {"total": total, "violations_lo": viol_lo, "violations_hi": viol_hi}


def christoffersen_independence_test(actual, var_pred):
    """Christoffersen Markov Interval Independence Test (LR_ind)."""
    hits = (actual < var_pred).astype(int)
    T = len(hits)

    if T < 2 or np.sum(hits) < 2:
        return {"stat": np.nan, "p_value": np.nan, "n00": 0, "n01": 0, "n10": 0, "n11": 0}

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
    """
    Diebold-Mariano test comparing pinball losses with Newey-West HAC SEs.

    CONVENTION (FIX, Round 23): y_pred1 = Baseline (GJR-GARCH), y_pred2 =
    Proposed (ECTFT/TFT). The loss differential is defined as
        d_t = Loss(Proposed) - Loss(Baseline)
    so a NEGATIVE mean_diff / dm_stat means the PROPOSED model (TFT) has
    LOWER pinball loss. This makes the sign convention unambiguous and matches
    the labels used in the reports and figures.
    """
    d_t = pinball_loss(y_true, y_pred2, q) - pinball_loss(y_true, y_pred1, q)
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


def tail_breach_depth_diagnostic(actual, var_pred, sigma, mu=0.0):
    """
    Descriptive diagnostic of the depth of realized VaR breaches.

    For every day where the realized return falls below the VaR forecast (an
    exceedance), standardize the exceedance by the forecast volatility:
        z_i = (r_i - mu) / sigma_i

    This is a descriptive "Tail Exceedance Depth" diagnostic (S7) -- NOT an
    Expected Shortfall backtest: no ES forecast is passed in, and no hypothesis
    test is performed. A large
    negative mean_standardized_resid indicates the model UNDERSTATES crash
    severity on the days its VaR is punctured (the model fails hardest when it
    does fail). Report it descriptively.

    Returns {n_exceed, mean_exceedance_loss, mean_standardized_resid}.
    """
    hits = actual < var_pred
    n_exceed = int(np.sum(hits))

    if n_exceed < 1:
        return {
            "n_exceed": n_exceed,
            "mean_exceedance_loss": np.nan,
            "mean_standardized_resid": np.nan,
        }

    sigma_vals = np.asarray(sigma)[hits]
    actual_vals = np.asarray(actual)[hits]

    mean_exceedance_loss = float(np.mean(actual_vals))

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (actual_vals - mu) / sigma_vals
    z = z[np.isfinite(z)]

    mean_standardized_resid = float(np.mean(z)) if len(z) > 0 else np.nan

    return {
        "n_exceed": n_exceed,
        "mean_exceedance_loss": mean_exceedance_loss,
        "mean_standardized_resid": mean_standardized_resid,
    }


def get_regulatory_inspired_zone(failures, total_obs, alpha=0.01):
    """
    REGULATORY-INSPIRED binomial coverage zone (S7 / F3).

    This is a custom, sample-size-adapted binomial classification inspired by
    the Basel traffic-light idea -- it is NOT the formal Basel III / FRTB
    traffic-light table and confers NO regulatory compliance. Returns
    (green_breach_limit, coverage_zone) where coverage_zone in {GREEN, YELLOW,
    RED}.
    """
    p_cum = stats.binom.cdf(failures, total_obs, alpha)
    green_limit = stats.binom.ppf(config.BASEL_GREEN_CUM, total_obs, alpha)
    if p_cum < config.BASEL_GREEN_CUM:
        coverage_zone = "GREEN"
    elif p_cum < config.BASEL_YELLOW_CUM:
        coverage_zone = "YELLOW"
    else:
        coverage_zone = "RED"
    return int(green_limit), coverage_zone


def calculate_metrics(actual_or_df, garch_var=None, tft_var=None, garch_sigma=None, alpha=0.01):
    """
    Universal dispatcher supporting both DataFrame and explicit array parameters.
    If a DataFrame is passed, columns are expected to include:
        Actual or Log_Ret, GARCH_VaR_99, TFT_VaR_99 (or TFT_Downside_99),
        and optionally GARCH_sigma (for the descriptive tail-exceedance-depth
        diagnostic; NOT a formal ES backtest).
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

    limit, coverage_zone = get_regulatory_inspired_zone(kupiec["N"], kupiec["T"], alpha=alpha)

    # Tail breach-depth diagnostic (descriptive; NOT an ES backtest).
    if garch_sigma is not None:
        es = tail_breach_depth_diagnostic(actual, tft_var, garch_sigma)
    else:
        es = {"n_exceed": np.nan, "mean_exceedance_loss": np.nan, "mean_standardized_resid": np.nan}

    return {
        "breaches": kupiec["N"],
        "tft_failures": kupiec["N"],
        "total_obs": kupiec["T"],
        "green_breach_limit": limit,
        "coverage_zone": coverage_zone,
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
        # F3: ES-era key names removed -- these are descriptive tail-exceedance
        # depth statistics, NOT an Expected Shortfall backtest.
        "tail_exceedance_count": es["n_exceed"],
        "tail_mean_return": es.get("mean_exceedance_loss", np.nan),
        "tail_mean_standardized_resid": es.get("mean_standardized_resid", np.nan),
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

    # FIX 10.5: assert all tickers have perfectly synchronized lengths. The
    # trailing truncation below is only correct if the panel shares common end
    # dates; if a future data revision leaves a ticker short at the END (not
    # the start), the trailing truncation would silently misalign the panel.
    lengths = [len(v) for v in actual_dict.values()]
    assert len(set(lengths)) == 1, f"[FATAL] Panel date alignment broken. Lengths: {lengths}"

    min_len = lengths[0]
    for t in tickers:
        actual_dict[t] = actual_dict[t][-min_len:]
        var_dict[t] = var_dict[t][-min_len:]

    # NOTE (Round 23): the multivariate co-breach test was REMOVED -- at
    # T=500, alpha=0.01, K=3 the expected joint-breach count under independence
    # is 0.0005, so the test has essentially zero power and is not meaningful
    # evidence about systemic tail dependence.
    return {"per_ticker": per_ticker}


def aggregate_seed_metrics(metrics_list):
    """
    Aggregates a list of per-seed metric dicts (from calculate_metrics) into
    robust summary rows, with explicit count of seeds. This enforces honest
    statistical disclosure across random seeds instead of reporting a single
    favorable seed.

    S8 (statistical validity): p-values / test statistics are reported PER
    SEED in `values` and summarized by MEDIAN + Std -- the mean of p-values is
    NOT reported because averaging p-values is statistically meaningless.
    Proper scores (e.g. pinball loss) use MEDIAN as the robust central
    statistic and Std as the dispersion.

    Returns a list of dicts, one per distinct metric key, each with:
        metric, mean, std, median, values (list), n_seeds
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

        vals_arr = np.array(vals, dtype=float)
        finite_vals = vals_arr[np.isfinite(vals_arr)]

        # Guard all-NaN slices (e.g. ES t-stat when no seed is testable): the
        # plain np.nanmean/np.nanstd emit a RuntimeWarning and return nan.
        if len(finite_vals) == 0:
            agg_rows.append({
                "metric": k,
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "values": [round(v, 6) for v in vals],
                "n_seeds": len(vals),
            })
            continue

        agg_rows.append({
            "metric": k,
            "mean": float(np.mean(finite_vals)),
            "std": float(np.std(finite_vals, ddof=1)) if len(finite_vals) > 1 else 0.0,
            "median": float(np.median(finite_vals)),
            "values": [round(v, 6) for v in vals],
            "n_seeds": len(vals),
        })

    return agg_rows
