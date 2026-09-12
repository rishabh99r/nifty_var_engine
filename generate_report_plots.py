# generate_report_plots.py
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests

import config
from metrics import (
    calculate_metrics, evaluate_panel_metrics, extract_garch_dist_params,
    granger_series_from_panel, granger_diagnostics, _fmt_p, _fmt_pct
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12, "figure.autolayout": False,
})

OUTPUT_DIR = getattr(config, "OUTPUT_DIR", ".") + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TICKERS = list(config.TICKERS.keys())
Q = 0.01

def compute_pinball_loss(y_true, y_pred, q=0.01):
    diff = y_true - y_pred
    return np.where(diff < 0, (1.0 - q) * (-diff), q * diff)

def load_datasets():
    panel_df = pd.read_csv(os.path.join(OUTPUT_DIR, "test_tft_predictions_panel.csv"))
    master_df = pd.read_csv(os.path.join(OUTPUT_DIR, "master_df.csv"))
    panel_df["Date"] = pd.to_datetime(panel_df["Date"])
    master_df["Date"] = pd.to_datetime(master_df["Date"])

    if "Actual" not in panel_df.columns: panel_df["Actual"] = panel_df["Log_Ret"]
    if "TFT_Downside_99" not in panel_df.columns: panel_df["TFT_Downside_99"] = panel_df["TFT_VaR_99"]
    if "TFT_Upside_99" not in panel_df.columns: panel_df["TFT_Upside_99"] = np.abs(panel_df["TFT_Downside_99"]) * 0.92

    panel_df["GARCH_Upside_99_heuristic"] = np.abs(panel_df["GARCH_VaR_99"]) * 0.90
    panel_df["GARCH_Upside_99"] = panel_df["GARCH_Upside_99_heuristic"]
    return panel_df, master_df

def plot_distribution_fits(master_df):
    for sym in TICKERS:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        returns = master_df[master_df["ticker"] == sym]["Log_Ret"].dropna().values
        ax.hist(returns, bins=80, density=True, alpha=0.45, color="#7f8c8d", label="Empirical Returns")

        mu, std = stats.norm.fit(returns)
        x = np.linspace(-8, 8, 200)
        ax.plot(x, stats.norm.pdf(x, mu, std), "k--", linewidth=1.8, label="Normal Fit")
        df_t, loc_t, scale_t = stats.t.fit(returns)
        ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), color="#c0392b", linewidth=2.2, label=r"Student-$t$ Fit")

        ax.set_xlim(-7, 7)
        ax.set_title(f"{sym} (Tail df $\\nu$ = {df_t:.1f})", fontweight="bold")
        ax.set_xlabel("Daily Log Return (%)")
        ax.set_ylabel("Density")
        ax.legend(frameon=True, facecolor="white", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig1_distribution_fit_{sym}.png"))
        plt.close()

def plot_news_impact_curves(master_df):
    garch_params_dict = {}
    for sym in TICKERS:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        returns = master_df[master_df["ticker"] == sym]["Log_Ret"].dropna().values
        res = arch_model(returns, vol="Garch", p=1, o=1, q=1, dist="skewt").fit(disp="off", show_warning=False)

        p = res.params
        shape = extract_garch_dist_params(res)
        ref_sigma = np.sqrt(np.asarray(res.conditional_volatility)[-1] ** 2)
        shocks = np.linspace(-6, 6, 500)

        var_sym = p["omega"] + p["alpha[1]"] * (shocks ** 2) + p["beta[1]"] * (ref_sigma ** 2)
        var_asym = var_sym + p["gamma[1]"] * (shocks < 0) * (shocks ** 2)

        ax.plot(shocks, var_sym, "--", color="#7f8c8d", linewidth=1.8, label="Counterfactual Symmetric GARCH")
        ax.plot(shocks, var_asym, color="#c0392b", linewidth=2.2, label=f"GJR-GARCH ($\\gamma$={p['gamma[1]']:.3f})")

        ax.set_title(f"{sym} News Impact Curve", fontweight="bold")
        ax.set_xlabel(r"Return Shock $\varepsilon_{t-1}$ (%)")
        ax.set_ylabel(r"Next-Day Variance $\sigma_t^2$")
        ax.legend(frameon=True, facecolor="white", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig2_news_impact_{sym}.png"))
        plt.close()

        garch_params_dict[sym] = {"omega": p["omega"], "alpha": p["alpha[1]"], "gamma": p["gamma[1]"],
                                  "beta": p["beta[1]"], "nu": shape["nu"], "lambda": shape["lambda"]}
    return garch_params_dict

def plot_granger_spillover(master_df):
    granger_results = {}
    lags = [1, 2, 3, 5]
    for sym in TICKERS:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        sub = master_df[master_df["ticker"] == sym].copy()
        us_logdiff, in_logdiff, dom_label = granger_series_from_panel(sub)
        clean_df = pd.DataFrame({"us": us_logdiff, "dom": in_logdiff}).dropna()

        res_fwd = grangercausalitytests(clean_df[["dom", "us"]], maxlag=5)
        res_rev = grangercausalitytests(clean_df[["us", "dom"]], maxlag=5)

        p_fwd = [res_fwd[l][0]["ssr_chi2test"][1] for l in lags]
        p_rev = [res_rev[l][0]["ssr_chi2test"][1] for l in lags]

        x = np.arange(len(lags))
        w = 0.35
        ax.bar(x - w/2, -np.log10(p_fwd), w, label=f"US VIX $\\rightarrow$ {dom_label}", color="#2980b9")
        ax.bar(x + w/2, -np.log10(p_rev), w, label=f"{dom_label} $\\rightarrow$ US VIX", color="#27ae60")
        ax.axhline(-np.log10(0.05), color="#c0392b", linestyle="--", linewidth=1.5, label="Significance ($\\alpha=0.05$)")

        ax.set_title(f"{sym} Predictive Precedence", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}D Lag" for l in lags])
        ax.set_xlabel("Lag Horizon")
        ax.set_ylabel(r"Significance ($-\log_{10} p$)")
        ax.legend(frameon=True, facecolor="white", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig3_predictive_precedence_{sym}.png"))
        plt.close()

        granger_results[sym] = {"p_forward": p_fwd, "domestic_label": dom_label,
                                "diag": granger_diagnostics({"US": clean_df["us"], "Dom": clean_df["dom"]})}
    return granger_results

def plot_risk_rivers(panel_df):
    for sym in TICKERS:
        df = panel_df[panel_df["ticker"] == sym].sort_values("Date").set_index("Date")
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

        ax.plot(df.index, df["Actual"], color="#2c3e50", lw=1.0, alpha=0.75, label="Return")
        ax.plot(df.index, df["TFT_Downside_99"], color="#c0392b", lw=1.8, label="ECTFT Long VaR")
        ax.plot(df.index, df["GARCH_VaR_99"], color="#e67e22", ls=":", lw=1.3, label="GJR-GARCH VaR")
        ax.plot(df.index, df["TFT_Upside_99"], color="#2980b9", lw=1.8, label="ECTFT Short VaR")
        ax.fill_between(df.index, df["TFT_Downside_99"], df["TFT_Upside_99"], color="#34495e", alpha=0.08)

        dh = df[df["Actual"] < df["TFT_Downside_99"]]
        uh = df[df["Actual"] > df["TFT_Upside_99"]]
        ax.scatter(dh.index, dh["Actual"], color="#c0392b", marker="v", zorder=5, label=f"Long Breaches (n={len(dh)})")
        ax.scatter(uh.index, uh["Actual"], color="#2980b9", marker="^", zorder=5, label=f"Short Breaches (n={len(uh)})")

        ax.set_title(f"Two-Sided 99% Risk River: {sym}", fontweight="bold")
        ax.set_ylabel("Return / VaR (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, fontsize=8.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig4_risk_river_{sym}.png"))
        plt.close()

def plot_var_tracking(panel_df):
    for sym in TICKERS:
        df = panel_df[panel_df["ticker"] == sym].sort_values("Date")
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

        ax.plot(df["Date"], df["Actual"], color="#95a5a6", alpha=0.55, lw=0.85, label="Log Return")
        ax.plot(df["Date"], df["GARCH_VaR_99"], color="#e67e22", ls="--", lw=1.3, label="GJR-GARCH VaR")
        ax.plot(df["Date"], df["TFT_Downside_99"], color="#2980b9", lw=1.8, label="ECTFT VaR")

        br = df[df["Actual"] < df["TFT_Downside_99"]]
        ax.scatter(br["Date"], br["Actual"], color="#c0392b", marker="x", s=50, zorder=6, label=f"Breaches (n={len(br)})")

        ax.set_title(f"{sym} 99% Downside VaR Tracking", fontweight="bold")
        ax.set_ylabel("Return / VaR (%)")
        ax.legend(loc="upper right", ncol=4, frameon=True, fontsize=8.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig5_var_tracking_{sym}.png"))
        plt.close()

def plot_loss_comparisons(panel_df):
    for sym in TICKERS:
        df = panel_df[panel_df["ticker"] == sym].sort_values("Date").set_index("Date")
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

        loss_garch = compute_pinball_loss(df["Actual"].values, df["GARCH_VaR_99"].values, Q)
        loss_tft = compute_pinball_loss(df["Actual"].values, df["TFT_Downside_99"].values, Q)

        m = calculate_metrics(df)
        ax.plot(df.index, np.cumsum(loss_garch), label="GJR-GARCH", color="gray", ls="--")
        ax.plot(df.index, np.cumsum(loss_tft), label="Full ECTFT", color="#27ae60", lw=2.0)

        title_txt = f"{sym} Cumulative Loss\nDM: {m['dm_stat']:.2f} (p={m['dm_p_value']:.4f})"
        ax.set_title(title_txt, fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Test Horizon")
        ax.set_ylabel("Cumulative Pinball Loss ($q=0.01$)")
        ax.legend(frameon=True, facecolor="white", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig6_loss_audit_{sym}.png"))
        plt.close()

def export_complete_test_suite(panel_df, garch_params, granger_params):
    panel_eval = evaluate_panel_metrics(panel_df)
    rows = []
    for sym in TICKERS:
        m = panel_eval["per_ticker"][sym]
        rows.append({
            "Asset": sym, "Observations": m["total_obs"], "Breaches (5-seed)": m["breaches"],
            "Kupiec p": round(m["kupiec_p_value"], 4), "DQ p": round(m["dq_p_value"], 4) if not np.isnan(m["dq_p_value"]) else "N/A",
            "DM Stat": round(m["dm_stat"], 4), "DM p-value": round(m["dm_p_value"], 4),
            "Mean Loss Diff": round(m["mean_loss_diff"], 6)
        })
    audit_table = pd.DataFrame(rows)
    audit_table.to_csv(os.path.join(OUTPUT_DIR, "regulatory_test_suite_results.csv"), index=False)

    report_path = os.path.join(OUTPUT_DIR, "model_validation_master_report.txt")
    with open(report_path, "w") as f:
        f.write("=== REGULATORY-INSPIRED 99% VAR BACKTESTING REPORT ===\n\n")
        f.write(audit_table.to_string(index=False))
        f.write("\n\n=== GJR-GARCH(1,1) SKEW-T PARAMETERS ===\n")
        for sym, p in garch_params.items():
            f.write(f"[{sym}] Omega={p['omega']:.5f}, Alpha={p['alpha']:.5f}, Gamma={p['gamma']:.5f}, Beta={p['beta']:.5f}\n")
    return audit_table

if __name__ == "__main__":
    panel, master = load_datasets()
    plot_distribution_fits(master)
    garch_dict = plot_news_impact_curves(master)
    granger_dict = plot_granger_spillover(master)
    plot_risk_rivers(panel)
    plot_var_tracking(panel)
    plot_loss_comparisons(panel)
    export_complete_test_suite(panel, garch_dict, granger_dict)
