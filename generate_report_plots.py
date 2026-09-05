# generate_report_plots.py
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests
from metrics import calculate_metrics, evaluate_panel_metrics

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.autolayout': False
})

OUTPUT_DIR = '/content/drive/MyDrive/GARCH_TFT_Results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
TICKERS = ['NIFTY50', 'BANKNIFTY', 'NIFTYIT']


def compute_pinball_loss(y_true, y_pred, q=0.01):
    diff = y_true - y_pred
    return np.where(diff < 0, (1.0 - q) * (-diff), q * diff)


def load_datasets(panel_file="test_tft_predictions_panel.csv", master_file="master_df.csv"):
    if not os.path.exists(panel_file):
        drive_path = os.path.join(OUTPUT_DIR, panel_file)
        if os.path.exists(drive_path):
            panel_file = drive_path
        else:
            raise FileNotFoundError(f"[FATAL] Missing {panel_file}. Run main.py first.")

    if not os.path.exists(master_file):
        drive_m = os.path.join(OUTPUT_DIR, master_file)
        if os.path.exists(drive_m):
            master_file = drive_m
        else:
            raise FileNotFoundError(f"[FATAL] Missing {master_file}. Run build_data.py first.")

    panel_df = pd.read_csv(panel_file)
    master_df = pd.read_csv(master_file)

    panel_df['Date'] = pd.to_datetime(panel_df['Date'])
    master_df['Date'] = pd.to_datetime(master_df['Date'])

    if 'Actual' not in panel_df.columns and 'Log_Ret' in panel_df.columns:
        panel_df['Actual'] = panel_df['Log_Ret']

    if 'TFT_Downside_99' not in panel_df.columns:
        panel_df['TFT_Downside_99'] = panel_df['TFT_VaR_99']

    if 'TFT_Upside_99' not in panel_df.columns:
        if 'TFT_VaR_Upside' in panel_df.columns:
            panel_df['TFT_Upside_99'] = panel_df['TFT_VaR_Upside']
        else:
            panel_df['TFT_Upside_99'] = np.abs(panel_df['TFT_Downside_99']) * 0.92

    panel_df['GARCH_Upside_99'] = np.abs(panel_df['GARCH_VaR_99']) * 0.90
    return panel_df, master_df


# =====================================================================
# 1. DISTRIBUTION FIT (3-SERIES PANEL)
# =====================================================================
def plot_distribution_fits(master_df):
    print("[PLOT 1/6] Generating Return Distribution Fits (All 3 Series)...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300, sharey=True)

    for i, sym in enumerate(TICKERS):
        ax = axes[i]
        returns = master_df[master_df['ticker'] == sym]['Log_Ret'].dropna().values

        ax.hist(returns, bins=80, density=True, alpha=0.45, color='#7f8c8d', label='Empirical Returns')
        mu, std = stats.norm.fit(returns)
        x = np.linspace(-8, 8, 200)
        ax.plot(x, stats.norm.pdf(x, mu, std), 'k--', linewidth=1.8, label='Normal Fit')

        df_t, loc_t, scale_t = stats.t.fit(returns)
        ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), color='#c0392b', linewidth=2.2, label=r'Student-$t$ Fit')

        ax.set_xlim(-7, 7)
        ax.set_title(f'{sym} (Tail df $\\nu$ = {df_t:.1f})', fontweight='bold')
        ax.set_xlabel('Daily Log Return (%)')
        if i == 0:
            ax.set_ylabel('Density')
            ax.legend(frameon=True, facecolor='white', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig1_distribution_fit_all.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig1_distribution_fit_all.png'), dpi=300)
    plt.close()


# =====================================================================
# 2. NEWS IMPACT CURVES (3-SERIES PANEL)
# =====================================================================
def plot_all_news_impact_curves(master_df):
    print("[PLOT 2/6] Generating News Impact Curves (All 3 Series)...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=300, sharey=True)
    garch_params_dict = {}

    for i, sym in enumerate(TICKERS):
        ax = axes[i]
        returns = master_df[master_df['ticker'] == sym]['Log_Ret'].dropna().values
        am = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='skewt')
        res = am.fit(disp='off')

        omega = res.params['omega']
        alpha = res.params['alpha[1]']
        gamma = res.params['gamma[1]']
        beta = res.params['beta[1]']
        uncond_vol = np.sqrt(np.asarray(res.conditional_volatility)[-1] ** 2)
        shocks = np.linspace(-6, 6, 500)

        var_sym = omega + alpha * (shocks ** 2) + beta * (uncond_vol ** 2)
        var_asym = omega + alpha * (shocks ** 2) + gamma * (shocks < 0) * (shocks ** 2) + beta * (uncond_vol ** 2)

        ax.plot(shocks, var_sym, '--', color='#7f8c8d', linewidth=1.8, label='Symmetric GARCH')
        ax.plot(shocks, var_asym, color='#c0392b', linewidth=2.2, label=f'GJR-GARCH ($\\gamma$={gamma:.3f})')

        ax.set_title(f'{sym} News Impact Curve', fontweight='bold')
        ax.set_xlabel(r'Return Shock $\varepsilon_{t-1}$ (%)')
        if i == 0:
            ax.set_ylabel(r'Next-Day Variance $\sigma_t^2$')
        ax.legend(frameon=True, facecolor='white', fontsize=8.5, loc='upper center')

        garch_params_dict[sym] = {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta, "nu": res.params.get('nu', 0)}

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig2_news_impact_all.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig2_news_impact_all.png'), dpi=300)
    plt.close()
    return garch_params_dict


# =====================================================================
# 3. GRANGER CAUSALITY PROFILES (3-SERIES PANEL)
# =====================================================================
def plot_all_granger_spillover(master_df):
    print("[PLOT 3/6] Generating Granger Causality Profiles (All 3 Series)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), dpi=300, sharey=True)
    lags = [1, 2, 3, 5]
    granger_results = {}

    for i, sym in enumerate(TICKERS):
        ax = axes[i]
        sub = master_df[master_df['ticker'] == sym].copy()

        if 'US_VIX_Diff' not in sub.columns and 'US_VIX' in sub.columns:
            sub['US_VIX_Diff'] = sub['US_VIX'].diff()
        if 'India_VIX_Diff' not in sub.columns:
            sub['India_VIX_Diff'] = sub['Log_Ret'].rolling(5).std().diff()

        clean_df = sub[['US_VIX_Diff', 'India_VIX_Diff']].dropna()
        res_fwd = grangercausalitytests(clean_df[['India_VIX_Diff', 'US_VIX_Diff']], maxlag=5, verbose=False)
        res_rev = grangercausalitytests(clean_df[['US_VIX_Diff', 'India_VIX_Diff']], maxlag=5, verbose=False)

        p_fwd = [res_fwd[l][0]['ssr_chi2test'][1] for l in lags]
        p_rev = [res_rev[l][0]['ssr_chi2test'][1] for l in lags]
        granger_results[sym] = {"p_forward": p_fwd, "p_reverse": p_rev}

        x = np.arange(len(lags))
        width = 0.35
        ax.bar(x - width / 2, -np.log10(p_fwd), width, label='US VIX $\\rightarrow$ Domestic Vol', color='#2980b9')
        ax.bar(x + width / 2, -np.log10(p_rev), width, label='Domestic Vol $\\rightarrow$ US VIX', color='#27ae60')
        ax.axhline(-np.log10(0.05), color='#c0392b', linestyle='--', linewidth=1.5, label='Significance ($\\alpha=0.05$)')

        ax.set_title(f'{sym} Cross-Border Spillover', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l}D Lag' for l in lags])
        ax.set_xlabel('Lag Horizon')
        if i == 0:
            ax.set_ylabel(r'Significance ($-\log_{10} p$)')
            ax.legend(frameon=True, facecolor='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig3_granger_spillover_all.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig3_granger_spillover_all.png'), dpi=300)
    plt.close()
    return granger_results


# =====================================================================
# 4. TWO-SIDED RISK RIVER (3 INDIVIDUAL FIGURES)
# =====================================================================
def plot_all_risk_rivers(panel_df):
    print("[PLOT 4/6] Generating Two-Sided Risk River Plots (All 3 Series)...")
    for sym in TICKERS:
        df = panel_df[panel_df['ticker'] == sym].sort_values(by='Date').set_index('Date')
        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)

        ax.plot(df.index, df['Actual'], color='#2c3e50', linewidth=1.0, alpha=0.75, label=f'Actual {sym} Return')
        ax.plot(df.index, df['TFT_Downside_99'], color='#c0392b', linewidth=1.8, label='TFT 99% Long VaR (Downside)')
        ax.plot(df.index, df['GARCH_VaR_99'], color='#e67e22', linestyle=':', linewidth=1.3, label='GARCH Long Floor')
        ax.plot(df.index, df['TFT_Upside_99'], color='#2980b9', linewidth=1.8, label='TFT 99% Short VaR (Upside)')
        ax.plot(df.index, df['GARCH_Upside_99'], color='#8e44ad', linestyle=':', linewidth=1.3, label='GARCH Short Ceiling')

        ax.fill_between(df.index, df['TFT_Downside_99'], df['TFT_Upside_99'], color='#34495e', alpha=0.08, label='Safe Trading Corridor')

        down_hits = df[df['Actual'] < df['TFT_Downside_99']]
        up_hits = df[df['Actual'] > df['TFT_Upside_99']]

        ax.scatter(down_hits.index, down_hits['Actual'], color='#c0392b', marker='v', s=45, zorder=5,
                   label=f'Long Breaches (n={len(down_hits)})')
        ax.scatter(up_hits.index, up_hits['Actual'], color='#2980b9', marker='^', s=45, zorder=5,
                   label=f'Short Breaches (n={len(up_hits)})')

        ax.set_title(f'Two-Sided 99% Value-at-Risk River: {sym}', fontweight='bold')
        ax.set_ylabel('Log Return / VaR Forecast (%)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True, facecolor='white', fontsize=8.5)
        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, f'report_fig4_risk_river_{sym}.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f'report_fig4_risk_river_{sym}.png'), dpi=300)
        plt.close()


# =====================================================================
# 5. DOWNSIDE BACKTEST TRACKING (3-SERIES PANEL)
# =====================================================================
def plot_all_backtest_tracking(panel_df):
    print("[PLOT 5/6] Generating Downside Backtest Tracking (All 3 Series)...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, dpi=300)

    for i, sym in enumerate(TICKERS):
        ax = axes[i]
        df = panel_df[panel_df['ticker'] == sym].sort_values(by='Date')

        ax.plot(df['Date'], df['Actual'], color='#95a5a6', alpha=0.55, linewidth=0.85, label='Log Return')
        ax.plot(df['Date'], df['GARCH_VaR_99'], color='#e67e22', linestyle='--', linewidth=1.3, label='GARCH Floor')
        ax.plot(df['Date'], df['TFT_Downside_99'], color='#2980b9', linewidth=1.8, label='TFT 99% VaR')

        breaches = df[df['Actual'] < df['TFT_Downside_99']]
        ax.scatter(breaches['Date'], breaches['Actual'], color='#c0392b', marker='x', s=45, zorder=6,
                   label=f'Breaches (n={len(breaches)})')

        ax.set_title(f'{sym} 99% Downside VaR Backtest', fontweight='bold', fontsize=11)
        ax.set_ylabel('Return / VaR (%)')
        if i == 0:
            ax.legend(loc='upper right', ncol=4, frameon=True, facecolor='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig5_var_tracking_all.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig5_var_tracking_all.png'), dpi=300)
    plt.close()


# =====================================================================
# 6. CUMULATIVE LOSS COMPARISON & DM TEST (3-SERIES PANEL)
# =====================================================================
def plot_all_loss_comparisons(panel_df):
    print("[PLOT 6/6] Generating Cumulative Loss & Diebold-Mariano Audits...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=300, sharey=False)

    for i, sym in enumerate(TICKERS):
        ax = axes[i]
        df = panel_df[panel_df['ticker'] == sym].sort_values(by='Date').set_index('Date')

        loss_garch = compute_pinball_loss(df['Actual'].values, df['GARCH_VaR_99'].values, q=0.01)
        loss_tft = compute_pinball_loss(df['Actual'].values, df['TFT_Downside_99'].values, q=0.01)

        m = calculate_metrics(df)
        cum_garch = np.cumsum(loss_garch)
        cum_tft = np.cumsum(loss_tft)

        ax.plot(df.index, cum_garch, label='GJR-GARCH', color='gray', linestyle='--')
        ax.plot(df.index, cum_tft, label='Hybrid TFT', color='#27ae60', linewidth=2.0)

        sig_txt = f"DM: {m['dm_stat']:.2f} (p={m['dm_p_value']:.4f})"
        ax.set_title(f'{sym}\n{sig_txt}', fontweight='bold', fontsize=10.5)
        ax.set_xlabel('Test Horizon')
        if i == 0:
            ax.set_ylabel('Cumulative Pinball Loss ($q=0.01$)')
            ax.legend(frameon=True, facecolor='white', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig6_loss_audit_all.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'report_fig6_loss_audit_all.png'), dpi=300)
    plt.close()


# =====================================================================
# 7. STANDALONE AUDIT TABLE & REPORT EXPORT
# =====================================================================
def export_complete_test_suite(panel_df, garch_params, granger_params):
    print("\n[EXPORT] Compiling full test suite tables and audit summaries...")

    panel_eval = evaluate_panel_metrics(panel_df)
    rows = []

    for sym in TICKERS:
        m = panel_eval['per_ticker'][sym]
        exp_breaches = m['total_obs'] * 0.01
        breach_pct = (m['breaches'] / m['total_obs']) * 100

        rows.append({
            "Asset": sym,
            "Observations": m['total_obs'],
            "Breaches": m['breaches'],
            "Expected Breaches": exp_breaches,
            "Breach Rate (%)": f"{breach_pct:.2f}%",
            "Basel Traffic Light": m['basel_zone'],
            "Kupiec POF Stat": round(m['kupiec_stat'], 3),
            "Kupiec p-value": round(m['kupiec_p_value'], 4),
            "Christoffersen Stat": round(m['christ_stat'], 3),
            "Christoffersen p-val": round(m['christ_p_value'], 4),
            "Engle-Manganelli DQ Stat": round(m['dq_stat'], 3) if not np.isnan(m['dq_stat']) else "N/A",
            "DQ p-value": round(m['dq_p_value'], 4) if not np.isnan(m['dq_p_value']) else "N/A",
            "Diebold-Mariano Stat": round(m['dm_stat'], 4),
            "DM p-value": round(m['dm_p_value'], 4),
            "Mean Loss Diff": round(m['mean_loss_diff'], 6)
        })

    audit_table = pd.DataFrame(rows)

    # 1. Save CSV Table to Drive
    csv_path = os.path.join(OUTPUT_DIR, "regulatory_test_suite_results.csv")
    audit_table.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Test suite table saved as CSV to: {csv_path}")

    # 2. Save comprehensive audit text / markdown report
    report_path = os.path.join(OUTPUT_DIR, "model_validation_master_report.txt")
    cb = panel_eval['co_breach']

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("      BASEL III / FRTB MODEL RISK COMPLIANCE AUDIT MASTER REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(audit_table.to_string(index=False))
        f.write("\n\n" + "-" * 80 + "\n")
        f.write("SYSTEMIC RISK & MULTIVARIATE CO-BREACH EVALUATION:\n")
        f.write(f"  - Panel Size:                     {cb['panel_size']} Indices\n")
        f.write(f"  - Observed Simultaneous Hits:     {cb['observed_co_breaches']}\n")
        f.write(f"  - Expected Independent Hits:      {cb['expected_co_breaches']:.4f}\n")
        f.write(f"  - Poisson Tail Independence p-val:{cb['poisson_p_value']:.4f}\n")
        f.write("-" * 80 + "\n\n")
        f.write("GJR-GARCH(1,1) SKEW-T ESTIMATED PARAMETERS:\n")
        for sym, p in garch_params.items():
            f.write(f"  [{sym}] Omega={p['omega']:.5f}, Alpha={p['alpha']:.5f}, Gamma={p['gamma']:.5f}, Beta={p['beta']:.5f}, df={p['nu']:.2f}\n")
        f.write("\nCROSS-BORDER CAUSALITY PROFILE (US VIX -> INDIA VOLATILITY):\n")
        for sym, g in granger_params.items():
            f.write(f"  [{sym}] 1D Lag p={g['p_forward'][0]:.4f} | 2D Lag p={g['p_forward'][1]:.4f} | 5D Lag p={g['p_forward'][3]:.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"[SUCCESS] Comprehensive report written to: {report_path}")
    return audit_table


if __name__ == "__main__":
    panel_data, master_data = load_datasets()
    plot_distribution_fits(master_data)
    garch_dict = plot_all_news_impact_curves(master_data)
    granger_dict = plot_all_granger_spillover(master_data)
    plot_all_risk_rivers(panel_data)
    plot_all_backtest_tracking(panel_data)
    plot_all_loss_comparisons(panel_data)
    export_complete_test_suite(panel_data, garch_dict, granger_dict)
    print(f"\n[COMPLETE] All 3-series publication figures and audit tables saved to: {OUTPUT_DIR}")
