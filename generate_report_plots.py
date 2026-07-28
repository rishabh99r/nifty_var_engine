# generate_report_plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def plot_news_impact_curve():
    print("[PLOT] Generating News Impact Curve (NIC)...")

    # Range of return shocks from -6% to +6%
    eps = np.linspace(-6.0, 6.0, 500)

    # Standard baseline parameters for daily Nifty 50 returns
    omega = 0.05
    alpha = 0.06
    beta = 0.85
    gamma_gjr = 0.1204  # Empirically derived from proof.py

    # 1. Symmetric GARCH(1,1): sigma^2 = omega + alpha * eps^2 + beta * sigma_prev^2
    # For standalone NIC, we evaluate the direct impulse: omega + alpha * eps^2
    nic_garch = omega + alpha * (eps ** 2)

    # 2. Asymmetric GJR-GARCH(1,1): adds gamma * eps^2 when eps < 0
    indicator = (eps < 0).astype(float)
    nic_gjr = omega + (alpha + gamma_gjr * indicator) * (eps ** 2)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(eps, nic_garch, label='Symmetric GARCH(1,1) (Omitted Leverage)', color='#7f8c8d', linestyle='--', linewidth=2)
    ax.plot(eps, nic_gjr, label='GJR-GARCH(1,1) ($\gamma = 0.1204$, Empirical Supreme)', color='#c0392b', linewidth=2.5)

    # Highlight crash disparity at -4% shock
    shock_val = -4.0
    var_garch_4 = omega + alpha * (shock_val ** 2)
    var_gjr_4 = omega + (alpha + gamma_gjr) * (shock_val ** 2)

    ax.vlines(x=shock_val, ymin=0, ymax=var_gjr_4, color='#2c3e50', linestyle=':', alpha=0.7)
    ax.scatter([shock_val, shock_val], [var_garch_4, var_gjr_4], color='#2c3e50', zorder=5)

    ax.annotate(f'Leverage Premium:\n+{((var_gjr_4/var_garch_4)-1)*100:.1f}% Variance',
                xy=(shock_val, var_gjr_4), xytext=(-5.5, var_gjr_4 * 0.85),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", lw=1.5))

    ax.set_title('News Impact Curve: Conditional Variance vs. Return Shocks ($\varepsilon_{t-1}$)')
    ax.set_xlabel('Previous Day Return Shock ($\varepsilon_{t-1}$ in %)')
    ax.set_ylabel('Conditional Variance ($\sigma_t^2$)')
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper center', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('report_fig1_news_impact_curve.png', dpi=300)
    plt.savefig('report_fig1_news_impact_curve.pdf')
    plt.close()
    print("  -> Saved: report_fig1_news_impact_curve.png / .pdf")

def plot_granger_spillover():
    print("[PLOT] Generating Granger Causality Spillover Profile...")

    lags = ['1 Day', '2 Days', '3 Days', '5 Days']

    # Empirical p-values derived from proof.py
    # Note: Using 1e-5 for display purposes where p=0.0000
    p_vals_us_to_ind = [0.0213, 1e-5, 1e-5, 1e-5]
    p_vals_ind_to_us = [1e-5, 1e-5, 1e-5, 1e-5]

    x = np.arange(len(lags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    # Plot log-transformed p-values for clear visual scaling: -log10(p)
    log_p_us_ind = -np.log10(p_vals_us_to_ind)
    log_p_ind_us = -np.log10(p_vals_ind_to_us)

    rects1 = ax.bar(x - width/2, log_p_us_ind, width, label='US VIX $\to$ India VIX (Wall St $\to$ Dalal St)', color='#2980b9')
    rects2 = ax.bar(x + width/2, log_p_ind_us, width, label='India VIX $\to$ US VIX (Dalal St $\to$ Wall St)', color='#27ae60')

    # Significance threshold at alpha = 0.05 -> -log10(0.05) = 1.301
    thresh = -np.log10(0.05)
    ax.axhline(y=thresh, color='#c0392b', linestyle='--', linewidth=1.5, label='Significance Threshold ($\alpha = 0.05$)')

    ax.set_title('Granger Causality Volatility Spillover Profile Across Time Lags')
    ax.set_xlabel('Vector Autoregression (VAR) Lag Horizon')
    ax.set_ylabel('Statistical Significance ($-\log_{10} p\text{-value}$)')
    ax.set_xticks(x)
    ax.set_xticklabels(lags)
    ax.set_ylim(0, 6)

    # Annotate bars
    for rect in rects1:
        height = rect.get_height()
        val_str = "p=0.021" if height < 2 else "p<0.001"
        ax.annotate(val_str, xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate("p<0.001", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('report_fig2_granger_spillover.png', dpi=300)
    plt.savefig('report_fig2_granger_spillover.pdf')
    plt.close()
    print("  -> Saved: report_fig2_granger_spillover.png / .pdf")

if __name__ == "__main__":
    plot_news_impact_curve()
    plot_granger_spillover()
    print("[PLOT] All report visualizations generated successfully.")
