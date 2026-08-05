import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

def plot_distribution_fit(df_path="master_df.csv"):
    print("[PLOT] 1/4: Nifty 50 Return Distribution vs. Normal vs. Skew-T...")
    if not os.path.exists(df_path): return

    df = pd.read_csv(df_path)
    returns = df['Log_Ret'].dropna().values

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.hist(returns, bins=100, density=True, alpha=0.5, color='#7f8c8d', label='Empirical Nifty 50 Returns')

    mu, std = stats.norm.fit(returns)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p_norm = stats.norm.pdf(x, mu, std)
    ax.plot(x, p_norm, 'k--', linewidth=2, label=r'Normal Fit (Fails to capture tails)')

    df_t, loc_t, scale_t = stats.t.fit(returns)
    p_t = stats.t.pdf(x, df_t, loc_t, scale_t)
    ax.plot(x, p_t, color='#c0392b', linewidth=2.5, label=r'Student-$t$ Fit (Captures leptokurtosis)')

    ax.set_xlim(-8, 8)
    ax.set_title('Nifty 50 Daily Returns: The Failure of Gaussian Assumptions')
    ax.set_xlabel('Daily Log Return (%)')
    ax.set_ylabel('Density')
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig('report_01_distribution.png', dpi=300)
    plt.close()

def plot_risk_river(pred_path="test_tft_predictions.csv", master_path="master_df.csv"):
    print("[PLOT] 2/4: Generating Risk River Backtest Plot...")
    if not os.path.exists(pred_path) or not os.path.exists(master_path): return

    # CRITICAL FIX: Dynamically merge predictions with master data to retrieve Actuals & Dates
    preds = pd.read_csv(pred_path)
    master = pd.read_csv(master_path)

    # Extract date column (Column 0)
    date_col = master.columns[0]
    master[date_col] = pd.to_datetime(master[date_col])

    df = preds.merge(master[[date_col, 'time_idx', 'Log_Ret', 'GARCH_VaR_99']], on='time_idx', how='inner')
    df.rename(columns={'Log_Ret': 'Actual'}, inplace=True)
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(df.index, df['Actual'], color='#2c3e50', linewidth=1.2, label='Actual Nifty Return', alpha=0.85)
    ax.plot(df.index, df['TFT_VaR_99'], color='#e74c3c', linewidth=2, label='Hybrid TFT 99% VaR Limit')
    ax.plot(df.index, df['GARCH_VaR_99'], color='#f39c12', linestyle='--', linewidth=1.5, label='GJR-GARCH 99% VaR Floor')

    ax.fill_between(df.index, df['TFT_VaR_99'], df['GARCH_VaR_99'], color='#e74c3c', alpha=0.15, label='Neural Adaptation Band')

    breaches = df[df['Actual'] < df['TFT_VaR_99']]
    ax.scatter(breaches.index, breaches['Actual'], color='black', marker='x', s=50, zorder=5, label='VaR Exceptions')

    ax.set_title('Out-of-Sample Risk River: Actual Returns vs. 99% VaR Boundaries')
    ax.set_ylabel('Log Return / VaR Forecast (%)')
    ax.legend(loc='lower left', frameon=True, facecolor='white', fontsize=9)
    plt.tight_layout()
    plt.savefig('report_02_risk_river.png', dpi=300)
    plt.close()

def plot_vsn_importance(vsn_path="vsn_feature_importance.csv"):
    print("[PLOT] 3/4: Generating Variable Selection Network (VSN) Importance...")
    if not os.path.exists(vsn_path): return

    vsn_df = pd.read_csv(vsn_path).sort_values(by="Percentage", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.barh(vsn_df['Feature'], vsn_df['Percentage'], color='#2980b9')

    for bar in bars:
        ax.annotate(f"{bar.get_width():.1f}%",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=9)

    ax.set_title('TFT Variable Selection Network (VSN): Macro Feature Attribution')
    ax.set_xlabel('Relative Importance Weight (%)')
    ax.set_xlim(0, vsn_df['Percentage'].max() * 1.15)
    plt.tight_layout()
    plt.savefig('report_03_vsn_importance.png', dpi=300)
    plt.close()

def plot_temporal_attention(attn_path="temporal_attention_distribution.csv"):
    print("[PLOT] 4/4: Generating Multi-Head Temporal Attention Curve...")
    if not os.path.exists(attn_path): return

    attn_df = pd.read_csv(attn_path)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(attn_df['Lookback_Lag_Days'], attn_df['Attention_Weight'] * 100, color='#8e44ad', linewidth=2)
    ax.fill_between(attn_df['Lookback_Lag_Days'], attn_df['Attention_Weight'] * 100, color='#8e44ad', alpha=0.2)

    ax.set_title('Temporal Self-Attention: Lookback Memory Distribution')
    ax.set_xlabel('Historical Lag (Trading Days Prior to Forecast)')
    ax.set_ylabel('Attention Weight (%)')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('report_04_temporal_attention.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_distribution_fit()
    plot_risk_river()
    plot_vsn_importance()
    plot_temporal_attention()
    print("[SUCCESS] All interview charts generated.")
