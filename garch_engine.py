import pandas as pd
import numpy as np
import os
import warnings
from arch import arch_model

# Suppress convergence warnings to keep the Colab console clean
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_rolling_garch(df, csv_path="master_df.csv"):
    # CRITICAL FIX 1: Prevent Date Index destruction on load
    if os.path.exists(csv_path):
        print(f"[GARCH] Found existing {csv_path}. Skipping 1-hour calculation.")
        # Must parse dates and set index to preserve PyTorch Forecasting time structures
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    print("[GARCH] No saved data. Starting dynamic Skew-T GARCH calculation...")

    window_size = 252 # 1 trading year
    df['GARCH_Resid'] = np.nan
    df['GARCH_Vol'] = np.nan
    df['GARCH_VaR_99'] = np.nan

    for i in range(window_size, len(df)):
        train_window = df['Log_Ret'].iloc[i - window_size : i]

        # Fit Skew-T GARCH(1,1) with a Constant Mean
        model = arch_model(train_window, vol='Garch', p=1, q=1, dist='skewt')
        res = model.fit(disp='off', update_freq=0, show_warning=False)

        # Predict t+1 Mean and Volatility
        forecast = res.forecast(horizon=1, align='origin')

        try:
            # The arch library stores the parameter as 'lambda', not 'lam'
            nu = res.params.get('nu')
            lam = res.params.get('lambda')

            if pd.isna(nu) or pd.isna(lam) or nu is None or lam is None:
                raise ValueError("Optimizer failed to converge.")

            # Extract dynamic multiplier (PPF)
            dynamic_multiplier = model.distribution.ppf(0.01, [nu, lam])

        except (ValueError, TypeError, KeyError):
            # Fallback to standard normal 1% quantile if Skew-T parameters fail
            dynamic_multiplier = -2.326

        # CRITICAL FIX 2: Econometric Purity (Include the forecasted mean)
        mean_t1 = forecast.mean.iloc[-1, 0]
        vol_t1 = np.sqrt(forecast.variance.iloc[-1, 0])

        # Save metrics
        df.iloc[i, df.columns.get_loc('GARCH_Resid')] = res.resid.iloc[-1]
        df.iloc[i, df.columns.get_loc('GARCH_Vol')] = vol_t1

        # True VaR = mu + (sigma * PPF)
        df.iloc[i, df.columns.get_loc('GARCH_VaR_99')] = mean_t1 + (vol_t1 * dynamic_multiplier)

        if i % 500 == 0:
            print(f"[GARCH] Processed {i}/{len(df)} days...")

    print("[GARCH] Cleaning up burn-in period and building PyTorch indices...")
    df.dropna(inplace=True)

    # Sort index to guarantee time_idx integrity
    df = df.sort_index()
    df['time_idx'] = np.arange(len(df))
    df['group'] = "Nifty50"

    # Save WITH the index
    df.to_csv(csv_path, index=True)
    print(f"[GARCH] Dynamic GARCH calculations complete and saved to {csv_path}.")

    return df
