# garch_engine.py
import pandas as pd
import numpy as np
import os
import warnings
from arch import arch_model

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_rolling_garch(df, csv_path="master_df.csv"):
    if os.path.exists(csv_path):
        print(f"[GARCH] Found existing {csv_path}. Skipping calculation.")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    print("[GARCH] No saved data. Starting dynamic Skew-T GJR-GARCH calculation...")
    print("[GARCH] NOTE: ~4,500 model fits on first run. Expect 60-90 mins on Colab CPU.")

    window_size = 252
    df = df.copy()
    df['GARCH_Resid'] = np.nan
    df['GARCH_Vol'] = np.nan
    df['GARCH_VaR_99'] = np.nan

    indices = range(window_size, len(df))
    iterator = tqdm(indices, desc="[GARCH] Rolling Fit", unit="day") if TQDM_AVAILABLE else indices

    for i in iterator:
        train_window = df['Log_Ret'].iloc[i - window_size: i]

        model = arch_model(train_window, vol='Garch', p=1, q=1, dist='skewt')
        res = model.fit(disp='off', update_freq=0, show_warning=False)

        forecast = res.forecast(horizon=1, align='origin')

        try:
            nu = res.params.get('nu')
            lam = res.params.get('lambda')

            if pd.isna(nu) or pd.isna(lam) or nu is None or lam is None:
                raise ValueError("Optimizer failed to converge.")

            dynamic_multiplier = model.distribution.ppf(0.01, [nu, lam])

        except (ValueError, TypeError, KeyError):
            dynamic_multiplier = -2.326  # Fallback: standard normal 1% quantile

        mean_t1 = forecast.mean.iloc[-1, 0]
        vol_t1 = np.sqrt(forecast.variance.iloc[-1, 0])

        # FIX (Point 2): Inject STANDARDIZED residual z_t = e_t / sigma_t
        # res.std_resid gets the scale-free Skew-t innovation
        df.iloc[i, df.columns.get_loc('GARCH_Resid')] = res.std_resid.iloc[-1]
        df.iloc[i, df.columns.get_loc('GARCH_Vol')] = vol_t1
        df.iloc[i, df.columns.get_loc('GARCH_VaR_99')] = mean_t1 + (vol_t1 * dynamic_multiplier)

        if not TQDM_AVAILABLE and i % 500 == 0:
            print(f"[GARCH] Processed {i}/{len(df)} days...")

    print("[GARCH] Cleaning up burn-in period and building PyTorch indices...")
    df.dropna(inplace=True)
    df = df.sort_index()
    df['time_idx'] = np.arange(len(df))
    df['group'] = "Nifty50"

    df.to_csv(csv_path, index=True)
    print(f"[GARCH] Complete. Saved to {csv_path}.")

    return df
