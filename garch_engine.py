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

    print("[GARCH] Starting dynamic Skew-T GJR-GARCH calculation...")
    window_size = 252
    df = df.copy()
    df['GARCH_Resid'] = np.nan
    df['GARCH_Vol'] = np.nan
    df['GARCH_VaR_99'] = np.nan

    indices = range(window_size, len(df))
    iterator = tqdm(indices, desc="[GARCH] Rolling Fit", unit="day") if TQDM_AVAILABLE else indices

    for i in iterator:
        train_window = df['Log_Ret'].iloc[i - window_size: i]
        # Asymmetric Leverage (o=1)
        model = arch_model(train_window, vol='Garch', p=1, o=1, q=1, dist='skewt')
        res = model.fit(disp='off', update_freq=0, show_warning=False)
        forecast = res.forecast(horizon=1, align='origin')

        try:
            nu = res.params.get('nu')
            lam = res.params.get('lambda')
            dynamic_multiplier = model.distribution.ppf(0.01, [nu, lam])
        except (ValueError, TypeError, KeyError):
            dynamic_multiplier = -2.326

        mean_t1 = forecast.mean.iloc[-1, 0]
        vol_t1 = np.sqrt(forecast.variance.iloc[-1, 0])

        df.iloc[i, df.columns.get_loc('GARCH_Resid')] = res.std_resid.iloc[-1]
        df.iloc[i, df.columns.get_loc('GARCH_Vol')] = vol_t1
        df.iloc[i, df.columns.get_loc('GARCH_VaR_99')] = mean_t1 + (vol_t1 * dynamic_multiplier)

    df.dropna(inplace=True)
    df = df.sort_index()
    df['time_idx'] = np.arange(len(df))
    df['group'] = "Nifty50"
    df.to_csv(csv_path, index=True)
    return df
