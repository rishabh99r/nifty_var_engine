# production_engine.py
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import torch
from pytorch_forecasting import TemporalFusionTransformer
from config import MAX_ENCODER_LENGTH

def run_live_daily_inference(model_checkpoint_path, live_csv_path="master_df.csv"):
    print("\n[PRODUCTION] === Launching Live End-of-Day VaR Inference Engine ===")

    # 1. Load trained TFT Model
    print(f"[PRODUCTION] Loading checkpoint from {model_checkpoint_path}...")
    tft = TemporalFusionTransformer.load_from_checkpoint(model_checkpoint_path)
    tft.eval()

    # 2. Ingest latest market data buffer (Requires at least 252 + 60 days of history)
    df = pd.read_csv(live_csv_path, index_col=0, parse_dates=True)
    latest_window = df.tail(252).copy()

    print(f"[PRODUCTION] Executing Online GJR-GARCH(1,1) MLE fit on latest 252 days ending {latest_window.index[-1].strftime('%Y-%m-%d')}...")

    # 3. Online GJR-GARCH Fit for t+1 innovation
    model = arch_model(latest_window['Log_Ret'], vol='Garch', p=1, o=1, q=1, dist='skewt')
    res = model.fit(disp='off')

    # Extract 1-step ahead forecast parameters
    forecast = res.forecast(horizon=1, align='origin')
    vol_t1 = np.sqrt(forecast.variance.iloc[-1, 0])
    mean_t1 = forecast.mean.iloc[-1, 0]

    nu = res.params.get('nu', 5.0)
    lam = res.params.get('lambda', -0.1)
    q01_mult = model.distribution.ppf(0.01, [nu, lam])
    parametric_floor_var = mean_t1 + (vol_t1 * q01_mult)

    latest_resid = res.std_resid.iloc[-1]
    print(f"[PRODUCTION] Online GJR Filter Complete | t+1 Vol: {vol_t1:.4f}% | Standardized Shock z_t: {latest_resid:.4f}")

    # 4. Construct 60-Day Encoder Sequence Tensor for TFT
    encoder_df = df.tail(MAX_ENCODER_LENGTH).copy()

    # Predict using PyTorch Forecasting DataLoader
    dataloader = tft.dataset_to_dataloader(
        tft.transform_to_dataset(encoder_df, predict=True), batch_size=1, num_workers=0
    )

    with torch.no_grad():
        raw_preds, _ = tft.predict(dataloader, mode="quantiles", return_index=True)
        tft_var_99 = raw_preds[0, 0, 0].item()

    print("-" * 65)
    print(f"  -> Raw TFT 99% VaR Forecast (t+1):        {tft_var_99:.4f}%")
    print(f"  -> GJR-GARCH Parametric Floor (t+1):      {parametric_floor_var:.4f}%")
    print("-" * 65)

    return tft_var_99, parametric_floor_var, latest_resid
