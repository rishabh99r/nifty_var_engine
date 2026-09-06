# production_engine.py
# =============================================================================
# Live daily inference for the Econometrically-Conditioned TFT.
#
# The forecast VaR is the RAW TFT quantile -- NO GARCH output floor is applied.
# This guarantees the deployed model is IDENTICAL to the validated model
# (validated == deployed). GARCH_sigma remains an INPUT prior fed through the
# Variable Selection Network, not an output-level override.
# =============================================================================
import os
import warnings

import numpy as np
import pandas as pd
import torch
from arch import arch_model
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

import config
from metrics import extract_garch_dist_params

warnings.filterwarnings("ignore")


def run_live_daily_inference(model_checkpoint_path, live_csv_path="master_df.csv", target_ticker="NIFTY50"):
    """
    Executes live End-of-Day (15:30 IST) VaR forecasting for target_ticker.
    Produces 1-step-ahead (t+1) 99% VaR bound using the RAW TFT quantile.
    """
    print(f"\n[PRODUCTION] === Launching Live VaR Engine for: {target_ticker} ===")

    if not os.path.exists(live_csv_path):
        raise FileNotFoundError(f"[ERROR] Live dataset buffer '{live_csv_path}' missing.")
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"[ERROR] Trained model checkpoint '{model_checkpoint_path}' missing.")

    # 1. Load trained TFT Model
    print(f"[PRODUCTION] Loading model weights from: {model_checkpoint_path}...")
    tft = TemporalFusionTransformer.load_from_checkpoint(model_checkpoint_path)
    tft.eval()

    # 2. Ingest market data buffer
    master_df = pd.read_csv(live_csv_path)

    # FIX 8.7 (schema-drift guard): the live buffer must carry every column the
    # checkpoint's dataset expects (e.g. Log_Ret_Feature), else from_parameters
    # fails silently or feeds NaNs. Rebuild the live buffer with the SAME
    # build_data.py that produced the training frame.
    required_cols = [c for c in (tft.dataset_parameters.get("time_varying_known_reals") or [])
                     + (tft.dataset_parameters.get("time_varying_unknown_reals") or [])]
    missing = [c for c in required_cols if c not in master_df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] Live buffer missing columns required by the checkpoint: {missing}. "
            f"Re-run build_data.py to regenerate master_df.csv with the same schema."
        )

    target_series = master_df[master_df["ticker"] == target_ticker].sort_values(by="time_idx").copy()

    if len(target_series) < config.LOOKBACK_DAYS:
        print(f"[WARNING] History depth ({len(target_series)}) < {config.LOOKBACK_DAYS} days. Using full available history.")
        history_window = target_series.copy()
    else:
        history_window = target_series.tail(config.LOOKBACK_DAYS).copy()

    # 3. Online Point-in-Time GJR-GARCH(1,1) Skew-T estimation for the
    #    econometric PRIOR (input feature), NOT an output override.
    print(f"[PRODUCTION] Fitting PIT Skew-T GJR-GARCH on prior {len(history_window)} days...")
    model = arch_model(history_window["Log_Ret"], mean="Constant", vol="Garch", p=1, o=1, q=1, dist="skewt")
    res = model.fit(disp="off", show_warning=False)

    forecast = res.forecast(horizon=1, align="origin")
    sigma_t1 = np.sqrt(forecast.variance.iloc[-1, 0])
    mu_t1 = forecast.mean.iloc[-1, 0]

    # Dynamic quantile multiplier (q = 0.01) -- FIX 8.6: use the hardened
    # extractor instead of fragile positional indexing (params[-2:]) which can
    # silently corrupt the GARCH reference if arch reorders its parameters.
    shape = extract_garch_dist_params(res)
    nu = shape["nu"] if not np.isnan(shape["nu"]) else 5.0
    lam = shape["lambda"] if not np.isnan(shape["lambda"]) else 0.0
    q01_multiplier = model.distribution.ppf(0.01, [nu, lam])
    garch_floor_var = mu_t1 + (sigma_t1 * q01_multiplier)
    latest_resid = res.std_resid.iloc[-1]

    # 4. Construct inference context for TFT (panel structure preserved)
    max_time_idx = master_df["time_idx"].max()
    encoder_df = master_df[master_df["time_idx"] > (max_time_idx - config.ENCODER_LENGTH)].copy()
    encoder_df["ticker"] = encoder_df["ticker"].astype(str)

    inference_dataset = TimeSeriesDataSet.from_parameters(
        tft.dataset_parameters,
        encoder_df,
        predict=True,
        stop_randomization=True,
    )
    inference_dataloader = inference_dataset.to_dataloader(
        batch_size=len(master_df["ticker"].unique()), num_workers=0
    )

    with torch.no_grad():
        preds, index_df = tft.predict(inference_dataloader, mode="quantiles", return_index=True)
        pred_values = preds.cpu().numpy()

    target_row_idx = index_df[index_df["ticker"] == target_ticker].index[0]
    raw_tft_var_99 = float(pred_values[target_row_idx, 0, 0])  # q = 0.01
    raw_tft_median = float(pred_values[target_row_idx, 0, 1])  # q = 0.50
    raw_tft_upside = float(pred_values[target_row_idx, 0, 2])  # q = 0.99

    # 5. The final VaR is the RAW TFT quantile (validated == deployed).
    #    GARCH sigma is reported only as the econometric prior, not applied.
    final_var_99 = raw_tft_var_99

    print("\n" + "=" * 70)
    print(f"  TARGET ASSET:                       {target_ticker}")
    print(f"  Forecast Horizon:                   Next Trading Session (t+1)")
    print(f"  Econometric Prior sigma_t+1:        {sigma_t1:.4f}%")
    print(f"  GJR-GARCH 99% Parametric Reference: {garch_floor_var:.4f}% (prior only, NOT applied)")
    print(f"  Raw ECTFT 99% Quantile Forecast:    {raw_tft_var_99:.4f}%")
    print(f"  -> FINAL REGULATORY 99% VaR BOUND:  {final_var_99:.4f}%")
    print("=" * 70 + "\n")

    return {
        "ticker": target_ticker,
        "final_var_99": final_var_99,
        "raw_tft_var_99": raw_tft_var_99,
        "garch_floor_var": garch_floor_var,
        "sigma_t1": sigma_t1,
        "latest_resid": latest_resid,
    }


if __name__ == "__main__":
    import glob
    ckpt_matches = glob.glob(os.path.join(config.OUTPUT_DIR, "*.ckpt"))
    if ckpt_matches:
        best_ckpt = ckpt_matches[0]
        run_live_daily_inference(best_ckpt, live_csv_path="master_df.csv", target_ticker="NIFTY50")
    else:
        print("[INFO] No trained model checkpoint found on Drive. Run main.py first.")
