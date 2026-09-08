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
from predict_utils import unpack_predictions

warnings.filterwarnings("ignore")


def _infer_quantiles_one_checkpoint(model_checkpoint_path, master_df, target_ticker):
    """
    Loads a single checkpoint and returns the raw (q0.01, q0.50, q0.99)
    quantile forecasts for the terminal trading day of the buffer.
    """
    tft = TemporalFusionTransformer.load_from_checkpoint(model_checkpoint_path)
    tft.eval()

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
        pred_values, index_df = unpack_predictions(
            tft.predict(inference_dataloader, mode="quantiles", return_index=True)
        )

    target_row_idx = index_df[index_df["ticker"] == target_ticker].index[0]
    return (
        float(pred_values[target_row_idx, 0, 0]),  # q = 0.01
        float(pred_values[target_row_idx, 0, 1]),  # q = 0.50
        float(pred_values[target_row_idx, 0, 2]),  # q = 0.99
    )


def run_live_ensemble_inference(checkpoint_paths, live_csv_path="master_df.csv", target_ticker="NIFTY50"):
    """
    Live 1-step-ahead 99% VaR from the pre-determined ENSEMBLE rule: the mean
    of the seed q0.01 (and q0.50 / q0.99) forecasts across all seed checkpoints.
    This mirrors the backtest-time ensemble exactly (validated == deployed).

    S3 (model governance): HARD-FAILS unless exactly the validated number of
    seed checkpoints is supplied. A partial (1- or 2-model) ensemble is never
    silently served -- the backtest was validated on N seeds, so deployment
    must use the same N or refuse to forecast.
    """
    expected_models = len(config.VALIDATION_SEEDS)
    if len(checkpoint_paths) != expected_models:
        raise RuntimeError(
            f"[FATAL GOVERNANCE ERROR] Expected exactly {expected_models} seed "
            f"checkpoints for the ensemble (validated seeds "
            f"{list(config.VALIDATION_SEEDS)}), but found {len(checkpoint_paths)}. "
            f"Refusing to run a partial ensemble."
        )

    master_df = pd.read_csv(live_csv_path)

    # Schema-drift guard: buffer must carry every column the checkpoints expect.
    probe = TemporalFusionTransformer.load_from_checkpoint(checkpoint_paths[0])
    required_cols = [c for c in (probe.dataset_parameters.get("time_varying_known_reals") or [])
                     + (probe.dataset_parameters.get("time_varying_unknown_reals") or [])]
    missing = [c for c in required_cols if c not in master_df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] Live buffer missing columns required by checkpoint: {missing}. "
            f"Re-run build_data.py to regenerate master_df.csv."
        )
    del probe

    # Freshness guard
    last_date = pd.to_datetime(master_df["Date"].max())
    days_stale = (pd.Timestamp.now().normalize() - last_date).days
    if days_stale > 5:
        raise ValueError(
            f"[ERROR] Live buffer is stale ({days_stale} days). "
            f"Re-run deployment.py to refresh master_df.csv."
        )

    q01_vals, q50_vals, q99_vals = [], [], []
    for ckpt in checkpoint_paths:
        q01, q50, q99 = _infer_quantiles_one_checkpoint(ckpt, master_df, target_ticker)
        q01_vals.append(q01)
        q50_vals.append(q50)
        q99_vals.append(q99)

    # Ensemble = mean of the seed quantiles (pre-determined rule).
    ens_q01 = float(np.mean(q01_vals))
    ens_q50 = float(np.mean(q50_vals))
    ens_q99 = float(np.mean(q99_vals))

    print("\n" + "=" * 70)
    print(f"  TARGET ASSET:                       {target_ticker}")
    print(f"  Forecast Horizon:                   Next Trading Session (t+1)")
    print(f"  Ensemble size:                      {len(checkpoint_paths)} seeds")
    print(f"  Seed q0.01 values:                  {[f'{v:.4f}' for v in q01_vals]}")
    print(f"  -> FINAL ENSEMBLE 99% VaR BOUND:    {ens_q01:.4f}%")
    print(f"  (median q0.50 = {ens_q50:.4f}%, upside q0.99 = {ens_q99:.4f}%)")
    print("=" * 70 + "\n")

    return {
        "ticker": target_ticker,
        "final_var_99": float(ens_q01),
        "seed_q01": q01_vals,
        "ensemble_size": len(checkpoint_paths),
    }


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

    # Freshness guard: the buffer must end on a recent trading day, else the
    # forecast is stale. (Allow a small grace for weekends/holidays.)
    last_date = pd.to_datetime(master_df["Date"].max())
    days_stale = (pd.Timestamp.now().normalize() - last_date).days
    if days_stale > 5:
        raise ValueError(
            f"[ERROR] Live buffer is stale ({days_stale} days). "
            f"Re-run build_data.py (or deployment.py) to refresh master_df.csv."
        )

    target_series = master_df[master_df["ticker"] == target_ticker].sort_values(by="time_idx").copy()

    if len(target_series) < config.LOOKBACK_DAYS:
        print(f"[WARNING] History depth ({len(target_series)}) < {config.LOOKBACK_DAYS} days. Using full available history.")
        history_window = target_series.copy()
    else:
        history_window = target_series.tail(config.LOOKBACK_DAYS).copy()

    # 3. Online Point-in-Time GJR-GARCH(1,1) Skew-T estimation for the
    #    econometric PRIOR (input feature), NOT an output override.
    # FIX 12.3 (train/serve consistency note): the live GARCH refit uses the
    # EXACT same model specification (mean='Constant', vol='Garch', p=1, o=1,
    # q=1, dist='skewt') as the build_data.py recursion. While rolling live
    # inference inherently re-fits on the trailing LOOKBACK_DAYS window rather
    # than the expanding build-time recursion, the structural prior generation
    # mechanism is identical, so the served GARCH_sigma prior is generated by
    # the same model family/formula as in training.
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
        pred_values, index_df = unpack_predictions(
            tft.predict(inference_dataloader, mode="quantiles", return_index=True)
        )

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
    from tft_model import select_seed_checkpoints
    ckpts = select_seed_checkpoints()
    if ckpts:
        run_live_ensemble_inference(ckpts, live_csv_path="master_df.csv", target_ticker="NIFTY50")
    else:
        print("[INFO] No trained model checkpoints found. Run main.py first.")
