# tft_model.py
# =============================================================================
# Temporal Fusion Transformer training & inference for the Nifty VaR pipeline.
#
# MODEL FRAMING (revised):
#   This is an "Econometrically-Conditioned TFT". GARCH is NOT an output-level
#   combination; rather GARCH_sigma is passed as an input prior through the
#   Variable Selection Network. The reported/backtested VaR is the RAW TFT
#   quantile (no GARCH floor is applied). This guarantees that the validated
#   model == the deployed model.
#
# Leakage controls:
#   - Log_Ret_Feature (a copy of the return series, created in build_data.py) is
#     listed in time_varying_unknown_reals. In PyTorch Forecasting, unknown reals
#     feed the ENCODER only (observed up to time t) and are hidden from the
#     decoder at t+1. Listing the return history as a FEATURE (rather than the
#     raw target) forces it through the Variable Selection Network so its
#     importance is attributed alongside the macro/econometric priors -- without
#     duplicating the target column. Log_Ret itself stays reserved as the target.
#   - Temporal splits keep validation/test strictly after training.
# =============================================================================
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

import config

warnings.filterwarnings("ignore", category=UserWarning)


class EpochHeartbeat(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            val_loss = trainer.callback_metrics.get("val_loss", 0.0)
            print(f"    >>> [Heartbeat] Epoch {trainer.current_epoch:02d}/{trainer.max_epochs} | Val Loss: {val_loss:.4f}", flush=True)


def build_datasets(df, encoder_length=None, backtest_days=None, val_days=None):
    """
    Builds training, validation, and testing TimeSeriesDataSets for the
    multi-series panel with clean temporal separation.
    """
    if backtest_days is None:
        backtest_days = config.BACKTEST_DAYS
    if encoder_length is None or isinstance(encoder_length, bool):
        encoder_length = config.ENCODER_LENGTH
    if val_days is None:
        val_days = config.VAL_DAYS

    df = df.copy()

    # Enforce strictly unique integer RangeIndex
    df = df.reset_index(drop=True)
    df["ticker"] = df["ticker"].astype(str)

    max_idx = df["time_idx"].max()
    test_cutoff = max_idx - backtest_days
    val_cutoff = test_cutoff - val_days

    # --- Feature roles -----------------------------------------------------
    # Known-at-forecast-time reals (observable using only data up to t-1):
    candidate_known = [
        "time_idx",
        "GARCH_sigma",      # econometric volatility prior (PIT, F_{t-1}-measurable)
        "US_VIX_Diff",      # lagged log-diff of US VIX
        "India_VIX_Diff",   # lagged log-diff of India VIX (or RV proxy)
    ]
    known_reals = [col for col in candidate_known if col in df.columns]

    # Unknown reals (contemporaneous with target). Log_Ret_Feature (a copy of
    # the return series created in build_data.py) is listed so the autoregressive
    # history passes through the VSN for attribution, while PyTorch Forecasting
    # feeds it to the ENCODER only (observed up to time t) and hides it from the
    # decoder at t+1 -- restoring momentum with NO look-ahead. Log_Ret itself
    # remains the target and is NOT duplicated as a feature.
    # FIX 14.1: GARCH_resid is deliberately EXCLUDED -- it is a scaled copy of
    # the same return innovation carried by Log_Ret_Feature, so including both
    # would feed two near-collinear channels into the VSN and fragment the
    # attribution. The econometric prior is already represented by GARCH_sigma.
    candidate_unknown = [
        "Log_Ret_Feature",  # return history as an unknown real -> encoder + VSN
        "GK_Vol",
    ]
    unknown_reals = [col for col in candidate_unknown if col in df.columns]

    train_df = df[df["time_idx"] <= val_cutoff].reset_index(drop=True)

    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="Log_Ret",
        group_ids=["ticker"],
        static_categoricals=["ticker"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_known_categoricals=[],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    val_df = df[(df["time_idx"] > val_cutoff - encoder_length) & (df["time_idx"] <= test_cutoff)].reset_index(drop=True)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_df, predict=False, stop_randomization=True)

    test_df = df[df["time_idx"] > test_cutoff - encoder_length].reset_index(drop=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df, predict=False, stop_randomization=True)

    return training_dataset, validation_dataset, test_dataset, test_cutoff


def train_tft(df, hidden_size=None, dropout=None, learning_rate=None, seed=42,
              max_epochs=None, enable_progress_bar=True, pruning_callback=None,
              encoder_length=None, backtest_days=None):
    """Trains the Econometrically-Conditioned TFT with the committed champion spec."""
    if hidden_size is None:
        hidden_size = config.HIDDEN_SIZE
    if dropout is None:
        dropout = config.DROPOUT
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if max_epochs is None:
        max_epochs = config.MAX_EPOCHS
    if encoder_length is None or isinstance(encoder_length, bool):
        encoder_length = config.ENCODER_LENGTH
    if backtest_days is None:
        backtest_days = config.BACKTEST_DAYS

    pl.seed_everything(seed, workers=True)

    training_dataset, validation_dataset, test_dataset, test_cutoff = build_datasets(
        df, encoder_length=encoder_length, backtest_days=backtest_days
    )

    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=False)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=False)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=False)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=config.ATTENTION_HEADS,
        dropout=dropout,
        hidden_continuous_size=config.HIDDEN_CONTINUOUS_SIZE,
        output_size=config.OUTPUT_SIZE,
        loss=QuantileLoss(quantiles=config.QUANTILES),
        optimizer="adam",
        reduce_on_plateau_patience=config.REDUCE_ON_PLATEAU_PATIENCE,
    )

    checkpoint_dir = config.OUTPUT_DIR if os.path.exists("/content/drive/MyDrive") else "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"ectft_seed{seed}_" + "{epoch:02d}_{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=config.EARLY_STOP_PATIENCE, verbose=False, mode="min"),
        checkpoint_callback,
        EpochHeartbeat(),
    ]
    if pruning_callback is not None:
        callbacks.append(pruning_callback)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="32-true",
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        logger=False,
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n[CHECKPOINT] Loading optimal model weights from: {best_model_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    val_loss = trainer.callback_metrics.get("val_loss")
    best_score = val_loss.item() if val_loss is not None else 0.0

    return tft, trainer, best_score, val_dataloader, test_dataloader


def select_median_checkpoint(median_seed=None):
    """
    Deterministically selects the deployment checkpoint.

    - If median_seed is provided, returns the checkpoint matching
      '*seed{median_seed}*' if found (local then Drive).
    - Otherwise reads config.MEDIAN_SEED_FILE if present, else falls back to
      a stable sort of all champion checkpoints and picks the middle one
      (never filesystem-arbitrary).
    Returns the checkpoint path or None.
    """
    import glob

    if median_seed is None:
        try:
            with open(config.MEDIAN_SEED_FILE, "r") as f:
                median_seed = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            median_seed = None

    if median_seed is not None:
        for base in (".", config.OUTPUT_DIR):
            matches = sorted(glob.glob(os.path.join(base, f"*seed{median_seed}*.ckpt")))
            if matches:
                return matches[0]

    # Fallback: stable sort of all champion checkpoints, pick the middle.
    candidates = sorted(glob.glob("*.ckpt") + glob.glob(os.path.join(config.OUTPUT_DIR, "*.ckpt")))
    if candidates:
        return candidates[len(candidates) // 2]
    return None


def generate_and_save_predictions(tft, test_dataloader, df, seed,
                                  output_csv=None, panel_csv=None):
    """
    Generates out-of-sample quantile forecasts across the panel.
    The exported TFT_VaR_99 is the RAW, unconstrained TFT quantile -- NO GARCH
    floor is applied, guaranteeing validated == deployed.
    """
    if output_csv is None:
        output_csv = f"test_tft_predictions_seed_{seed}.csv"
    if panel_csv is None:
        panel_csv = f"test_tft_predictions_panel_seed_{seed}.csv"

    print("\n[INFERENCE] Generating out-of-sample multi-quantile tail forecasts across panel...")

    res = tft.predict(test_dataloader, mode="quantiles", return_index=True)

    if hasattr(res, "output") and hasattr(res, "index"):
        pred_values = res.output.cpu().numpy()
        pred_index = res.index
    else:
        pred_values = res[0].cpu().numpy()
        pred_index = res[1]

    pred_df = pred_index.copy()
    pred_df["TFT_VaR_99_Raw"] = pred_values[:, 0, 0]  # q = 0.01
    pred_df["TFT_Median"] = pred_values[:, 0, 1]      # q = 0.50
    pred_df["TFT_VaR_Upside"] = pred_values[:, 0, 2]  # q = 0.99

    panel_meta = df[["time_idx", "ticker", "Date", "Log_Ret", "GARCH_VaR_99", "GARCH_sigma"]].copy()
    merged_panel = pred_df.merge(panel_meta, on=["time_idx", "ticker"], how="inner")

    # FIX 12.4: mathematically guarantee no test day was silently dropped by the
    # inner merge (e.g. due to a NaN GARCH column in master_df on a test day).
    expected_len = len(pred_df)
    assert len(merged_panel) == expected_len, (
        f"[FATAL] Merge dropped {expected_len - len(merged_panel)} rows. "
        f"Check master_df for NaNs in GARCH columns on the out-of-sample horizon."
    )

    # The reported VaR is the RAW TFT quantile (validated == deployed).
    merged_panel["TFT_VaR_99"] = merged_panel["TFT_VaR_99_Raw"]
    merged_panel["Date"] = pd.to_datetime(merged_panel["Date"]).dt.strftime("%Y-%m-%d")
    merged_panel = merged_panel.sort_values(by=["Date", "ticker"]).reset_index(drop=True)

    merged_panel.to_csv(panel_csv, index=False)
    print(f"[SUCCESS] Exported full panel predictions ({len(merged_panel)} rows) to {panel_csv}")

    nifty_merged = merged_panel[merged_panel["ticker"] == "NIFTY50"].sort_values(by="Date").reset_index(drop=True)
    nifty_merged.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Exported NIFTY50 predictions ({len(nifty_merged)} rows) to {output_csv}")

    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        shutil.copy(output_csv, os.path.join(config.OUTPUT_DIR, output_csv))
        shutil.copy(panel_csv, os.path.join(config.OUTPUT_DIR, panel_csv))
        print(f"[PERSISTENCE] Successfully mirrored prediction files to {config.OUTPUT_DIR}")

    return nifty_merged
