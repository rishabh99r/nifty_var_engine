# tft_model.py
import os
import glob
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore", category=UserWarning)

DRIVE_DIR = "/content/drive/MyDrive/GARCH_TFT_Results"


class EpochHeartbeat(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            val_loss = trainer.callback_metrics.get('val_loss', 0.0)
            print(f"    >>> [Heartbeat] Epoch {trainer.current_epoch:02d}/{trainer.max_epochs} | Val Loss: {val_loss:.4f}", flush=True)


def build_datasets(df, encoder_length=21, backtest_days=500, val_days=250):
    """
    Builds training, validation, and testing TimeSeriesDataSets for the multi-series panel.
    Enforces a clean RangeIndex across all splits to satisfy PyTorch Forecasting integrity checks.
    """
    if backtest_days is None:
        backtest_days = 500
    if encoder_length is None or isinstance(encoder_length, bool):
        encoder_length = 21

    df = df.copy()

    # Enforce strictly unique integer RangeIndex
    if not df.index.is_unique or isinstance(df.index, pd.DatetimeIndex):
        if 'Date' not in df.columns:
            df = df.reset_index()
        else:
            df = df.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df['ticker'] = df['ticker'].astype(str)

    max_idx = df["time_idx"].max()
    test_cutoff = max_idx - backtest_days
    val_cutoff = test_cutoff - val_days

    candidate_known = ["time_idx", "GARCH_sigma", "US_VIX_Diff", "India_VIX_Diff", "FII_Net_Flow_Z"]
    known_reals = [col for col in candidate_known if col in df.columns]

    candidate_unknown = [
        "Log_Ret", "GARCH_resid", "GK_Vol",
        "Log_Ret_Lag1", "Log_Ret_Lag2", "TRMI_Fear"
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
        allow_missing_timesteps=True
    )

    val_df = df[(df["time_idx"] > val_cutoff - encoder_length) & (df["time_idx"] <= test_cutoff)].reset_index(drop=True)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_df, predict=False, stop_randomization=True)

    test_df = df[df["time_idx"] > test_cutoff - encoder_length].reset_index(drop=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df, predict=False, stop_randomization=True)

    return training_dataset, validation_dataset, test_dataset, test_cutoff


def train_tft(df, hidden_size=64, dropout=0.30, learning_rate=0.001552, seed=42,
              max_epochs=80, enable_progress_bar=True, pruning_callback=None,
              encoder_length=21, backtest_days=500):
    """
    Trains TFT with the champion architecture and saves top checkpoints to persistent storage.
    """
    if backtest_days is None:
        backtest_days = 500
    if encoder_length is None or isinstance(encoder_length, bool):
        encoder_length = 21

    pl.seed_everything(seed, workers=True)
    training_dataset, validation_dataset, test_dataset, test_cutoff = build_datasets(
        df, encoder_length=encoder_length, backtest_days=backtest_days
    )

    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=64, num_workers=0, pin_memory=False)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=64, num_workers=0, pin_memory=False)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0, pin_memory=False)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=4,
        dropout=dropout,
        hidden_continuous_size=max(4, hidden_size // 2),
        output_size=3,
        loss=QuantileLoss(quantiles=[0.01, 0.5, 0.99]),
        optimizer="adam",
        reduce_on_plateau_patience=4
    )

    # Checkpoint configuration: write to Google Drive if mounted, otherwise local
    checkpoint_dir = DRIVE_DIR if os.path.exists("/content/drive/MyDrive") else "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="champion_tft_{epoch:02d}_{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, verbose=False, mode="min"),
        checkpoint_callback,
        EpochHeartbeat()
    ]
    if pruning_callback is not None:
        callbacks.append(pruning_callback)

    precision_mode = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision=precision_mode,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        logger=False
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n[CHECKPOINT] Loading optimal model weights from: {best_model_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    val_loss = trainer.callback_metrics.get("val_loss")
    best_score = val_loss.item() if val_loss is not None else 0.0

    return tft, trainer, best_score, val_dataloader, test_dataloader


def generate_and_save_predictions(tft, test_dataloader, df,
                                  output_csv="test_tft_predictions.csv",
                                  panel_csv="test_tft_predictions_panel.csv"):
    """
    Generates out-of-sample quantile forecasts across the panel, applies the Stage 1
    GJR-GARCH asymmetric circuit breaker per ticker, and exports both NIFTY50 and full-panel files.
    """
    print("\n[INFERENCE] Generating out-of-sample multi-quantile tail forecasts across panel...")

    res = tft.predict(test_dataloader, mode="quantiles", return_index=True)

    if hasattr(res, 'output') and hasattr(res, 'index'):
        pred_values = res.output.cpu().numpy()
        pred_index = res.index
    else:
        pred_values = res[0].cpu().numpy()
        pred_index = res[1]

    pred_df = pred_index.copy()
    pred_df["TFT_VaR_99_Raw"] = pred_values[:, 0, 0]  # q = 0.01
    pred_df["TFT_Median"] = pred_values[:, 0, 1]      # q = 0.50
    pred_df["TFT_VaR_Upside"] = pred_values[:, 0, 2]  # q = 0.99

    # Join metadata across all tickers
    panel_meta = df[["time_idx", "ticker", "Date", "Log_Ret", "GARCH_VaR_99"]].copy()
    merged_panel = pred_df.merge(panel_meta, on=["time_idx", "ticker"], how="inner")

    # Enforce Asymmetric Model Risk Circuit Breaker: min(Raw_TFT, GARCH_Floor)
    merged_panel["TFT_VaR_99"] = np.minimum(merged_panel["TFT_VaR_99_Raw"], merged_panel["GARCH_VaR_99"])
    merged_panel["Date"] = pd.to_datetime(merged_panel["Date"]).dt.strftime('%Y-%m-%d')
    merged_panel = merged_panel.sort_values(by=["Date", "ticker"]).reset_index(drop=True)

    # 1. Save multi-ticker panel predictions
    merged_panel.to_csv(panel_csv, index=False)
    print(f"[SUCCESS] Exported full panel predictions ({len(merged_panel)} rows) to {panel_csv}")

    # 2. Isolate NIFTY50 for primary backtesting and legacy plotting modules
    nifty_merged = merged_panel[merged_panel["ticker"] == "NIFTY50"].sort_values(by="Date").reset_index(drop=True)
    nifty_merged.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Exported NIFTY50 predictions ({len(nifty_merged)} rows) to {output_csv}")

    # 3. Mirror directly to Google Drive if mounted
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        shutil.copy(output_csv, os.path.join(DRIVE_DIR, output_csv))
        shutil.copy(panel_csv, os.path.join(DRIVE_DIR, panel_csv))
        print(f"[PERSISTENCE] Successfully mirrored prediction files to {DRIVE_DIR}")

    return nifty_merged


def run_optimization_and_train(master_file="master_df.csv", n_trials=0, skip_optuna=True):
    """
    Pipeline orchestrator: fits the locked champion model directly in ~7 minutes by default,
    or executes an Optuna sweep if skip_optuna=False.
    """
    if not os.path.exists(master_file):
        raise FileNotFoundError(f"Missing {master_file}. Run build_data.py first.")

    df = pd.read_csv(master_file)
    print(f"[INIT] Loaded {master_file} with {len(df)} rows across tickers: {df['ticker'].unique()}")

    # Locked Champion Architecture
    best_params = {
        "hidden_size": 64,
        "dropout": 0.30,
        "learning_rate": 0.001552
    }

    if not skip_optuna and n_trials > 0:
        print(f"\n[OPTUNA] Launching accelerated {n_trials}-trial sweep with Median Pruner...")
        training_dataset, validation_dataset, _, _ = build_datasets(df, encoder_length=21, backtest_days=500, val_days=250)
        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        def objective(trial):
            hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
            dropout = trial.suggest_float("dropout", 0.10, 0.30, step=0.05)
            learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)

            tft = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                attention_head_size=4,
                dropout=dropout,
                hidden_continuous_size=max(4, hidden_size // 2),
                output_size=3,
                loss=QuantileLoss(quantiles=[0.01, 0.5, 0.99]),
                optimizer="adam"
            )

            early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min")
            trainer = pl.Trainer(
                max_epochs=20,
                accelerator="auto",
                devices="auto",
                precision="16-mixed" if torch.cuda.is_available() else "32-true",
                gradient_clip_val=0.1,
                callbacks=[early_stop],
                enable_progress_bar=False,
                logger=False
            )

            trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            val_loss = trainer.callback_metrics.get("val_loss")
            return val_loss.item() if val_loss is not None else float("inf")

        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=4)
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=n_trials)

        print(f"\n[OPTUNA] Optimization complete! Best Val Loss: {study.best_value:.5f}")
        best_params = study.best_params

    print(f"\n[TRAIN] Fitting champion architecture to full convergence (max 80 epochs)...")
    print(f"        Parameters: {best_params}")

    # Correct dataloader unpacking order: (tft, trainer, score, val_dl, test_dl)
    champion_tft, trainer, score, val_dataloader, test_dataloader = train_tft(
        df,
        hidden_size=best_params["hidden_size"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        max_epochs=80,
        encoder_length=21,
        seed=42
    )

    # Generate predictions on the true test dataloader
    generate_and_save_predictions(champion_tft, test_dataloader, df)


if __name__ == "__main__":
    run_optimization_and_train(master_file="master_df.csv", skip_optuna=True)
