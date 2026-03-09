# tft_model.py
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def train_tft(df, hidden_size, dropout, learning_rate, seed, max_epochs=150, enable_progress_bar=False, pruning_callback=None):
    print(f"\n[TFT] === Initializing Network Engine (Seed: {seed}) ===")

    pl.seed_everything(seed, workers=True)

    target_col = "Log_Ret"

    # NOTE: GARCH_Vol is intentionally excluded.
    # When included, the TFT assigns it excessive weight (path of least resistance),
    # suppressing the contribution of exogenous variables and causing lazy learning.
    unknown_reals = [
        "Log_Ret",
        "GARCH_Resid",
        "VIX_Diff",
        "US_10Y_Diff",
        "DXY_Ret",
        "Crude_Oil_Ret",
        "Global_CPU_Ret"
    ]

    print("[TFT] Slicing data for out-of-sample validation...")
    training_cutoff = df["time_idx"].max() - 250

    print(f"[TFT] Building Training Dataset (time_idx <= {training_cutoff})...")
    training_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["group"],
        min_encoder_length=60,
        max_encoder_length=60,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    print("[TFT] Building Validation Dataset...")
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df, predict=False, stop_randomization=True
    )

    print(f"[TFT] Train samples: {len(training_dataset)} | Val samples: {len(validation_dataset)}")

    # num_workers=0 avoids multiprocessing deadlocks in Colab
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=32, num_workers=0)

    print(f"[TFT] Compiling TFT (hidden={hidden_size}, dropout={dropout}, lr={learning_rate:.4f})...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=4,
        dropout=dropout,
        hidden_continuous_size=hidden_size // 2,
        output_size=3,  # quantiles: [0.01, 0.50, 0.99]
        loss=QuantileLoss(quantiles=[0.01, 0.5, 0.99]),
        optimizer="adam"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    callbacks_list = [early_stop_callback]
    if pruning_callback is not None:
        callbacks_list.append(pruning_callback)

    print("[TFT] Starting training...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=0.1,
        callbacks=callbacks_list,
        deterministic=False,
        enable_progress_bar=enable_progress_bar,
        num_sanity_val_steps=0,
        logger=False
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_val_loss = early_stop_callback.best_score.item()
    stopped = early_stop_callback.stopped_epoch
    optimal_epochs = stopped - early_stop_callback.patience if stopped > 0 else max_epochs

    print(f"[TFT] Training complete. Best Val Loss: {best_val_loss:.4f} | Optimal Epoch: {optimal_epochs}")

    # Returns 4 values. val_dataloader is needed by generate_predictions() in main.py.
    return tft, trainer, best_val_loss, val_dataloader
