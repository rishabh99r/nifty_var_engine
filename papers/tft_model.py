# tft_model.py
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class EpochHeartbeat(pl.Callback):
    """
    Forces a live heartbeat print to stdout every 5 epochs to override
    Google Colab subshell buffering during non-interactive execution.
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            val_loss = trainer.callback_metrics.get('val_loss', 0.0)
            print(f"   >>> [Heartbeat] Epoch {trainer.current_epoch:02d}/{trainer.max_epochs} | Val Loss: {val_loss:.4f}", flush=True)

def build_datasets(df, encoder_length=60):
    max_idx = df["time_idx"].max()

    # 250 days for Test, 250 days for Validation to eliminate HPO data leakage
    test_cutoff = max_idx - 250
    val_cutoff = test_cutoff - 250

    target_col = "Log_Ret"

    # NOTE: GARCH_Vol is intentionally excluded to prevent attention head laziness.
    unknown_reals = [
        "Log_Ret",
        "GARCH_Resid",
        "VIX_Diff",
        "US_10Y_Diff",
        "DXY_Ret",
        "Crude_Oil_Ret",
        "Global_CPU_Ret"
    ]

    # 1. Train Dataset (Historical training data up to val_cutoff)
    train_df = df[df["time_idx"] <= val_cutoff]
    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["group"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
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

    # 2. Validation Dataset (Includes encoder_length buffer before val_cutoff for windowing)
    val_df = df[(df["time_idx"] > val_cutoff - encoder_length) & (df["time_idx"] <= test_cutoff)]
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=False, stop_randomization=True
    )

    # 3. Test Dataset (Includes encoder_length buffer before test_cutoff)
    test_df = df[df["time_idx"] > test_cutoff - encoder_length]
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, test_df, predict=False, stop_randomization=True
    )

    return training_dataset, validation_dataset, test_dataset

def train_tft(df, hidden_size, dropout, learning_rate, seed, max_epochs=150, enable_progress_bar=True, pruning_callback=None):
    print(f"\n[TFT] === Initializing Network Engine (Seed: {seed}) ===")
    pl.seed_everything(seed, workers=True)

    training_dataset, validation_dataset, test_dataset = build_datasets(df)

    # pin_memory=False and num_workers=0 prevent Colab CUDA and multiprocessing deadlocks
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=32, num_workers=0, pin_memory=False)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=32, num_workers=0, pin_memory=False)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=32, num_workers=0, pin_memory=False)

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

    # Inject heartbeat callback directly into the training loop
    callbacks_list = [early_stop_callback, EpochHeartbeat()]
    if pruning_callback is not None:
        callbacks_list.append(pruning_callback)

    # Clean trainer setup: auto device resolution prevents DDP container hangs
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=0.1,
        callbacks=callbacks_list,
        enable_progress_bar=enable_progress_bar,
        logger=False
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    best_val_loss = early_stop_callback.best_score.item()

    return tft, trainer, best_val_loss, test_dataloader
