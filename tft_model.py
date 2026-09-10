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
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

import config

warnings.filterwarnings("ignore", category=UserWarning)


class EpochHeartbeat(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            val_loss = trainer.callback_metrics.get("val_loss", 0.0)
            print(f"    >>> [Heartbeat] Epoch {trainer.current_epoch:02d}/{trainer.max_epochs} | Val Loss: {val_loss:.4f}", flush=True)


# ----------------------------------------------------------------------------
# CANONICAL FEATURE SPEC (FIX Phase-3 / Reviewer): time_idx is REMOVED from the
# known-feature list. Absolute calendar position is an unnecessary source of
# temporal-shortcut learning and its presence made the canonical model differ
# from the ablation "Full ECTFT" arm. PyTorch Forecasting still injects the
# bounded, window-relative positional columns (relative_time_idx via
# add_relative_time_idx=True and encoder_length via add_encoder_length=True),
# which are legitimate within-sequence positional signals, identical across
# every ablation arm. The canonical model is now STRUCTURALLY IDENTICAL to the
# tournament's Full ECTFT arm.
# ----------------------------------------------------------------------------
DEFAULT_KNOWN_FEATURES = [
    "GARCH_sigma",      # econometric volatility prior (PIT, F_{t-1}-measurable)
    "US_VIX_Diff",      # lagged log-diff of US VIX (cross-border)
    "India_VIX_Diff",   # lagged log-diff of India VIX (or RV proxy, domestic)
]
DEFAULT_UNKNOWN_FEATURES = [
    "Log_Ret_Feature",  # return history as an unknown real -> encoder + VSN
    "GK_Vol",           # Garman-Klass intraday-range volatility
]

# Documented fallback: when the real India VIX history is too short,
# build_data.py fills India_VIX_Diff with a Domestic_RV_Proxy and the column is
# ALWAYS present. If it is ever missing, warn-and-continue (known fallback path);
# any OTHER requested feature that is absent is a hard schema error.
_INDIA_VIX_FALLBACK_COLS = {"India_VIX_Diff"}


def _resolve_feature_list(requested, available_columns, role_label):
    """
    Filters a requested feature list against the columns actually present in df.

    Warn-and-continue ONLY for the documented India VIX RV-proxy fallback;
    raise ValueError for any other requested-but-missing feature (no silent
    drops -- a silent drop would make two tournament arms accidentally equal).
    """
    missing = [c for c in requested if c not in available_columns]
    for col in missing:
        if col in _INDIA_VIX_FALLBACK_COLS:
            print(f"  [WARN] Requested {role_label} feature '{col}' not in df "
                  f"(India VIX fallback path). Continuing without it.")
        else:
            raise ValueError(
                f"[FATAL] Requested {role_label} feature '{col}' is not a column "
                f"of the input df. Available: {sorted(available_columns)}"
            )
    return [c for c in requested if c in available_columns]


def build_datasets(df, encoder_length=None, backtest_days=None, val_days=None,
                   known_features=None, unknown_features=None):
    """
    Builds training, validation, and testing TimeSeriesDataSets for the
    multi-series panel with clean temporal separation.

    known_features / unknown_features may be passed explicitly (used by the
    ablation tournament); when None they default to the canonical ECTFT spec
    (DEFAULT_KNOWN_FEATURES / DEFAULT_UNKNOWN_FEATURES) -- WITHOUT time_idx.
    """
    if known_features is None:
        known_features = list(DEFAULT_KNOWN_FEATURES)
    if unknown_features is None:
        unknown_features = list(DEFAULT_UNKNOWN_FEATURES)
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
    known_reals = _resolve_feature_list(known_features, df.columns, "known")

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
    unknown_reals = _resolve_feature_list(unknown_features, df.columns, "unknown")

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
              encoder_length=None, backtest_days=None,
              known_features=None, unknown_features=None,
              attention_heads=None, target_quantiles=None, logger=None):
    """Trains the Econometrically-Conditioned TFT with the committed champion spec.

    known_features / unknown_features override the canonical feature lists
    (used by the ablation tournament). None -> canonical ECTFT spec (no
    time_idx). attention_heads / target_quantiles allow HPO and the q0.01-
    specialist experiment to override the architecture/output (RD pass).
    A per-seed experiment_manifest.json entry is written so the deployed model
    is discovered deterministically, never by lexical globbing.
    """
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
    if known_features is None:
        known_features = list(DEFAULT_KNOWN_FEATURES)
    if unknown_features is None:
        unknown_features = list(DEFAULT_UNKNOWN_FEATURES)
    if attention_heads is None:
        attention_heads = config.ATTENTION_HEADS
    if target_quantiles is None:
        target_quantiles = list(config.QUANTILES)
    # Default logger = CSVLogger (learning diagnostics for Fig 6/7). Callers can
    # pass logger=False to disable (e.g. inside HPO where per-trial logging is
    # wasteful) or a custom logger.
    if logger is None:
        logger = CSVLogger(save_dir="lightning_logs", name=f"tft_run_seed_{seed}")
        try:
            logger.log_hyperparams({"seed": int(seed), "hidden_size": hidden_size,
                                    "attention_heads": attention_heads,
                                    "target_quantiles": target_quantiles})
        except Exception:
            pass

    pl.seed_everything(seed, workers=True)

    training_dataset, validation_dataset, test_dataset, test_cutoff = build_datasets(
        df, encoder_length=encoder_length, backtest_days=backtest_days,
        known_features=known_features, unknown_features=unknown_features,
    )
    resolved_spec = {
        "known": list(getattr(training_dataset, "time_varying_known_reals", known_features)),
        "unknown": list(getattr(training_dataset, "time_varying_unknown_reals", unknown_features)),
        "encoder_length": int(encoder_length),
        "backtest_days": int(backtest_days),
        "dataset_cutoff": getattr(config, "END_DATE", None),
        "attention_heads": int(attention_heads),
        "quantiles": [float(q) for q in target_quantiles],
    }

    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=False)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=False)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=False)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        hidden_continuous_size=max(4, hidden_size // 2),
        output_size=len(target_quantiles),
        loss=QuantileLoss(quantiles=target_quantiles),
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
        logger=logger,
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n[CHECKPOINT] Loading optimal model weights from: {best_model_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # RD1 (best_score fix): the checkpoint callback tracks the TRUE monitored
    # best val_loss (it is always set after fit when best_model_path exists).
    best_score = (
        checkpoint_callback.best_model_score.item()
        if checkpoint_callback.best_model_score is not None
        else float("nan")
    )

    # Stash the resolved feature spec on the model for reproducibility/audit.
    try:
        tft._feature_spec = dict(resolved_spec)
    except Exception:
        pass

    # Deterministic model registry: write/refresh this seed's manifest entry.
    if best_model_path and os.path.exists(best_model_path):
        write_experiment_manifest(
            seed=seed,
            checkpoint_path=best_model_path,
            val_loss=best_score,
            feature_spec=resolved_spec,
            max_epochs=max_epochs,
        )

    return tft, trainer, best_score, val_dataloader, test_dataloader


# ----------------------------------------------------------------------------
# Deterministic checkpoint registry (Phase-3 hardening, Reviewer #23/#24).
# Replaces fragile lexical globbing with an experiment_manifest.json that logs
# git commit, dataset cutoff, feature spec, seed, val_loss and the EXACT
# checkpoint path, so deployment loads the intended model deterministically.
# ----------------------------------------------------------------------------
CHECKPOINT_SEARCH_DIRS = ("checkpoints", ".", config.OUTPUT_DIR)
EXPERIMENT_MANIFEST_FILE = "experiment_manifest.json"


def _git_commit():
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_experiment_manifest(seed, checkpoint_path, val_loss,
                              feature_spec=None, max_epochs=None):
    """
    Upserts an entry for `seed` into experiment_manifest.json.

    Fields: seed, checkpoint_path, val_loss, git_commit, dataset_cutoff,
    feature_spec, encoder/backtest window, timestamp. Deployment reads this
    manifest instead of globbing for checkpoint files.
    """
    import datetime
    import json

    manifest = {}
    if os.path.exists(EXPERIMENT_MANIFEST_FILE):
        try:
            with open(EXPERIMENT_MANIFEST_FILE, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {}

    entry = {
        "seed": int(seed),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "git_commit": _git_commit(),
        "dataset_cutoff": getattr(config, "END_DATE", None),
        "feature_spec": feature_spec or {},
        "max_epochs": int(max_epochs) if max_epochs is not None else int(config.MAX_EPOCHS),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    manifest[f"seed_{seed}"] = entry

    with open(EXPERIMENT_MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Registered seed {seed} checkpoint -> {EXPERIMENT_MANIFEST_FILE}")


def load_experiment_manifest(seeds=None):
    """
    Returns {seed: {entry...}} from experiment_manifest.json for the requested
    seeds whose recorded checkpoint path still exists (empty dict if absent).
    """
    import json

    if seeds is None:
        seeds = config.VALIDATION_SEEDS
    if not os.path.exists(EXPERIMENT_MANIFEST_FILE):
        return {}
    try:
        with open(EXPERIMENT_MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    for seed in seeds:
        entry = manifest.get(f"seed_{seed}")
        if entry and os.path.exists(entry.get("checkpoint_path", "")):
            out[int(seed)] = entry
    return out


def select_seed_checkpoints(seeds=None, prefer_manifest=True):
    """
    Returns the list of available checkpoint paths for the validation seeds
    (used by the ensemble deployment path).

    Manifest-first: if experiment_manifest.json has a valid entry for a seed,
    its recorded checkpoint_path is used verbatim (deterministic). Falls back
    to searching CHECKPOINT_SEARCH_DIRS (including the local "checkpoints/"
    dir, which was previously missed) only when the manifest is missing or the
    recorded file no longer exists.
    """
    import glob

    if seeds is None:
        seeds = config.VALIDATION_SEEDS

    manifest_entries = load_experiment_manifest(seeds) if prefer_manifest else {}
    found = []
    for seed in seeds:
        entry = manifest_entries.get(int(seed))
        if entry is not None:
            found.append(entry["checkpoint_path"])
            continue
        # Manifest miss -> fall back to a strict search across known dirs.
        matches = []
        for base in CHECKPOINT_SEARCH_DIRS:
            matches += sorted(glob.glob(os.path.join(base, f"*seed{seed}*.ckpt")))
        if matches:
            print(f"[WARN] No manifest entry for seed {seed}; "
                  f"falling back to lexical search -> {matches[0]}")
            found.append(matches[0])
    return found


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

    # RD2: dynamic quantile width -- the q0.01-SPECIALIST model outputs only
    # one quantile; the aggregate model outputs three.
    n_q = pred_values.shape[2]
    if n_q >= 3:
        pred_df["TFT_Median"] = pred_values[:, 0, 1]      # q = 0.50
        pred_df["TFT_VaR_Upside"] = pred_values[:, 0, 2]  # q = 0.99
    else:
        pred_df["TFT_Median"] = np.nan
        pred_df["TFT_VaR_Upside"] = np.nan

    # FIX (Round 23): quantile-crossing audit. Quantile loss does NOT enforce
    # q0.01 <= q0.50 <= q0.99; verify the hierarchy holds for every forecast.
    # (Skipped for the single-quantile specialist -- there is no hierarchy.)
    if n_q >= 3:
        cross_lo = int(np.sum(pred_values[:, 0, 0] > pred_values[:, 0, 1]))
        cross_hi = int(np.sum(pred_values[:, 0, 1] > pred_values[:, 0, 2]))
        total_cross = cross_lo + cross_hi
        print(f"[AUDIT] Quantile crossing violations: {total_cross} / {len(pred_values)} "
              f"(q01>q50: {cross_lo}, q50>q99: {cross_hi})")
        if total_cross > 0:
            print("[WARNING] Neural network predicted inverted quantiles on some rows.")
    else:
        print("[AUDIT] Single-quantile (q0.01-specialist) model: crossing audit skipped.")

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
