# hpo_optuna.py
# =============================================================================
# Lightweight Optuna Challenger Tournament (Experiments B & C of the review).
#
# A thin wrapper over the existing train_tft() pipeline -- it searches the
# expanded Reviewer-3 grid (including encoder_length) with Optuna's aggressive
# pruning to kill unpromising configurations early and save Colab compute.
#
# Search space:
#   hidden_size     : [32, 48, 64, 96, 128]
#   attention_heads : [1, 2, 4, 8]
#   encoder_length  : [10, 21, 42, 63]
#   dropout         : [0.05, 0.40]
#   learning_rate   : log-uniform [1e-4, 5e-3]
#
# Objective: best VALIDATION loss (checkpoint_callback.best_model_score).
# Pruning: MedianPruner via the PyTorchLightning (Optuna) pruning callback,
# which kills a trial mid-training if it is behind the median of its peers.
#
# Usage:  python hpo_optuna.py --trials 20 --seed 42 --max_epochs 40
# =============================================================================
import argparse
import os

import optuna
import pandas as pd

import config
from tft_model import train_tft


def _make_pruning_callback(trial, monitor="val_loss"):
    """Best-effort Optuna-Lightning pruning callback (name differs across
    optuna-integrations versions)."""
    try:
        from optuna.integration import PyTorchLightningPruningCallback
        return PyTorchLightningPruningCallback(trial, monitor=monitor)
    except Exception:
        pass
    try:
        from optuna.integration.lightning import PyTorchLightningPruningCallback
        return PyTorchLightningPruningCallback(trial, monitor=monitor)
    except Exception:
        print("  [WARN] No Optuna-Lightning pruning callback available; "
              "running without mid-training pruning.")
        return None


def _make_objective(master_df, study_seed, max_epochs):
    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 48, 64, 96, 128])
        attention_heads = trial.suggest_categorical("attention_heads", [1, 2, 4, 8])
        encoder_length = trial.suggest_categorical("encoder_length", [10, 21, 42, 63])
        dropout = trial.suggest_float("dropout", 0.05, 0.40)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

        pruning_callback = _make_pruning_callback(trial)
        try:
            _, _, best_val_loss, _, _ = train_tft(
                df=master_df,
                hidden_size=hidden_size,
                attention_heads=attention_heads,
                encoder_length=encoder_length,
                dropout=dropout,
                learning_rate=learning_rate,
                seed=study_seed,            # fixed base seed for fast HPO
                max_epochs=max_epochs,
                enable_progress_bar=False,
                pruning_callback=pruning_callback,
                logger=False,               # no CSVLogger spam inside HPO
            )
            return float(best_val_loss)
        except Exception as e:
            # A config that crashes (e.g. OOM / invalid shape) is pruned, not fatal.
            raise optuna.TrialPruned(f"Trial failed: {e}")

    return objective


def run_hpo_optuna(trials=20, seed=42, max_epochs=None, timeout=14400):
    if max_epochs is None:
        max_epochs = 40  # shorter budget for the search
    if not os.path.exists("master_df.csv"):
        raise FileNotFoundError("[FATAL] master_df.csv not found. Run build_data.py first.")

    master_df = pd.read_csv("master_df.csv")
    print("=== LAUNCHING REVIEWER-3 OPTUNA CHALLENGER TOURNAMENT ===")
    print(f"  Space: hidden[32..128] heads[1..8] enc[10..63] dropout[0.05..0.40] "
          f"lr[1e-4..5e-3] | trials={trials} | epochs={max_epochs} | seed={seed}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(_make_objective(master_df, seed, max_epochs),
                   n_trials=trials, timeout=timeout, show_progress_bar=True)

    best = study.best_trial
    best_params = {k: v for k, v in best.params.items()}
    best_params["seed"] = seed
    best_params["max_epochs"] = max_epochs
    best_params["target_quantiles"] = list(config.QUANTILES)

    # Persist the winner for specialist_runner / final runs.
    import json
    with open("hpo_optuna_best_trial.json", "w") as f:
        json.dump({"best_value": float(best.value), "best_params": best_params}, f, indent=2)

    # Save the full trials dataframe to CSV.
    df_trials = study.trials_dataframe()
    df_trials.to_csv("hpo_optuna_results.csv", index=False)

    print("\n=== OPTUNA TOURNAMENT COMPLETE ===")
    print(f"  Best Trial: #{best.number}  Val Loss: {best.value:.6f}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"  Saved -> hpo_optuna_best_trial.json | hpo_optuna_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lightweight Optuna HPO for the ECTFT.")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=14400, help="seconds (default 4h)")
    args = ap.parse_args()
    run_hpo_optuna(trials=args.trials, seed=args.seed,
                   max_epochs=args.max_epochs, timeout=args.timeout)
