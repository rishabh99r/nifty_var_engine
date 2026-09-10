# hpo_runner.py
# =============================================================================
# RD3: Optuna Challenger Tournament -- the Reviewer-3 hyperparameter search.
#
# Searches the exact space Reviewer 3 demanded to find the "absolute best" TFT:
#   hidden_size     : [32, 48, 64, 96, 128]
#   attention_heads : [1, 2, 4, 8]
#   encoder_length  : [10, 21, 42, 63]   (tests "GARCH replaces long memory")
#   dropout         : [0.05, 0.40]
#   learning_rate   : Log-uniform [1e-4, 5e-3]
#
# Objective: mean validation pinball loss (trainer's best val_loss via
# checkpoint_callback). HPO is run on the VALIDATION split ONLY -- test
# outcomes are never used for model selection (no selection leakage).
#
# Usage:  python hpo_runner.py --trials 40 --seed 42 --max_epochs 80
# =============================================================================
import argparse
import json
import os

import numpy as np
import pandas as pd
import optuna

import config
from tft_model import train_tft  # feature defaults are the canonical ECTFT spec

HPO_RESULTS_JSON = "hpo_best_trial.json"
HPO_REPORT_TXT = "hpo_trials_report.txt"


def _make_objective(master_df, study_seed, max_epochs):
    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 48, 64, 96, 128])
        attention_heads = trial.suggest_categorical("attention_heads", [1, 2, 4, 8])
        encoder_length = trial.suggest_categorical("encoder_length", [10, 21, 42, 63])
        dropout = trial.suggest_float("dropout", 0.05, 0.40)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

        # Deterministic per-trial seed: base study seed + trial number.
        seed = study_seed + trial.number

        # Disable CSVLogger inside HPO (dozens of tiny log dirs would be
        # wasteful); logger=False keeps the search lean.
        _, _, best_score, _, _ = train_tft(
            df=master_df,
            hidden_size=hidden_size,
            attention_heads=attention_heads,
            encoder_length=encoder_length,
            dropout=dropout,
            learning_rate=learning_rate,
            seed=seed,
            max_epochs=max_epochs,
            enable_progress_bar=False,
            logger=False,
        )
        return float(best_score)

    return objective


def run_hpo(trials=40, study_seed=42, max_epochs=None):
    if max_epochs is None:
        max_epochs = config.MAX_EPOCHS
    if not os.path.exists("master_df.csv"):
        raise FileNotFoundError("[FATAL] master_df.csv not found. Run build_data.py first.")

    master_df = pd.read_csv("master_df.csv")
    print(f"[HPO] Loaded {len(master_df)} rows across {master_df['ticker'].nunique()} tickers.")
    print(f"[HPO] Space: hidden[32..128] heads[1..8] enc[10..63] "
          f"dropout[0.05..0.40] lr[1e-4..5e-3] | trials={trials} | epochs={max_epochs}")

    sampler = optuna.samplers.TPESampler(seed=study_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"ectft_hpo_seed{study_seed}",
    )
    study.optimize(
        _make_objective(master_df, study_seed, max_epochs),
        n_trials=trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_params = {k: v for k, v in best.params.items()}
    best_params["seed"] = study_seed + best.number
    best_params["max_epochs"] = max_epochs
    best_params["target_quantiles"] = list(config.QUANTILES)

    with open(HPO_RESULTS_JSON, "w") as f:
        json.dump({
            "best_value": float(best.value),
            "best_params": best_params,
            "n_trials": len(study.trials),
            "study_seed": study_seed,
        }, f, indent=2)
    print(f"[HPO] Best trial saved -> {HPO_RESULTS_JSON}: {best_params} (val_loss={best.value:.6f})")

    # Human-readable per-trial report.
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append({"trial": t.number, "val_loss": float(t.value), **t.params})
    pd.DataFrame(rows).to_csv(HPO_REPORT_TXT, index=False)
    print(f"[HPO] Trial report saved -> {HPO_REPORT_TXT}")

    return best_params


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optuna HPO for the ECTFT.")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=None)
    args = ap.parse_args()
    run_hpo(trials=args.trials, study_seed=args.seed, max_epochs=args.max_epochs)
