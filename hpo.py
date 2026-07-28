# hpo.py
import optuna
import time
import warnings
import gc
import os
import torch
from tqdm import tqdm
from tft_model import train_tft

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

def optimize_hyperparameters(df, n_trials=30):
    print("\n[HPO] === Starting Robust Hyperparameter Optimization ===")

    # WIPE OLD DB TO PREVENT TRIAL COUNT LEAKAGE (e.g., Trial 48/30 errors)
    db_path = "tft_nifty_optimization.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"[HPO] Removed stale database '{db_path}'. Starting fresh 30-trial study.")

    # Master progress bar for the entire HPO phase
    pbar = tqdm(total=n_trials, desc="[HPO] Optimizing Architecture", unit="trial", leave=True)

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)

        try:
            # enable_progress_bar=False keeps console clean while tqdm tracks overall progress
            _, _, val_loss, _ = train_tft(
                df=df,
                hidden_size=hidden_size,
                dropout=dropout,
                learning_rate=learning_rate,
                seed=42,
                max_epochs=30,
                enable_progress_bar=False,
                pruning_callback=None
            )

            # Update live progress bar postfix with current and best scores
            best_val = trial.study.best_value if len(trial.study.trials) > 0 and trial.study.best_value is not None else val_loss
            best_val = min(best_val, val_loss)

            pbar.set_postfix({
                "Best Loss": f"{best_val:.4f}",
                "Curr Loss": f"{val_loss:.4f}",
                "Hidden": hidden_size,
                "LR": f"{learning_rate:.4f}"
            })
            pbar.update(1)
            return val_loss

        except Exception as e:
            tqdm.write(f"[HPO] Trial {trial.number+1} FAILED | Error: {e}")
            pbar.update(1)
            return float("inf")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study = optuna.create_study(
        study_name="tft_nifty_optimization",
        storage=f"sqlite:///{db_path}?timeout=60",
        direction="minimize",
        load_if_exists=False
    )

    study.optimize(objective, n_trials=n_trials)
    pbar.close()

    print(f"\n[HPO] Optimization Complete | Best Val Loss: {study.best_value:.4f}")
    print(f"[HPO] Optimal Parameters: {study.best_params}")
    return study.best_params
