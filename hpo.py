# hpo.py
import optuna
import time
import warnings
import gc
import torch
from tft_model import train_tft

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

def optimize_hyperparameters(df, n_trials=30):
    print("\n[HPO] === Starting Robust Hyperparameter Optimization ===")

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)

        t0 = time.time()

        try:
            # Set max_epochs to 30 for faster HPO sweeps
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
            elapsed = time.time() - t0
            print(f"[HPO] Trial {trial.number+1:02d}/{n_trials} | hidden={hidden_size:<3d} | dropout={dropout:.1f} | lr={learning_rate:.5f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.0f}s")
            return val_loss

        except Exception as e:
            print(f"[HPO] Trial {trial.number+1:02d}/{n_trials} FAILED | Error: {e}")
            return float("inf")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study = optuna.create_study(
        study_name="tft_nifty_optimization",
        storage="sqlite:///tft_nifty_optimization.db?timeout=60",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study.optimize(objective, n_trials=n_trials)
    print(f"\n[HPO] Best Val Loss: {study.best_value:.4f} | Params: {study.best_params}")
    return study.best_params
