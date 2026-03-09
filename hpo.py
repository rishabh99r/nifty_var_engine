# hpo.py
import optuna
import time
import warnings
from optuna_integration import PyTorchLightningPruningCallback  # requires: pip install optuna-integration
from tft_model import train_tft

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

def optimize_hyperparameters(df, n_trials=30):
    print("\n[HPO] === Starting Robust Hyperparameter Optimization ===")
    print(f"[HPO] {n_trials} trials | Early pruning enabled | Fault-tolerant SQLite storage")
    print(f"[HPO] NOTE: Each trial trains up to 50 epochs. Expect 2-4 hrs on Colab CPU/GPU.")
    print(f"[HPO] Per-trial timing is shown below — it will NOT appear frozen.\n")

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)

        print(f"[HPO] Trial {trial.number+1}/{n_trials} started | hidden={hidden_size}, dropout={dropout:.1f}, lr={learning_rate:.5f}")
        t0 = time.time()

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

        try:
            # train_tft returns 4 values: tft, trainer, val_loss, val_dataloader
            _, _, val_loss, _ = train_tft(
                df=df,
                hidden_size=hidden_size,
                dropout=dropout,
                learning_rate=learning_rate,
                seed=42,
                max_epochs=50,
                enable_progress_bar=True,
                pruning_callback=pruning_callback
            )
            elapsed = time.time() - t0
            print(f"[HPO] Trial {trial.number+1}/{n_trials} done    | Val Loss: {val_loss:.4f} | Elapsed: {elapsed:.0f}s")
            return val_loss

        except optuna.exceptions.TrialPruned:
            elapsed = time.time() - t0
            print(f"[HPO] Trial {trial.number+1}/{n_trials} PRUNED  | Elapsed: {elapsed:.0f}s")
            raise

        except Exception as e:
            print(f"[HPO] Trial {trial.number+1}/{n_trials} FAILED  | Error: {e}")
            return float("inf")

    study_name = "tft_nifty_optimization"
    storage_name = f"sqlite:///{study_name}.db?timeout=60"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study.optimize(objective, n_trials=n_trials)

    print("\n[HPO] Optimization Complete!")
    print(f"[HPO] Best validation loss: {study.best_value:.4f}")
    print(f"[HPO] Best parameters:      {study.best_params}")

    return study.best_params
