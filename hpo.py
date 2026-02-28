# hpo.py
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from tft_model import train_tft
import warnings

# Suppress Optuna's overly verbose trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

def optimize_hyperparameters(df, n_trials=30):
    print("\n[HPO] === Starting Robust Hyperparameter Optimization ===")
    print(f"[HPO] Executing {n_trials} trials with early pruning enabled.")

    def objective(trial):
        # 1. Define the Search Space
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)

        print(f"[HPO] Trial {trial.number} Started | hidden_size={hidden_size}, dropout={dropout}, lr={learning_rate:.4f}")

        # 2. Setup Pruning Callback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

        # 3. Train and evaluate
        try:
            # We must pass the pruning callback to the PyTorch Lightning trainer
            # To do this cleanly, we need to modify train_tft to accept external callbacks,
            # but for this script, Optuna will handle the exception if it prunes.
            _, _, val_loss = train_tft(
                df=df,
                hidden_size=hidden_size,
                dropout=dropout,
                learning_rate=learning_rate,
                seed=42,
                max_epochs=50,
                enable_progress_bar=False,
                # Note: We assume train_tft is updated to accept an optional 'pruning_callback'
                pruning_callback=pruning_callback
            )
            print(f"[HPO] Trial {trial.number} Finished | Val Loss: {val_loss:.4f}")
            return val_loss

        except optuna.exceptions.TrialPruned:
            print(f"[HPO] Trial {trial.number} PRUNED (Unpromising architecture).")
            raise optuna.exceptions.TrialPruned()

        except Exception as e:
            print(f"[HPO]  Trial {trial.number} FAILED: {e}")
            return float("inf")

    # 4. Fault-Tolerant SQLite Database with explicit timeout to prevent Colab locks
    study_name = "tft_nifty_optimization"
    storage_name = f"sqlite:///{study_name}.db?timeout=60"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10) # Pruning logic
    )

    # 5. Run the optimization
    study.optimize(objective, n_trials=n_trials)

    print("\n[HPO]  Optimization Complete!")
    print(f"[HPO] Best validation loss: {study.best_value:.4f}")
    print(f"[HPO] Best parameters: {study.best_params}")

    return study.best_params
