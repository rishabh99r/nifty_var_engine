# hpo.py
import optuna
import time
import warnings
import gc
import os
import torch
import numpy as np
from tqdm import tqdm
from tft_model import train_tft
from metrics import quantile_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

def optimize_hyperparameters(df, n_trials=30):
    print("\n[HPO] === Starting Robust Hyperparameter Optimization (Target: 99% VaR Tick Loss) ===")

    db_path = "tft_nifty_optimization.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"[HPO] Removed stale database '{db_path}'. Starting fresh 30-trial study.")

    pbar = tqdm(total=n_trials, desc="[HPO] Optimizing 99% VaR Tail Precision", unit="trial", leave=True)

    def objective(trial):
        # Clamped parameter bounds based on structural capacity limits
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.2, 0.4, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

        try:
            tft, _, _, val_dataloader = train_tft(
                df=df,
                hidden_size=hidden_size,
                dropout=dropout,
                learning_rate=learning_rate,
                seed=42,
                max_epochs=35,
                enable_progress_bar=False,
                pruning_callback=None
            )

            # Extract validation predictions and isolate 99% VaR (q=0.01 tail)
            val_preds, val_index = tft.predict(val_dataloader, mode="quantiles", return_index=True)
            val_var_99 = val_preds[:, 0, 0].numpy()

            # Align actual validation returns
            val_actuals = df.loc[df['time_idx'].isin(val_index['time_idx']), 'Log_Ret'].values

            # OPTUNA TARGET: Exact 0.01 Asymmetric Tick Loss on Validation Slice
            val_q01_loss = np.mean(quantile_loss(val_actuals, val_var_99, q=0.01))

            try:
                best_val = trial.study.best_value
            except ValueError:
                best_val = val_q01_loss
            best_val = min(best_val, val_q01_loss)

            pbar.set_postfix({
                "Best q01 Loss": f"{best_val:.4f}",
                "Curr q01 Loss": f"{val_q01_loss:.4f}",
                "Hidden": hidden_size,
                "LR": f"{learning_rate:.4f}"
            })
            pbar.update(1)
            return val_q01_loss

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

    print(f"\n[HPO] Optimization Complete | Best 99% VaR Tick Loss: {study.best_value:.4f}")
    print(f"[HPO] Optimal Parameters: {study.best_params}")
    return study.best_params
