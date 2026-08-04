# hpo.py
import optuna
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
    print("\n[HPO] === Optimizing for 99% VaR Tick Loss ===")
    db_path = "tft_nifty_optimization.db"
    if os.path.exists(db_path): os.remove(db_path)

    pbar = tqdm(total=n_trials, desc="[HPO] Architectures", unit="trial")

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.2, 0.4, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

        try:
            # Extract val_dataloader to evaluate HPO out-of-sample properly
            tft, _, _, val_dataloader, _ = train_tft(df, hidden_size, dropout, learning_rate, 42, 35, False, None)

            val_preds, val_index = tft.predict(val_dataloader, mode="quantiles", return_index=True)
            val_var_99 = val_preds[:, 0, 0].numpy()
            val_actuals = df.loc[df['time_idx'].isin(val_index['time_idx']), 'Log_Ret'].values

            val_q01_loss = np.mean(quantile_loss(val_actuals, val_var_99, q=0.01))

            try: best_val = trial.study.best_value
            except ValueError: best_val = val_q01_loss

            pbar.set_postfix({"Best q01 Loss": f"{min(best_val, val_q01_loss):.4f}", "Hidden": hidden_size, "LR": f"{learning_rate:.4f}"})
            pbar.update(1)
            return val_q01_loss
        except Exception:
            pbar.update(1)
            return float("inf")
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    study = optuna.create_study(storage=f"sqlite:///{db_path}?timeout=60", direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    pbar.close()
    return study.best_params
