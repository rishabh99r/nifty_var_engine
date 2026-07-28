# main.py
import pandas as pd
import numpy as np
import os
from hpo import optimize_hyperparameters
from tft_model import train_tft
from metrics import calculate_metrics
from config import VALIDATION_SEEDS, set_seed

def generate_predictions(tft, test_dataloader, df):
    print("[INFERENCE] Extracting out-of-sample predictions on TEST set...")
    raw_predictions, index = tft.predict(test_dataloader, mode="quantiles", return_index=True)

    tft_var_99 = raw_predictions[:, 0, 0].numpy()
    time_indices = index["time_idx"].values

    results_df = pd.DataFrame({
        "time_idx": time_indices,
        "TFT_VaR_99": tft_var_99
    })

    merged_df = results_df.merge(df[['time_idx', 'Log_Ret', 'GARCH_VaR_99']], on="time_idx", how="inner")
    merged_df.rename(columns={"Log_Ret": "Actual"}, inplace=True)

    return merged_df

def main():
    print("===== INITIALIZING NIFTY 50 RISK ENGINE (FAST TRAINING PIPELINE) =====")

    csv_path = "master_df.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] {csv_path} not found! Run 'python build_data.py' first to generate data.")

    print(f"[LOAD] Loading pre-computed dataset from {csv_path}...")
    master_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"[LOAD] Dataset successfully loaded ({len(master_df)} trading days).")

    print("\n=== PHASE 1: HYPERPARAMETER OPTIMIZATION (VALIDATION SET) ===")
    best_params = optimize_hyperparameters(master_df, n_trials=30)

    print("\n=== PHASE 2: 5-SEED ROBUSTNESS AUDIT (TEST SET) ===")
    all_metrics = []

    for seed in VALIDATION_SEEDS:
        print(f"\n[AUDIT] Launching Network with Seed: {seed}")
        set_seed(seed)

        tft, trainer, val_loss, test_dataloader = train_tft(
            df=master_df,
            hidden_size=best_params['hidden_size'],
            dropout=best_params['dropout'],
            learning_rate=best_params['learning_rate'],
            seed=seed,
            max_epochs=100,
            enable_progress_bar=True,
            pruning_callback=None
        )

        results_df = generate_predictions(tft, test_dataloader, master_df)
        seed_metrics = calculate_metrics(results_df)

        print(f"\n[AUDIT] Seed {seed} Results:")
        print(f"  -> Test Days:     {len(results_df)}")
        print(f"  -> TFT Failures:  {seed_metrics['tft_failures']} (Limit: {seed_metrics['basel_limit']})")
        print(f"  -> Kupiec p-val:  {seed_metrics['kupiec_p_value']:.4f}")
        print(f"  -> DM Statistic:  {seed_metrics['dm_statistic']:.4f} (p-val: {seed_metrics['dm_p_value']:.4f})")

        all_metrics.append(seed_metrics)

    print("\n=== PIPELINE EXECUTION COMPLETE ===")
    avg_dm_pval = np.mean([m['dm_p_value'] for m in all_metrics])
    avg_failures = np.mean([m['tft_failures'] for m in all_metrics])
    print(f"Final Architecture: {best_params}")
    print(f"Average TFT Failures across 5 seeds: {avg_failures}")
    print(f"Average Diebold-Mariano p-value: {avg_dm_pval:.4f}")

if __name__ == "__main__":
    main()
