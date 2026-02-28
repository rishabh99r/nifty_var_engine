# main.py
import pandas as pd
import numpy as np
from data_loader import fetch_and_clean_data
from garch_engine import run_rolling_garch
from hpo import optimize_hyperparameters
from tft_model import train_tft
from metrics import calculate_metrics
from config import VALIDATION_SEEDS, set_seed

def generate_predictions(tft, val_dataloader, df):
    print("[INFERENCE] Extracting out-of-sample quantile predictions...")

    # 1. Run the PyTorch Forecasting prediction engine
    # return_index=True is critical to map predictions back to the actual dates
    raw_predictions, index = tft.predict(val_dataloader, mode="quantiles", return_index=True)

    # 2. Tensor Extraction
    # raw_predictions shape is (batch, time_steps, quantiles)
    # Our quantiles were defined as [0.01, 0.5, 0.99]
    # Value at Risk looks at the worst 1% tail loss, so we want index 0 (0.01)
    tft_var_99 = raw_predictions[:, 0, 0].numpy()
    time_indices = index["time_idx"].values

    # 3. Reconstruct the Results DataFrame
    results_df = pd.DataFrame({
        "time_idx": time_indices,
        "TFT_VaR_99": tft_var_99
    })

    # 4. Merge with Actuals and GARCH Baseline for the Diebold-Mariano showdown
    merged_df = results_df.merge(df[['time_idx', 'Log_Ret', 'GARCH_VaR_99']], on="time_idx", how="inner")
    merged_df.rename(columns={"Log_Ret": "Actual"}, inplace=True)

    return merged_df

def main():
    print("===== INITIALIZING NIFTY 50 RISK ENGINE =====")

    # Step 1: Data
    raw_df = fetch_and_clean_data()
    master_df = run_rolling_garch(raw_df, csv_path="master_df.csv")

    # Step 2: HPO (Find the Optimal Architecture)
    print("\n=== PHASE 1: HYPERPARAMETER OPTIMIZATION ===")
    best_params = optimize_hyperparameters(master_df, n_trials=30)

    # Step 3: The Robustness Audit (5-Seed Loop)
    print("\n=== PHASE 2: 5-SEED ROBUSTNESS AUDIT ===")

    all_metrics = []

    for seed in VALIDATION_SEEDS:
        print(f"\n[AUDIT] Launching Network with Seed: {seed}")
        set_seed(seed)

        # Train the model
        tft, trainer, val_loss, val_dataloader = train_tft(
            df=master_df,
            hidden_size=best_params['hidden_size'],
            dropout=best_params['dropout'],
            learning_rate=best_params['learning_rate'],
            seed=seed,
            max_epochs=100,
            enable_progress_bar=False,
            pruning_callback=None # No pruning during final evaluation
        )

        # Generate Predictions & Calculate Formal Metrics
        results_df = generate_predictions(tft, val_dataloader, master_df)
        seed_metrics = calculate_metrics(results_df)

        print(f"[AUDIT] Seed {seed} Results:")
        print(f"  -> Val Loss:      {val_loss:.4f}")
        print(f"  -> TFT Failures:  {seed_metrics['tft_failures']} (Limit: {seed_metrics['basel_limit']})")
        print(f"  -> Kupiec p-val:  {seed_metrics['kupiec_p_value']:.4f}")
        print(f"  -> DM Statistic:  {seed_metrics['dm_statistic']:.4f} (p-val: {seed_metrics['dm_p_value']:.4f})")

        all_metrics.append(seed_metrics)

    # Step 4: Final Aggregation
    print("\n=== PIPELINE EXECUTION COMPLETE ===")
    avg_dm_pval = np.mean([m['dm_p_value'] for m in all_metrics])
    avg_failures = np.mean([m['tft_failures'] for m in all_metrics])
    print(f"Final Architecture: {best_params}")
    print(f"Average TFT Failures across 5 seeds: {avg_failures}")
    print(f"Average Diebold-Mariano p-value: {avg_dm_pval:.4f}")

if __name__ == "__main__":
    main()
