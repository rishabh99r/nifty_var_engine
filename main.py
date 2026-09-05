# main.py
import os
import shutil
import warnings
import numpy as np
import pandas as pd
from tft_model import train_tft, generate_and_save_predictions
from metrics import calculate_metrics, evaluate_panel_metrics

warnings.filterwarnings("ignore")

DRIVE_DIR = "/content/drive/MyDrive/GARCH_TFT_Results"

# Discovered Champion Configuration
CHAMPION_PARAMS = {
    'hidden_size': 64,
    'dropout': 0.30,
    'learning_rate': 0.001552
}

# 3-Seed Validation Suite
VALIDATION_SEEDS = [42, 123, 777]


def main():
    print("===== INITIALIZING NIFTY PANEL RISK ENGINE (FAST-TRACK EXECUTION) =====")
    csv_path = "master_df.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] {csv_path} not found! Run 'python build_data.py' first.")

    # Load master panel cleanly without coercing arbitrary index columns
    master_df = pd.read_csv(csv_path)
    print(f"[LOAD] Multi-series dataset loaded ({len(master_df)} rows across tickers: {master_df['ticker'].unique()}).")

    print(f"\n=== PHASE 1: DIRECT CHAMPION ARCHITECTURE TRAINING ===")
    print(f"Target Configuration: {CHAMPION_PARAMS}")

    all_seed_results = []

    for seed in VALIDATION_SEEDS:
        print(f"\n{'-'*60}")
        print(f"[RUN] Training Panel TFT with Seed {seed}...")
        print(f"{'-'*60}")

        tft, trainer, best_score, val_dataloader, test_dataloader = train_tft(
            df=master_df,
            hidden_size=CHAMPION_PARAMS['hidden_size'],
            dropout=CHAMPION_PARAMS['dropout'],
            learning_rate=CHAMPION_PARAMS['learning_rate'],
            seed=seed,
            max_epochs=80,
            encoder_length=21,
            backtest_days=500,
            enable_progress_bar=True
        )

        print(f"\n[INFERENCE] Generating out-of-sample test predictions for Seed {seed}...")
        nifty_preds = generate_and_save_predictions(
            tft, test_dataloader, master_df,
            output_csv=f"test_tft_predictions_seed_{seed}.csv",
            panel_csv=f"test_tft_predictions_panel_seed_{seed}.csv"
        )

        # Retain canonical file pointers for downstream report scripts
        shutil.copy(f"test_tft_predictions_seed_{seed}.csv", "test_tft_predictions.csv")
        shutil.copy(f"test_tft_predictions_panel_seed_{seed}.csv", "test_tft_predictions_panel.csv")

        # Evaluate NIFTY 50 baseline
        nifty_metrics = calculate_metrics(nifty_preds)

        print(f"\n[AUDIT] Seed {seed} (NIFTY 50 Results):")
        print(f"  -> Out-of-Sample Days:  {nifty_metrics['total_obs']}")
        print(f"  -> 99% VaR Breaches:    {nifty_metrics['breaches']} (Basel {nifty_metrics['basel_zone']} Zone, Green Limit: <= {nifty_metrics['basel_limit']})")
        print(f"  -> Kupiec POF p-value:  {nifty_metrics['kupiec_p_value']:.4f}")
        print(f"  -> Christoffersen Ind:  {nifty_metrics['christ_p_value']:.4f}")
        print(f"  -> Diebold-Mariano Stat: {nifty_metrics['dm_stat']:.4f} (p-value: {nifty_metrics['dm_p_value']:.4f})")

        all_seed_results.append(nifty_metrics)

    # Persist predictions and summaries to Google Drive
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        shutil.copy("test_tft_predictions.csv", os.path.join(DRIVE_DIR, "test_tft_predictions.csv"))
        shutil.copy("test_tft_predictions_panel.csv", os.path.join(DRIVE_DIR, "test_tft_predictions_panel.csv"))
        print(f"\n[PERSISTENCE] Prediction artifacts backed up to {DRIVE_DIR}")

    print("\n=================== MULTI-SEED AUDIT SUMMARY ===================")
    avg_breaches = np.mean([m['breaches'] for m in all_seed_results])
    avg_dm_stat = np.mean([m['dm_stat'] for m in all_seed_results])
    avg_dm_pval = np.mean([m['dm_p_value'] for m in all_seed_results])
    print(f"Seeds Evaluated:               {VALIDATION_SEEDS}")
    print(f"Average NIFTY 50 Breaches:     {avg_breaches:.1f} / 500 days")
    print(f"Average Diebold-Mariano Stat:  {avg_dm_stat:.4f} (p-val: {avg_dm_pval:.4f})")
    print("================================================================")


if __name__ == "__main__":
    main()
