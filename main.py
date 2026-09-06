# main.py
# =============================================================================
# Orchestrates the full multi-seed validation suite for the
# Econometrically-Conditioned TFT VaR pipeline.
#
# Statistical disclosure requirements (for publishability):
#   - All 3 seeds are trained and evaluated.
#   - Per-seed artifacts are retained (test_tft_predictions_panel_seed_<s>.csv).
#   - Aggregated Mean +/- Std metrics are computed across seeds and written to
#     multi_seed_validation_report.txt.
#   - The "canonical" panel for report plotting is the MEDIAN-performing seed
#     (by NIFTY50 pinball loss), explicitly captioned, never a cherry-picked
#     best seed.
# =============================================================================
import os
import shutil

import numpy as np
import pandas as pd

import config
from metrics import calculate_metrics, aggregate_seed_metrics
from tft_model import train_tft, generate_and_save_predictions

CHAMPION_PARAMS = {
    "hidden_size": config.HIDDEN_SIZE,
    "dropout": config.DROPOUT,
    "learning_rate": config.LEARNING_RATE,
}


def _fmt_val(v, decimals=4):
    """NaN-safe numeric formatter for the multi-seed report."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if v != v:  # NaN check (NaN != NaN)
        return "N/A"
    return f"{v:.{decimals}f}"


def _rank_seeds_by_pinball(seed_pred_files):
    """
    Ranks seeds by the NIFTY50 mean pinball loss on the out-of-sample horizon.
    Returns the seed with the MEDIAN performance (neither best nor worst).
    """
    from metrics import pinball_loss

    scores = {}
    for seed, path in seed_pred_files.items():
        nifty = pd.read_csv(path)
        actual = nifty["Log_Ret"].values
        var = nifty["TFT_VaR_99"].values
        scores[seed] = float(np.mean(pinball_loss(actual, var, q=0.01)))

    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    median_seed = ordered[len(ordered) // 2][0]
    return median_seed, scores


def main():
    print("===== INITIALIZING NIFTY PANEL RISK ENGINE (MULTI-SEED VALIDATION SUITE) =====")
    csv_path = "master_df.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] {csv_path} not found! Run 'python build_data.py' first.")

    master_df = pd.read_csv(csv_path)
    print(f"[LOAD] Multi-series dataset loaded ({len(master_df)} rows across tickers: {master_df['ticker'].unique()}).")

    print(f"\n=== PHASE 1: MULTI-SEED ECONOMETRICALLY-CONDITIONED TFT TRAINING ===")
    print(f"Target Configuration: {CHAMPION_PARAMS} | Seeds: {config.VALIDATION_SEEDS}")

    all_seed_metrics = []
    seed_pred_files = {}
    seed_panel_files = {}

    for seed in config.VALIDATION_SEEDS:
        print(f"\n{'-' * 60}")
        print(f"[RUN] Training Panel TFT with Seed {seed}...")
        print(f"{'-' * 60}")

        tft, trainer, best_score, val_dataloader, test_dataloader = train_tft(
            df=master_df,
            hidden_size=CHAMPION_PARAMS["hidden_size"],
            dropout=CHAMPION_PARAMS["dropout"],
            learning_rate=CHAMPION_PARAMS["learning_rate"],
            seed=seed,
            max_epochs=config.MAX_EPOCHS,
            encoder_length=config.ENCODER_LENGTH,
            backtest_days=config.BACKTEST_DAYS,
            enable_progress_bar=True,
        )

        print(f"\n[INFERENCE] Generating out-of-sample test predictions for Seed {seed}...")
        nifty_preds = generate_and_save_predictions(
            tft, test_dataloader, master_df, seed=seed,
            output_csv=f"test_tft_predictions_seed_{seed}.csv",
            panel_csv=f"test_tft_predictions_panel_seed_{seed}.csv",
        )

        seed_pred_files[seed] = f"test_tft_predictions_seed_{seed}.csv"
        seed_panel_files[seed] = f"test_tft_predictions_panel_seed_{seed}.csv"

        nifty_metrics = calculate_metrics(nifty_preds)
        all_seed_metrics.append(nifty_metrics)

        print(f"\n[AUDIT] Seed {seed} (NIFTY 50 Results):")
        print(f"  -> Out-of-Sample Days:  {nifty_metrics['total_obs']}")
        print(f"  -> 99% VaR Breaches:    {nifty_metrics['breaches']} (Basel {nifty_metrics['basel_zone']} Zone)")
        print(f"  -> Kupiec POF p-value:  {nifty_metrics['kupiec_p_value']:.4f}")
        print(f"  -> Christoffersen Ind:  {nifty_metrics['christ_p_value']:.4f}")
        print(f"  -> Diebold-Mariano Stat: {nifty_metrics['dm_stat']:.4f} (p-value: {nifty_metrics['dm_p_value']:.4f})")
        print(f"  -> McNeil-Frey ES t-stat: {nifty_metrics['es_t_stat']:.4f} (p-value: {nifty_metrics['es_p_value']:.4f})")

    # ------------------------------------------------------------------
    # Aggregate across seeds (honest statistical disclosure)
    # ------------------------------------------------------------------
    print("\n=== MULTI-SEED AGGREGATION (Mean +/- Std) ===")
    agg_rows = aggregate_seed_metrics(all_seed_metrics)
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("      MULTI-SEED VALIDATION REPORT (MEAN +/- STD ACROSS SEEDS)")
    report_lines.append("=" * 80)
    report_lines.append(f"Seeds: {config.VALIDATION_SEEDS} (n = {len(all_seed_metrics)})")
    report_lines.append("")
    report_lines.append(f"{'Metric':<24}{'Mean':>16}{'Std':>16}")
    report_lines.append("-" * 60)
    # FIX 14.6: NaN-safe formatting (module-level _fmt_val) so non-testable
    # metrics (e.g. ES t-stat when no seed is testable) render as N/A not nan.
    for row in agg_rows:
        report_lines.append(
            f"{row['metric']:<24}{_fmt_val(row['mean']):>16}{_fmt_val(row['std']):>16}   values={row['values']}"
        )

    # Select median-performing seed for canonical report artifacts
    median_seed, scores = _rank_seeds_by_pinball(seed_pred_files)
    report_lines.append("")
    report_lines.append(f"Median-performing seed (by NIFTY50 pinball loss): {median_seed}")
    report_lines.append(f"Pinball loss per seed: {scores}")
    report_lines.append("NOTE: The canonical report tables show the MEDIAN seed trajectory, not a cherry-picked best seed.")

    with open("multi_seed_validation_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    # Canonical artifacts = median seed, so downstream report scripts
    # (generate_report_plots.py) can read a single unambiguous file.
    shutil.copy(seed_pred_files[median_seed], "test_tft_predictions.csv")
    shutil.copy(seed_panel_files[median_seed], "test_tft_predictions_panel.csv")
    print(f"[CANONICAL] Median seed {median_seed} promoted to test_tft_predictions*.csv")

    # Persist to Google Drive if mounted
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        for seed in config.VALIDATION_SEEDS:
            for f in (seed_pred_files[seed], seed_panel_files[seed]):
                shutil.copy(f, os.path.join(config.OUTPUT_DIR, f))
        shutil.copy("multi_seed_validation_report.txt", os.path.join(config.OUTPUT_DIR, "multi_seed_validation_report.txt"))
        shutil.copy("test_tft_predictions.csv", os.path.join(config.OUTPUT_DIR, "test_tft_predictions.csv"))
        shutil.copy("test_tft_predictions_panel.csv", os.path.join(config.OUTPUT_DIR, "test_tft_predictions_panel.csv"))
        print(f"\n[PERSISTENCE] Prediction artifacts backed up to {config.OUTPUT_DIR}")

    print("\n=================== MULTI-SEED AUDIT SUMMARY ===================")
    breaches_vals = [m["breaches"] for m in all_seed_metrics]
    print(f"Seeds Evaluated:               {config.VALIDATION_SEEDS}")
    print(f"NIFTY50 Breaches per seed:     {breaches_vals} (mean {np.mean(breaches_vals):.1f} / {config.BACKTEST_DAYS} days)")
    dm_pvals = [m["dm_p_value"] for m in all_seed_metrics]
    print(f"DM p-values per seed:          {[f'{p:.3f}' for p in dm_pvals]}")
    print("=================================================================")


if __name__ == "__main__":
    main()
