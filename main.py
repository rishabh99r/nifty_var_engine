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
import datetime
import json
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


def _load_deployment_state():
    """Load the deployment cadence state JSON (empty dict if absent/corrupt)."""
    try:
        with open(config.DEPLOYMENT_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_deployment_state(state):
    with open(config.DEPLOYMENT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _record_deployment_retrain(seed):
    """Record that a TFT retrain happened now (used by deployment.py cadence).

    Stores the current trading-day index (from the freshly built master_df) as
    the retrain anchor and the date. deployment.py compares the current
    time_idx against this anchor to decide when to retrain next. The canonical
    forecast is the 3-seed ENSEMBLE (not a single seed); 'seed' is recorded
    for provenance only.
    """
    state = _load_deployment_state()
    try:
        master = pd.read_csv("master_df.csv")
        max_idx = int(master["time_idx"].max())
    except Exception:
        max_idx = state.get("last_tft_retrain_idx", 0)
    state["last_tft_retrain_idx"] = max_idx
    state["retrain_provenance_seed"] = int(seed)
    state["ensemble"] = True
    state["last_tft_retrain_date"] = datetime.date.today().isoformat()
    _save_deployment_state(state)


def _fmt_val(v, decimals=4):
    """NaN-safe numeric formatter for the multi-seed report."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if v != v:  # NaN check (NaN != NaN)
        return "N/A"
    return f"{v:.{decimals}f}"


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
        print(f"  -> Diebold-Mariano Stat: {nifty_metrics['dm_stat']:.4f} (p-value: {nifty_metrics['dm_p_value']:.4f}) "
              f"[d_t = L_TFT - L_GARCH; negative = TFT lower loss]")
        print(f"  -> Tail breach depth: {nifty_metrics['es_n_exceed']} breaches; "
              f"mean std resid z = {nifty_metrics['es_mean_resid']:.3f}")

    # ------------------------------------------------------------------
    # Multi-seed aggregation (per-asset Mean +/- Std) for disclosure
    # ------------------------------------------------------------------
    print("\n=== MULTI-SEED AGGREGATION (Mean +/- Std) ===")
    agg_rows = aggregate_seed_metrics(all_seed_metrics)  # NIFTY50-focused per-seed summary
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("      MULTI-SEED VALIDATION REPORT (MEAN +/- STD ACROSS SEEDS)")
    report_lines.append("=" * 80)
    report_lines.append(f"Seeds: {config.VALIDATION_SEEDS} (n = {len(all_seed_metrics)})")
    report_lines.append("")
    report_lines.append(f"{'Metric':<24}{'Mean':>16}{'Std':>16}")
    report_lines.append("-" * 60)
    for row in agg_rows:
        report_lines.append(
            f"{row['metric']:<24}{_fmt_val(row['mean']):>16}{_fmt_val(row['std']):>16}   values={row['values']}"
        )

    # ------------------------------------------------------------------
    # ENSEMBLE (Round 23): the canonical forecast is the pre-determined mean
    # of the q={0.01,0.50,0.99} quantile columns across seeds, computed per
    # (ticker, time_idx). This is fixed BEFORE looking at test outcomes and is
    # NOT a test-selected seed.
    # ------------------------------------------------------------------
    print("\n=== BUILDING 3-SEED QUANTILE ENSEMBLE (pre-determined rule) ===")
    seed_panels = []
    for seed in config.VALIDATION_SEEDS:
        sp = pd.read_csv(seed_panel_files[seed])
        sp["Date"] = pd.to_datetime(sp["Date"])
        seed_panels.append(sp)

    # Align on a common key and average the quantile columns per (ticker, time_idx)
    q_cols = {
        "TFT_VaR_99_Raw": "TFT_VaR_99_Raw",
        "TFT_Median": "TFT_Median",
        "TFT_VaR_Upside": "TFT_VaR_Upside",
    }
    ensemble_parts = []
    for sp in seed_panels:
        keep = ["time_idx", "ticker", "Date", "Log_Ret", "GARCH_VaR_99", "GARCH_sigma"] + list(q_cols)
        ensemble_parts.append(sp[keep].copy())

    # Merge on (time_idx, ticker) with suffixes per seed
    merged = ensemble_parts[0]
    for i, sp in enumerate(ensemble_parts[1:], start=1):
        merged = merged.merge(
            sp[["time_idx", "ticker"] + list(q_cols)],
            on=["time_idx", "ticker"],
            suffixes=("", f"_s{i}"),
            how="inner",
        )

    # Average the q-columns across the seed-suffixed copies
    for col in q_cols:
        cols_to_avg = [col] + [f"{col}_s{i}" for i in range(1, len(ensemble_parts))]
        merged[col] = merged[cols_to_avg].mean(axis=1)

    ens_panel = merged[["time_idx", "ticker", "Date", "Log_Ret", "GARCH_VaR_99", "GARCH_sigma",
                        "TFT_VaR_99_Raw", "TFT_Median", "TFT_VaR_Upside"]].copy()
    ens_panel["TFT_VaR_99"] = ens_panel["TFT_VaR_99_Raw"]
    ens_panel["Date"] = ens_panel["Date"].dt.strftime("%Y-%m-%d")
    ens_panel = ens_panel.sort_values(by=["Date", "ticker"]).reset_index(drop=True)

    ens_panel.to_csv("test_tft_predictions_panel.csv", index=False)
    nifty_ens = ens_panel[ens_panel["ticker"] == "NIFTY50"].copy()
    nifty_ens.to_csv("test_tft_predictions.csv", index=False)
    print(f"[CANONICAL] 3-seed ENSEMBLE panel written to test_tft_predictions_panel.csv "
          f"({len(ens_panel)} rows across {ens_panel['ticker'].nunique()} tickers)")

    # Ensemble metrics per asset (computed once on the predetermined ensemble)
    ensemble_metrics = {}
    for t in ens_panel["ticker"].unique():
        sub = ens_panel[ens_panel["ticker"] == t].copy()
        ensemble_metrics[t] = calculate_metrics(sub)
    report_lines.append("")
    report_lines.append("ENSEMBLE (mean-of-seeds q0.01) per-asset breach counts:")
    for t, m in ensemble_metrics.items():
        report_lines.append(f"  {t}: {m['breaches']} breaches / {m['total_obs']} "
                            f"(Basel {m['basel_zone']}), Kupiec p={m['kupiec_p_value']:.4f}, "
                            f"DM={m['dm_stat']:.4f} (p={m['dm_p_value']:.4f})")
    report_lines.append("NOTE: canonical tables use the pre-determined 3-seed ensemble, "
                        "never a test-selected seed.")
    report_lines.append("")

    with open("multi_seed_validation_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    # Persist deployment retrain anchor (ensemble version: no single median seed).
    # Keep MEDIAN_SEED_FILE for backward compat but note it is not used for the
    # canonical forecast.
    with open(config.MEDIAN_SEED_FILE, "w") as f:
        f.write("ENSEMBLE")
    _record_deployment_retrain(seed=config.VALIDATION_SEEDS[0])

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
