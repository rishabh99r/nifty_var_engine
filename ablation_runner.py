# ablation_runner.py
# =============================================================================
# Phase 2: Feature Ablation Tournament
# Isolates the predictive alpha of econometric and macro priors.
#
# AUDITED (against tft_model.py / metrics.py / pytorch-forecasting 0.10+):
#   FIX 1 (FATAL): PTF >= 0.9 returns a 5-field `Prediction` namedtuple from
#         predict(mode="quantiles", return_index=True), NOT a 2-tuple. The
#         original `preds, index_df = tft.predict(...)` raised
#         "ValueError: too many values to unpack (expected 2)".  Handled
#         defensively like the champion code (tft_model.generate_and_save_predictions).
#   FIX 2 (validity): panels were fed to calculate_metrics() UNSORTED. Kupiec
#         is order-free, but Christoffersen (transition counts), Engle-Manganelli
#         DQ (lagged hits) and the Diebold-Mariano HAC autocovariances ALL assume
#         strict chronological order per asset. Sort by Date/ticker first.
#   FIX 3 (robustness): assert prediction rows are unique per (time_idx, ticker)
#         and that the GARCH-VaR merge drops no rows (mirror of FIX 12.4 in
#         tft_model.py) -- otherwise a NaN GARCH column silently shrinks a seed
#         panel and corrupts the cross-seed ensemble.
#   FIX 4 (hygiene): restore the ORIGINAL tft_model.build_datasets after each
#         configuration so a failure inside one arm cannot bleed its feature
#         lists into the next arm.
#   FIX 5 (completeness): append the GJR-GARCH baseline (already validated
#         elsewhere) as arm "0_GJR_GARCH_Baseline" on the SAME out-of-sample
#         window, so ablation_tournament_results.csv is the full 4-way table:
#         Baseline | Raw TFT | Conditioned TFT | Full ECTFT.
#
# METHODOLOGY NOTES:
#   - time_idx is excluded from the feature lists of ALL arms so no variant can
#     learn an absolute calendar-time shortcut (Reviewer 3 hypothesis). Note
#     PTF still injects the BOUNDED positional identifiers
#     relative_time_idx (-encoder..0) and encoder_length because
#     add_relative_time_idx=True / add_encoder_length=True. Those are window-
#     relative, reset per sequence, and identical across arms -- they do not
#     reintroduce the unbounded absolute time trend that time_idx carried.
#   - DM convention (metrics.diebold_mariano_test): d_t = L_TFT - L_GARCH, so a
#     NEGATIVE dm_stat / mean_loss_diff means the TFT variant is BETTER than the
#     GJR-GARCH baseline. The baseline arm compares GARCH against itself (dm=0).
# =============================================================================
import os
import sys

import numpy as np
import pandas as pd

import config
import tft_model
from metrics import calculate_metrics

# ----------------------------------------------------------------------------
# Feature sets for the tournament. 'time_idx' is intentionally EXCLUDED from
# every arm (see header). GARCH_VaR_99 / Log_Ret remain the comparison baseline
# and the target respectively -- never listed as model features.
# ----------------------------------------------------------------------------
ABLATION_CONFIGS = {
    "1_Raw_TFT": {
        "known": [],
        "unknown": ["Log_Ret_Feature", "GK_Vol"],
    },
    "2_GARCH_TFT": {
        "known": ["GARCH_sigma"],
        "unknown": ["Log_Ret_Feature", "GK_Vol"],
    },
    "3_Full_ECTFT": {
        "known": ["GARCH_sigma", "US_VIX_Diff", "India_VIX_Diff"],
        "unknown": ["Log_Ret_Feature", "GK_Vol"],
    },
}

ASSETS = list(config.TICKERS.keys())          # NIFTY50, BANKNIFTY, NIFTYIT
TFT_VAR_COL = "TFT_VaR_99"


# ----------------------------------------------------------------------------
# Monkey-patch of tft_model.build_datasets -> dynamic feature lists.
# Signature must match the original (df, encoder_length, backtest_days, val_days).
# ----------------------------------------------------------------------------
def inject_custom_datasets(known_features, unknown_features):
    from pytorch_forecasting import TimeSeriesDataSet

    def custom_build_datasets(df, encoder_length=None, backtest_days=None, val_days=None):
        if backtest_days is None:
            backtest_days = config.BACKTEST_DAYS
        if encoder_length is None or isinstance(encoder_length, bool):
            encoder_length = config.ENCODER_LENGTH
        if val_days is None:
            val_days = config.VAL_DAYS

        df = df.copy().reset_index(drop=True)
        df["ticker"] = df["ticker"].astype(str)

        max_idx = df["time_idx"].max()
        test_cutoff = max_idx - backtest_days
        val_cutoff = test_cutoff - val_days

        valid_known = [c for c in known_features if c in df.columns]
        valid_unknown = [c for c in unknown_features if c in df.columns]

        train_df = df[df["time_idx"] <= val_cutoff].reset_index(drop=True)

        training_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="Log_Ret",
            group_ids=["ticker"],
            static_categoricals=["ticker"],
            min_encoder_length=encoder_length,
            max_encoder_length=encoder_length,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_known_categoricals=[],
            time_varying_known_reals=valid_known,
            time_varying_unknown_reals=valid_unknown,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        val_df = df[
            (df["time_idx"] > val_cutoff - encoder_length) & (df["time_idx"] <= test_cutoff)
        ].reset_index(drop=True)
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, val_df, predict=False, stop_randomization=True
        )

        test_df = df[df["time_idx"] > test_cutoff - encoder_length].reset_index(drop=True)
        test_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, test_df, predict=False, stop_randomization=True
        )

        return training_dataset, validation_dataset, test_dataset, test_cutoff

    # Keep the pristine function for restoration after each arm.
    inject_custom_datasets._original = tft_model.build_datasets
    tft_model.build_datasets = custom_build_datasets


def restore_build_datasets():
    """Restore the pristine tft_model.build_datasets (defensive against bleed)."""
    original = getattr(inject_custom_datasets, "_original", None)
    if original is not None:
        tft_model.build_datasets = original


# ----------------------------------------------------------------------------
# Predict-unpacking that is robust across PTF versions:
#   >= 0.9 : Prediction namedtuple(output, x, index, decoder_lengths, y)
#   <  0.9 : tuple (output, index)
# Mirrors the defensive handling in tft_model.generate_and_save_predictions().
# ----------------------------------------------------------------------------
def unpack_predictions(result):
    if hasattr(result, "output") and hasattr(result, "index"):
        pred_values = result.output.cpu().numpy()
        index_df = result.index.copy()
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        pred_values = result[0].cpu().numpy()
        index_df = result[1].copy()
    else:
        raise TypeError(
            f"Unexpected predict() return type {type(result)}. "
            "Expected a PTF Prediction namedtuple or a (output, index) tuple."
        )
    return pred_values, index_df


# ----------------------------------------------------------------------------
# GJR-GARCH baseline on the SAME out-of-sample window used by the TFT arms.
# DM fields are self-comparison (0 / p=1.0), included only to anchor the table.
# ----------------------------------------------------------------------------
def evaluate_garch_baseline(master_df):
    max_idx = master_df["time_idx"].max()
    test_cutoff = max_idx - config.BACKTEST_DAYS  # same OOS window as build_datasets
    oos = master_df[master_df["time_idx"] > test_cutoff].copy()
    oos = oos.dropna(subset=["Log_Ret", "GARCH_VaR_99"])

    rows = []
    for ticker in ASSETS:
        sub = oos[oos["ticker"] == ticker].sort_values(by="Date").copy()
        sub[TFT_VAR_COL] = sub["GARCH_VaR_99"]  # baseline evaluated against itself
        m = calculate_metrics(sub)
        rows.append(_format_metrics_row("0_GJR_GARCH_Baseline", ticker, m))
    return rows


def _format_metrics_row(model_name, ticker, m):
    return {
        "Model Variant": model_name,
        "Asset": ticker,
        "Breaches": m["breaches"],
        "Total Obs": m["total_obs"],
        "Kupiec p-value": round(m["kupiec_p_value"], 4),
        "DM Stat (neg=TFT)": round(m["dm_stat"], 4),
        "DM p-value": round(m["dm_p_value"], 4),
        "Mean Loss Diff": round(m["mean_loss_diff"], 6),
    }


# ----------------------------------------------------------------------------
# Main tournament
# ----------------------------------------------------------------------------
def run_tournament():
    print("===== PHASE 2: ABLATION TOURNAMENT =====")

    if not os.path.exists("master_df.csv"):
        sys.exit("[FATAL] master_df.csv not found. Run 'python build_data.py' first.")

    master_df = pd.read_csv("master_df.csv")
    print(f"[LOAD] master_df.csv: {len(master_df)} rows across "
          f"{master_df['ticker'].nunique()} tickers.")

    results = evaluate_garch_baseline(master_df)
    print("\n>>> BASELINE (GJR-GARCH, self-reference on OOS window) complete. <<<")

    for model_name, features in ABLATION_CONFIGS.items():
        print(f"\n>>> RUNNING CONFIGURATION: {model_name} <<<")
        print(f"Features -> Known: {features['known']} | Unknown: {features['unknown']}")

        inject_custom_datasets(features["known"], features["unknown"])
        seed_panels = []

        try:
            for seed in config.VALIDATION_SEEDS:
                print(f"  -> Training Seed {seed}...", flush=True)
                tft, trainer, best_score, val_dl, test_dl = train_tft(
                    df=master_df,
                    seed=seed,
                    enable_progress_bar=False,
                )

                res = tft.predict(test_dl, mode="quantiles", return_index=True)
                pred_values, index_df = unpack_predictions(res)

                # FIX 3a: no duplicate (ticker, time_idx) rows from the DataLoader.
                dup = index_df.duplicated(subset=["time_idx", "ticker"]).sum()
                if dup:
                    raise RuntimeError(f"[FATAL] Seed {seed}: {dup} duplicate prediction rows.")
                n_pred = len(index_df)

                pred_df = index_df.copy()
                pred_df[TFT_VAR_COL] = pred_values[:, 0, 0]  # q = 0.01

                # Merge with GARCH baselines for the DM comparison.
                panel_meta = master_df[
                    ["time_idx", "ticker", "Date", "Log_Ret", "GARCH_VaR_99"]
                ].copy()
                merged = pred_df.merge(panel_meta, on=["time_idx", "ticker"], how="inner")

                # FIX 3b: mirror of tft_model FIX 12.4 -- a dropped row (e.g. NaN
                # GARCH column on a test day) would silently corrupt the ensemble.
                if len(merged) != n_pred:
                    raise RuntimeError(
                        f"[FATAL] Seed {seed}: merge dropped {n_pred - len(merged)} rows. "
                        "Check master_df for NaNs in GARCH_VaR_99 on the OOS horizon."
                    )

                seed_panels.append(merged)
                print(f"  -> Seed {seed}: {len(merged)} OOS rows.", flush=True)
        finally:
            # FIX 4: never let one arm's feature lists leak into the next arm.
            restore_build_datasets()

        # --- 3-Seed ensemble for this configuration (mean of q0.01 per cell) ---
        lengths = {len(p) for p in seed_panels}
        if len(lengths) != 1:
            raise RuntimeError(f"[FATAL] {model_name}: seed panels differ in length: {lengths}")

        print(f"  -> Ensembling {model_name} across {len(seed_panels)} seeds...")
        ens_panel = seed_panels[0].copy()
        for i in range(1, len(seed_panels)):
            ens_panel = ens_panel.merge(
                seed_panels[i][["time_idx", "ticker", TFT_VAR_COL]],
                on=["time_idx", "ticker"],
                suffixes=("", f"_s{i}"),
                how="inner",
            )

        if len(ens_panel) != len(seed_panels[0]):
            raise RuntimeError(f"[FATAL] {model_name}: ensemble merge lost rows.")

        cols_to_avg = [TFT_VAR_COL] + [f"{TFT_VAR_COL}_s{i}" for i in range(1, len(seed_panels))]
        ens_panel[TFT_VAR_COL] = ens_panel[cols_to_avg].mean(axis=1)

        # Drop the per-seed q0.01 duplicates left over by the suffix merge.
        keep_cols = ["time_idx", "ticker", "Date", "Log_Ret", "GARCH_VaR_99", TFT_VAR_COL]
        ens_panel = ens_panel[keep_cols].copy()

        # FIX 2: chronological order per asset BEFORE any time-series test.
        ens_panel["Date"] = pd.to_datetime(ens_panel["Date"])
        ens_panel = ens_panel.sort_values(by=["Date", "ticker"]).reset_index(drop=True)

        # Persist each arm's ensemble panel for reproducibility / audit plots.
        out_panel = f"ablation_ens_panel_{model_name}.csv"
        ens_panel.to_csv(out_panel, index=False)
        print(f"  -> Ensemble panel ({len(ens_panel)} rows) written to {out_panel}")

        # --- Evaluate the ensemble per asset ---
        for ticker in ASSETS:
            sub = ens_panel[ens_panel["ticker"] == ticker].copy()
            m = calculate_metrics(sub)
            results.append(_format_metrics_row(model_name, ticker, m))

    results_df = pd.DataFrame(results)
    results_df.to_csv("ablation_tournament_results.csv", index=False)

    print("\n================ ABLATION TOURNAMENT RESULTS ================")
    print(results_df.to_string(index=False))
    print("=============================================================")


if __name__ == "__main__":
    run_tournament()
