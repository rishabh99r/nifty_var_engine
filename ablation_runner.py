# ablation_runner.py
# =============================================================================
# Phase 2 (final): Definitive 6-Arm Feature Ablation Tournament
# =============================================================================
# Isolates the predictive alpha of econometric and macro priors under a
# strict progression:
#
#   0  Skew-t GJR-GARCH benchmark (not a TFT)
#   1  Raw TFT                       known: []                       (returns+GK)
#   2  GARCH-TFT                     known: [GARCH_sigma]
#   3  +US cross-border              known: [GARCH_sigma, US_VIX_Diff]
#   4  +India domestic               known: [GARCH_sigma, India_VIX_Diff]
#   5  Full ECTFT                    known: [GARCH_sigma, US_VIX_Diff, India_VIX_Diff]
#
# All TFT arms share the SAME unknown reals [Log_Ret_Feature, GK_Vol], the SAME
# temporal split, architecture, seeds [42,123,777] and OOS horizon. time_idx is
# NOT a feature in any arm (canonical == Full ECTFT; see tft_model.py header).
#
# Phase-3 hardening (Reviewer):
#   - No monkey-patching: known/unknown features are passed EXPLICITLY to
#     train_tft() (monkey-patch eliminated).
#   - Per-seed results are saved (initiation sensitivity, Reviewer #31).
#   - Ensemble = mean-of-seed quantile forecast (Reviewer #8); the q0.01/q0.50/
#     q0.99 hierarchy is AUDITED on every seed AND on the averaged ensemble
#     (Reviewer #9).
#   - Direct paired DM comparisons (Full vs GARCH-TFT, Full vs GARCH, US-only
#     vs GARCH-TFT, India-only vs GARCH-TFT) are reported with Holm-Bonferroni
#     p-values adjusted within each 3-asset comparison family (Reviewer #32).
#   - Robust PTF predict() unpacking via predict_utils.unpack_predictions.
#
# DM sign convention (metrics.diebold_mariano_test):
#   d_t = L_second - L_first   -> NEGATIVE dm = second model better.
# For "vs GJR-GARCH" rows the columns are labelled (neg=TFT better).
# =============================================================================
import os
import sys

import numpy as np
import pandas as pd

import config
import tft_model
from tft_model import train_tft
from metrics import calculate_metrics, diebold_mariano_test, holm_bonferroni, audit_quantile_monotonicity
from predict_utils import unpack_predictions

ASSETS = list(config.TICKERS.keys())
SEEDS = list(config.VALIDATION_SEEDS)

# Baseline columns merged from master_df onto every forecast panel.
BASELINE_GARCH_COL = "GARCH_VaR_99"

# ----------------------------------------------------------------------------
# Tournament specification (arm key -> known features). Unknown features are
# constant across arms. Arm 0 (GJR-GARCH) is not a TFT and is handled by
# evaluate_garch_baseline().
# ----------------------------------------------------------------------------
TFT_UNKNOWN_FEATURES = ["Log_Ret_Feature", "GK_Vol"]

ABLATION_CONFIGS = {
    "1_Raw_TFT":          {"known": []},
    "2_GARCH_TFT":        {"known": ["GARCH_sigma"]},
    "3_US_VIX_TFT":       {"known": ["GARCH_sigma", "US_VIX_Diff"]},
    "4_India_VIX_TFT":    {"known": ["GARCH_sigma", "India_VIX_Diff"]},
    "5_Full_ECTFT":       {"known": ["GARCH_sigma", "US_VIX_Diff", "India_VIX_Diff"]},
}

# Pre-specified direct paired comparisons: (comparison_label, first_arm_or_col,
# second_arm_or_col). DM is computed as d_t = L(second) - L(first), so a
# NEGATIVE dm_stat means the SECOND model has lower pinball loss.
PAIRWISE_COMPARISONS = [
    ("Full ECTFT vs GARCH-TFT",     "5_Full_ECTFT",  "2_GARCH_TFT"),
    ("Full ECTFT vs GJR-GARCH",     "5_Full_ECTFT",  "__GARCH__"),
    ("US-only vs GARCH-TFT",        "3_US_VIX_TFT",  "2_GARCH_TFT"),
    ("India-only vs GARCH-TFT",     "4_India_VIX_TFT", "2_GARCH_TFT"),
]

Q_COLS = ["TFT_q01", "TFT_q50", "TFT_q99"]


# ----------------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------------
def _score_panel(panel, asset, tft_var_col="TFT_q01"):
    """
    Runs the full metrics battery on one (already chronological per-asset)
    panel by aliasing the requested TFT column to TFT_VaR_99.
    Returns the calculate_metrics dict.
    """
    sub = panel[panel["ticker"] == asset].sort_values(by="time_idx").copy()
    sub["TFT_VaR_99"] = sub[tft_var_col]
    return calculate_metrics(sub)


def _format_row(model_label, asset, m):
    return {
        "Model Variant": model_label,
        "Asset": asset,
        "Breaches": m["breaches"],
        "Total Obs": m["total_obs"],
        "Kupiec p-value": round(m["kupiec_p_value"], 4),
        "DM Stat (neg=TFT)": round(m["dm_stat"], 4),
        "DM p-value": round(m["dm_p_value"], 4),
        "Mean Loss Diff": round(m["mean_loss_diff"], 6),
    }


def _chrono_panel(panel):
    """Sort by ticker then time_idx; Date parsed once for downstream join."""
    out = panel.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values(by=["ticker", "time_idx"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Arm 0: GJR-GARCH baseline on the same OOS window used by every TFT arm.
# DM fields are a self-comparison (0 / p=1.0) included only to anchor the table.
# ----------------------------------------------------------------------------
def build_garch_oos_panel(master_df):
    max_idx = master_df["time_idx"].max()
    test_cutoff = max_idx - config.BACKTEST_DAYS
    oos = master_df[master_df["time_idx"] > test_cutoff].copy()
    oos = oos.dropna(subset=["Log_Ret", "GARCH_VaR_99"])
    return _chrono_panel(oos)


def evaluate_garch_baseline(master_df):
    oos = build_garch_oos_panel(master_df)
    rows = []
    for ticker in ASSETS:
        sub = oos[oos["ticker"] == ticker].sort_values(by="time_idx").copy()
        sub["TFT_VaR_99"] = sub[BASELINE_GARCH_COL]
        m = calculate_metrics(sub)
        rows.append(_format_row("0_GJR_GARCH_Baseline", ticker, m))
    return rows, oos


# ----------------------------------------------------------------------------
# Tournament main
# ----------------------------------------------------------------------------
def run_tournament():
    print("===== PHASE 2: DEFINITIVE 6-ARM ABLATION TOURNAMENT =====")
    if not os.path.exists("master_df.csv"):
        sys.exit("[FATAL] master_df.csv not found. Run 'python build_data.py' first.")

    master_df = pd.read_csv("master_df.csv")
    print(f"[LOAD] master_df.csv: {len(master_df)} rows across "
          f"{master_df['ticker'].nunique()} tickers.")
    print(f"[SPEC] Seeds: {SEEDS} | Unknown reals (all TFT arms): {TFT_UNKNOWN_FEATURES}")
    print(f"[SPEC] time_idx is NOT a feature in any arm (canonical == Full ECTFT).")

    # ---- Arm 0: GJR-GARCH benchmark -------------------------------------
    ensemble_results = []
    seed_results = []
    baseline_rows, garch_oos = evaluate_garch_baseline(master_df)
    ensemble_results.extend(baseline_rows)
    print("\n>>> BASELINE (Skew-t GJR-GARCH, self-reference on OOS window) complete. <<<")

    # ---- Per-arm training -------------------------------------------------
    arm_ensembles = {}          # arm -> ensemble panel (chronological)
    arm_seed_panels = {}        # arm -> {seed: panel}

    for arm, features in ABLATION_CONFIGS.items():
        print(f"\n>>> RUNNING CONFIGURATION: {arm} <<<")
        print(f"    Known: {features['known']} | Unknown: {TFT_UNKNOWN_FEATURES}")

        seed_panels = {}
        for seed in SEEDS:
            print(f"  -> Training Seed {seed}...", flush=True)
            tft, trainer, best_score, val_dl, test_dl = train_tft(
                df=master_df,
                seed=seed,
                enable_progress_bar=False,
                known_features=features["known"],
                unknown_features=TFT_UNKNOWN_FEATURES,
            )

            res = tft.predict(test_dl, mode="quantiles", return_index=True)
            pred_values, index_df = unpack_predictions(res)

            dup = index_df.duplicated(subset=["time_idx", "ticker"]).sum()
            if dup:
                raise RuntimeError(f"[FATAL] {arm} seed {seed}: {dup} duplicate prediction rows.")
            n_pred = len(index_df)

            pred_df = index_df.copy()
            pred_df["TFT_q01"] = pred_values[:, 0, 0]
            pred_df["TFT_q50"] = pred_values[:, 0, 1]
            pred_df["TFT_q99"] = pred_values[:, 0, 2]

            # Merge with GARCH baselines for the DM comparison.
            panel_meta = master_df[
                ["time_idx", "ticker", "Date", "Log_Ret", BASELINE_GARCH_COL]
            ].copy()
            merged = pred_df.merge(panel_meta, on=["time_idx", "ticker"], how="inner")
            if len(merged) != n_pred:
                raise RuntimeError(
                    f"[FATAL] {arm} seed {seed}: merge dropped {n_pred - len(merged)} rows. "
                    "Check master_df for NaNs in GARCH_VaR_99 on the OOS horizon."
                )
            merged = _chrono_panel(merged)

            # Quantile-crossing audit on THIS seed's raw output.
            audit = audit_quantile_monotonicity(
                merged["TFT_q01"], merged["TFT_q50"], merged["TFT_q99"],
                raise_on_violation=True,
            )
            print(f"  -> Seed {seed}: {len(merged)} OOS rows | crossing audit: "
                  f"{audit['violations_lo'] + audit['violations_hi']} violations")

            seed_panels[seed] = merged

            # Seed-level results (initiation sensitivity, Reviewer #31).
            for ticker in ASSETS:
                m = _score_panel(merged, ticker, tft_var_col="TFT_q01")
                row = {"Model Variant": arm, "Asset": ticker, "Seed": int(seed),
                       "Breaches": m["breaches"], "Total Obs": m["total_obs"],
                       "Kupiec p-value": round(m["kupiec_p_value"], 4),
                       "DM Stat (neg=TFT)": round(m["dm_stat"], 4),
                       "DM p-value": round(m["dm_p_value"], 4),
                       "Mean Loss Diff": round(m["mean_loss_diff"], 6)}
                seed_results.append(row)
            print(f"  -> Seed {seed} scored.", flush=True)

        arm_seed_panels[arm] = seed_panels

        # ---- Ensemble = mean-of-seed quantile forecast -------------------
        print(f"  -> Ensembling {arm} across {len(SEEDS)} seeds...")
        lengths = {len(p) for p in seed_panels.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"[FATAL] {arm}: seed panels differ in length: {lengths}")

        ens = seed_panels[SEEDS[0]].copy()
        for i, seed in enumerate(SEEDS[1:], start=1):
            ens = ens.merge(
                seed_panels[seed][["time_idx", "ticker"] + Q_COLS],
                on=["time_idx", "ticker"],
                suffixes=("", f"_s{i}"),
                how="inner",
            )
        if len(ens) != len(seed_panels[SEEDS[0]]):
            raise RuntimeError(f"[FATAL] {arm}: ensemble merge lost rows.")

        for qcol in Q_COLS:
            cols_to_avg = [qcol] + [f"{qcol}_s{i}" for i in range(1, len(SEEDS))]
            ens[qcol] = ens[cols_to_avg].mean(axis=1)
        ens = _chrono_panel(ens[["time_idx", "ticker", "Date", "Log_Ret",
                                BASELINE_GARCH_COL] + Q_COLS])

        # Quantile-crossing audit on the AVERAGED ensemble output.
        audit_ens = audit_quantile_monotonicity(
            ens["TFT_q01"], ens["TFT_q50"], ens["TFT_q99"], raise_on_violation=True,
        )
        print(f"  -> Ensemble crossing audit: "
              f"{audit_ens['violations_lo'] + audit_ens['violations_hi']} violations")

        out_panel = f"ablation_ens_panel_{arm}.csv"
        ens.to_csv(out_panel, index=False)
        _mirror_to_output_dir(out_panel)
        print(f"  -> Ensemble panel ({len(ens)} rows) written to {out_panel}")
        arm_ensembles[arm] = ens

        # ---- Ensemble-level vs GJR-GARCH results --------------------------
        for ticker in ASSETS:
            m = _score_panel(ens, ticker, tft_var_col="TFT_q01")
            ensemble_results.append(_format_row(arm, ticker, m))

    # ---- Direct paired DM table (Reviewer's key comparison) ---------------
    print("\n=== DIRECT PAIRED DM COMPARISONS (Holm-adjusted within 3-asset family) ===")
    pairwise_rows = _build_pairwise_table(arm_ensembles, garch_oos, master_df)
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv("ablation_pairwise_dm.csv", index=False)
    print(pairwise_df.to_string(index=False))

    # ---- Persist result tables --------------------------------------------
    ensemble_df = pd.DataFrame(ensemble_results)
    ensemble_df.to_csv("ablation_tournament_results.csv", index=False)
    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv("ablation_seed_level_results.csv", index=False)
    _mirror_to_output_dir("ablation_tournament_results.csv")
    _mirror_to_output_dir("ablation_seed_level_results.csv")
    _mirror_to_output_dir("ablation_pairwise_dm.csv")

    print("\n================ ABLATION TOURNAMENT RESULTS (ENSEMBLE) ================")
    print(ensemble_df.to_string(index=False))
    print("========================================================================")
    print("\nPer-seed results written to ablation_seed_level_results.csv")
    print("Pairwise DM table written to ablation_pairwise_dm.csv")


def _mirror_to_output_dir(filename):
    """
    Mirrors a workspace artifact into config.OUTPUT_DIR (the GARCH_TFT_Results
    folder -- Google Drive when mounted, else a local dir) so every tournament
    CSV/report is deposited alongside the figures and checkpoints.
    """
    import shutil

    src = filename
    if not os.path.exists(src):
        return
    try:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        shutil.copy2(src, os.path.join(config.OUTPUT_DIR, os.path.basename(src)))
        print(f"  -> Mirrored {src} -> {os.path.join(config.OUTPUT_DIR, os.path.basename(src))}")
    except Exception as e:
        print(f"  [WARN] Could not mirror {src} to OUTPUT_DIR: {e}")


def _build_pairwise_table(arm_ensembles, garch_oos, master_df):
    """
    Computes the direct paired DM tests on the ENSEMBLE q0.01 forecast series
    (one loss per OOS day, aligned per asset), plus Holm-corrected p-values.
    """
    rows = []
    for label, first_arm, second_arm in PAIRWISE_COMPARISONS:
        family_p = []          # raw p per asset (for Holm within the family)
        family_asset = []
        for ticker in ASSETS:
            first_df = _align_pair(arm_ensembles, garch_oos, first_arm, ticker)
            second_df = _align_pair(arm_ensembles, garch_oos, second_arm, ticker)

            actual = first_df["Log_Ret"].values
            first_var = first_df["TFT_q01"].values
            second_var = second_df["TFT_q01"].values

            # DM convention: d_t = L(second) - L(first); negative => second better.
            dm = diebold_mariano_test(actual, first_var, second_var, q=0.01)
            family_p.append(dm["dm_p_value"])
            family_asset.append(ticker)
            rows.append({
                "Comparison": label,
                "Asset": ticker,
                "DM Stat (neg=2nd better)": round(dm["dm_stat"], 4),
                "DM p-value (raw)": round(dm["dm_p_value"], 4),
                "Mean Loss Diff (2nd-1st)": round(dm["mean_diff"], 6),
                "Holm p (3-asset family)": np.nan,  # filled after family completes
            })
        # Holm-Bonferroni within this 3-asset comparison family.
        adj = holm_bonferroni(family_p)
        for i, ticker in enumerate(family_asset):
            for r in rows:
                if r["Comparison"] == label and r["Asset"] == ticker:
                    r["Holm p (3-asset family)"] = round(float(adj[i]), 4)
    return rows


def _align_pair(arm_ensembles, garch_oos, arm_key, ticker):
    """Returns the q01/actual/GARCH series for one asset from either a TFT arm
    ensemble panel or the GJR-GARCH OOS baseline."""
    if arm_key == "__GARCH__":
        src = garch_oos
        sub = src[src["ticker"] == ticker].sort_values(by="time_idx").copy()
        sub["TFT_q01"] = sub[BASELINE_GARCH_COL]
        return sub.reset_index(drop=True)
    src = arm_ensembles[arm_key]
    return src[src["ticker"] == ticker].sort_values(by="time_idx").reset_index(drop=True)


if __name__ == "__main__":
    run_tournament()
