# specialist_runner.py
# =============================================================================
# RD4: q0.01-Specialist vs Aggregate Objective (Step 3).
#
# After HPO finds the best architecture (hpo_best_trial.json), this trains it
# TWICE on the same split:
#   (a) Aggregate: QuantileLoss on [0.01, 0.50, 0.99]   (3 outputs)
#   (b) Specialist: QuantileLoss on [0.01] only          (1 output)
# then compares the out-of-sample 1% VaR metrics against the GJR-GARCH
# benchmark on the same OOS horizon.
#
# If the q0.01-SPECIALIST still fails to beat GJR-GARCH, the "econometric
# benchmark is too strong" narrative is supported: no neural architecture
# (even one optimized solely for the 1% quantile) can out-predict the
# skew-t GJR-GARCH benchmark on this sample.
#
# Usage:  python specialist_runner.py --hpo hpo_best_trial.json --seed 42
# =============================================================================
import argparse
import json
import os

import numpy as np
import pandas as pd

import config
from tft_model import train_tft
from predict_utils import unpack_predictions
from metrics import calculate_metrics, audit_quantile_monotonicity

# ---------------------------------------------------------------------------
# Patch pytorch_forecasting bug when target_quantiles does not include 0.5.
# QuantileLoss.to_prediction() defaults to the MEDIAN (q=0.5) quantile, which
# does not exist for the q0.01-SPECIALIST model (quantiles=[0.01] only) --
# without this patch, any call to to_prediction() would index out of range.
# The fallback returns the first (only) quantile instead.
# ---------------------------------------------------------------------------
import pytorch_forecasting.metrics.quantile as qm


def _safe_to_prediction(self, y_pred, *args, **kwargs):
    if 0.5 in self.quantiles:
        idx = self.quantiles.index(0.5)
    else:
        idx = 0
    return y_pred[..., idx]


qm.QuantileLoss.to_prediction = _safe_to_prediction

ASSETS = list(config.TICKERS.keys())
Q01 = 0.01
AGG_QUANTILES = [0.01, 0.50, 0.99]
SPEC_QUANTILES = [0.01]


def _build_oos_panel(master_df, test_dl, tft, label, seed):
    """Predict on the OOS test split, merge GARCH benchmark, return sorted panel."""
    res = tft.predict(test_dl, mode="quantiles", return_index=True)
    pred_values, index_df = unpack_predictions(res)

    dup = index_df.duplicated(subset=["time_idx", "ticker"]).sum()
    if dup:
        raise RuntimeError(f"[FATAL] {label} seed {seed}: {dup} duplicate rows.")

    pred_df = index_df.copy()
    pred_df["TFT_q01"] = pred_values[:, 0, 0]
    if pred_values.shape[2] >= 3:
        pred_df["TFT_q50"] = pred_values[:, 0, 1]
        pred_df["TFT_q99"] = pred_values[:, 0, 2]
        audit = audit_quantile_monotonicity(
            pred_df["TFT_q01"], pred_df["TFT_q50"], pred_df["TFT_q99"])
        print(f"  -> {label} seed {seed}: crossing audit "
              f"{audit['violations_lo'] + audit['violations_hi']} violations")
    else:
        pred_df["TFT_q50"] = np.nan
        pred_df["TFT_q99"] = np.nan
        print(f"  -> {label} seed {seed}: single-quantile (q0.01 specialist) -- no crossing audit.")

    panel_meta = master_df[["time_idx", "ticker", "Date", "Log_Ret", "GARCH_VaR_99"]].copy()
    merged = pred_df.merge(panel_meta, on=["time_idx", "ticker"], how="inner")
    if len(merged) != len(pred_df):
        raise RuntimeError(f"[FATAL] {label} seed {seed}: merge dropped rows.")
    merged = merged.sort_values(by=["ticker", "time_idx"]).reset_index(drop=True)
    return merged


def _score(panel, tft_q_col="TFT_q01"):
    out = {}
    for asset in ASSETS:
        sub = panel[panel["ticker"] == asset].sort_values(by="time_idx").copy()
        sub["TFT_VaR_99"] = sub[tft_q_col]
        m = calculate_metrics(sub)
        out[asset] = m
    return out


def _load_hpo_params(hpo_json):
    """
    Loads the best HPO architecture from hpo_best_trial.json, tolerating BOTH
    schemas:
      (a) flat dict  -> {"hidden_size": 64, "attention_head_size": 4, ...}
      (b) nested     -> {"best_value": ..., "best_params": {...}}
    Normalizes 'attention_head_size' -> 'attention_heads' (the train_tft arg).
    """
    with open(hpo_json) as f:
        data = json.load(f)
    if isinstance(data, dict) and "best_params" in data:
        hp = dict(data["best_params"])
    else:
        hp = dict(data)
    # Normalize the attention-heads key name.
    if "attention_head_size" in hp and "attention_heads" not in hp:
        hp["attention_heads"] = hp.pop("attention_head_size")
    return hp


def run_specialist(hpo_json="hpo_best_trial.json", seed=42, max_epochs=None):
    if max_epochs is None:
        max_epochs = config.MAX_EPOCHS
    if not os.path.exists(hpo_json):
        raise FileNotFoundError(
            f"[FATAL] {hpo_json} not found. Run hpo_runner.py first to get the "
            f"best architecture."
        )
    hp = _load_hpo_params(hpo_json)
    print("===== q0.01-SPECIALIST vs AGGREGATE OBJECTIVE =====")
    print(f"[HPO] Best architecture: {hp}")

    if not os.path.exists("master_df.csv"):
        raise FileNotFoundError("[FATAL] master_df.csv not found. Run build_data.py first.")
    master_df = pd.read_csv("master_df.csv")

    results = {}
    for label, qs in [("aggregate", AGG_QUANTILES), ("specialist", SPEC_QUANTILES)]:
        print(f"\n>>> Training {label} (quantiles={qs}) <<<")
        tft, trainer, best_score, val_dl, test_dl = train_tft(
            df=master_df,
            hidden_size=hp["hidden_size"],
            attention_heads=hp["attention_heads"],
            encoder_length=hp["encoder_length"],
            dropout=hp["dropout"],
            learning_rate=hp["learning_rate"],
            seed=seed,
            max_epochs=max_epochs,
            enable_progress_bar=False,
            target_quantiles=qs,
        )
        panel = _build_oos_panel(master_df, test_dl, tft, label, seed)
        out_csv = f"specialist_{label}_panel_seed{seed}.csv"
        panel.to_csv(out_csv, index=False)
        print(f"  -> {label} panel saved -> {out_csv}")
        results[label] = _score(panel)

    # GJR-GARCH benchmark on the same OOS window.
    max_idx = master_df["time_idx"].max()
    test_cutoff = max_idx - config.BACKTEST_DAYS
    oos = master_df[master_df["time_idx"] > test_cutoff].copy()
    oos = oos.dropna(subset=["Log_Ret", "GARCH_VaR_99"]).sort_values(by=["ticker", "time_idx"])
    garch_assets = {}
    for asset in ASSETS:
        sub = oos[oos["ticker"] == asset].sort_values(by="time_idx").copy()
        sub["TFT_VaR_99"] = sub["GARCH_VaR_99"]
        garch_assets[asset] = calculate_metrics(sub)

    # ---- Comparison table (DM is vs GJR-GARCH: d_t = L_TFT - L_GARCH) ------
    rows = []
    for label in ("aggregate", "specialist"):
        for asset in ASSETS:
            m = results[label][asset]
            g = garch_assets[asset]
            rows.append({
                "Model": f"TFT-{label}",
                "Asset": asset,
                "Breaches": m["breaches"],
                "Kupiec p": round(m["kupiec_p_value"], 4),
                "DM Stat (neg=TFT)": round(m["dm_stat"], 4),
                "DM p": round(m["dm_p_value"], 4),
                "Mean Loss Diff": round(m["mean_loss_diff"], 6),
            })
    for asset in ASSETS:
        g = garch_assets[asset]
        rows.append({
            "Model": "GJR-GARCH",
            "Asset": asset,
            "Breaches": g["breaches"],
            "Kupiec p": round(g["kupiec_p_value"], 4),
            "DM Stat (neg=TFT)": 0.0,
            "DM p": 1.0,
            "Mean Loss Diff": 0.0,
        })
    results_df = pd.DataFrame(rows)
    out_csv = f"specialist_vs_aggregate_results_seed{seed}.csv"
    results_df.to_csv(out_csv, index=False)

    print("\n================ SPECIALIST vs AGGREGATE vs GARCH ================")
    print(results_df.to_string(index=False))
    print(f"\nSaved -> {out_csv}")
    return results_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="q0.01-specialist vs aggregate objective.")
    ap.add_argument("--hpo", default="hpo_best_trial.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=None)
    args = ap.parse_args()
    run_specialist(hpo_json=args.hpo, seed=args.seed, max_epochs=args.max_epochs)
