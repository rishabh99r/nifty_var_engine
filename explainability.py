# explainability.py
# =============================================================================
# Multi-seed TFT interpretability (VSN feature attribution + temporal attention).
#
# Cross-seed robustness requirement (for publishability): VaR metrics are
# averaged across seeds [42, 123, 777] in main.py, so the explainability must
# be aggregated across the SAME seeds. Reporting feature attribution from a
# single checkpoint would invite the reviewer question: "Is the attention
# structure a model invariant, or a quirk of one random initialization?"
#
# This script loops over every seed checkpoint, extracts VSN importance and
# temporal attention weights, and reports Mean +/- Std across seeds.
# =============================================================================
import glob
import os

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

import config

OUTPUT_DIR = config.OUTPUT_DIR + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORY_MAP = {
    "GK_Vol": "Intraday Range Volatility",
    "Log_Ret_Feature": "Autoregressive Target History",
    "GARCH_sigma": "Econometric Volatility Prior",
    "relative_time_idx": "Temporal Indexing",
    "India_VIX_Diff": "Domestic Volatility (India VIX or RV proxy)",
    "US_VIX_Diff": "Cross-Border Macro Spillover (lagged 2)",
    "time_idx": "Panel Timeline",
}


def _locate_seed_checkpoint(seed):
    """Find the champion checkpoint for a given seed (local or Drive)."""
    patterns = [
        os.path.join(OUTPUT_DIR, f"*seed{seed}*.ckpt"),
        os.path.join(OUTPUT_DIR, f"*{seed}*.ckpt"),
        f"*seed{seed}*.ckpt",
        f"*{seed}*.ckpt",
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def _extract_vsn_pct(interpretation, tft):
    """Extract encoder VSN weights as % per feature from an interpretation dict."""
    enc_weights = interpretation["encoder_variables"].detach().cpu().numpy()
    if enc_weights.ndim > 1:
        enc_weights = enc_weights.mean(axis=tuple(range(enc_weights.ndim - 1)))
    enc_weights = enc_weights.flatten()

    var_names = list(tft.encoder_variables)
    min_len = min(len(var_names), len(enc_weights))
    var_names = var_names[:min_len]
    enc_weights = enc_weights[:min_len]

    total_w = np.sum(enc_weights)
    if total_w > 0:
        pct = enc_weights / total_w * 100
    else:
        pct = np.ones_like(enc_weights) / len(enc_weights) * 100
    return {name: float(p) for name, p in zip(var_names, pct)}


def _extract_attention(interpretation):
    """
    Extract mean temporal attention weights (index 0 = MOST RECENT lag).

    FIX 14.5 (documented assumption): pytorch_forecasting's interpret_output
    attention tensor indexes the encoder history in a fixed order. We assume it
    is ordered OLDEST->NEWEST, hence the [::-1] reversal so index 0 maps to the
    most recent lag (lag-1). If the library ever changes/orders it
    NEWEST->OLDEST, this reversal would invert the reported lag labels. Verify
    against a printed sample once per pytorch_forecasting version before
    trusting the lag axis in a publication figure.
    """
    attn = interpretation["attention"].detach().cpu().numpy()
    if attn.ndim > 1:
        attn = attn.mean(axis=tuple(range(attn.ndim - 1)))
    attn = attn.flatten()
    total = np.sum(attn)
    norm = attn / total if total > 0 else np.ones_like(attn) / len(attn)
    return norm[::-1]  # assumes oldest->newest encoder ordering; index 0 = lag-1


def _build_eval_dataloader(tft, df, encoder_len):
    max_t = df["time_idx"].max()
    # FIX 12.1: predict=False (default) evaluates the FULL out-of-sample
    # horizon. predict=True would collapse the dataset to a single terminal
    # forecast per group (the live-inference mode), which is NOT the horizon
    # the VaR backtest was validated on.
    test_df = df[df["time_idx"] >= (max_t - config.BACKTEST_DAYS - encoder_len)].copy()
    eval_dataset = TimeSeriesDataSet.from_parameters(
        tft.dataset_parameters,
        test_df,
        predict=False,
        stop_randomization=True,
    )
    return eval_dataset.to_dataloader(batch_size=64, shuffle=False, num_workers=0)


def run_multi_seed_explainability():
    print("===== AGGREGATING VSN & ATTENTION ACROSS SEEDS =====")

    master_path = "master_df.csv" if os.path.exists("master_df.csv") else os.path.join(OUTPUT_DIR, "master_df.csv")
    df = pd.read_csv(master_path)
    df["ticker"] = df["ticker"].astype(str)

    seeds = config.VALIDATION_SEEDS
    vsn_records = []       # one row per (seed, feature)
    attn_records = []      # one row per (seed, lag)
    processed_seeds = []

    for seed in seeds:
        ckpt_path = _locate_seed_checkpoint(seed)
        if ckpt_path is None:
            print(f"[SKIP] No checkpoint found for seed {seed}.")
            continue

        print(f"[MODEL] Loading checkpoint for seed {seed}: {ckpt_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location=device)
        tft.to(device)
        tft.eval()

        encoder_len = int(getattr(tft, "max_encoder_length", config.ENCODER_LENGTH))
        dataloader = _build_eval_dataloader(tft, df, encoder_len)

        with torch.no_grad():
            raw_preds = tft.predict(dataloader, mode="raw", return_x=True)
            if hasattr(raw_preds, "output"):
                out = raw_preds.output
            elif isinstance(raw_preds, (tuple, list)):
                out = raw_preds[0]
            else:
                out = raw_preds
            interpretation = tft.interpret_output(out, reduction="mean")

        # VSN feature importance
        vsn_pct = _extract_vsn_pct(interpretation, tft)
        for feat, p in vsn_pct.items():
            vsn_records.append({"Seed": seed, "Feature": feat, "Percentage": p})

        # Temporal attention distribution
        lag_weights = _extract_attention(interpretation)
        for lag_idx, w in enumerate(lag_weights, start=1):
            attn_records.append({"Seed": seed, "Lag": lag_idx, "Weight": float(w)})

        processed_seeds.append(seed)

    if not processed_seeds:
        raise FileNotFoundError(
            f"[ERROR] No seed checkpoints found under patterns containing "
            f"'seed<num>' in {OUTPUT_DIR} or workspace. Run main.py first."
        )

    # ---- Aggregate VSN across seeds (Mean +/- Std) -------------------------
    vsn_df = pd.DataFrame(vsn_records)
    vsn_agg = (
        vsn_df.groupby("Feature")["Percentage"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    vsn_agg["std"] = vsn_agg["std"].fillna(0.0)
    vsn_agg = vsn_agg.sort_values(by="mean", ascending=False)

    vsn_agg.to_csv(os.path.join(OUTPUT_DIR, "vsn_feature_importance_seed_aggregated.csv"), index=False)
    vsn_agg.to_csv("vsn_feature_importance_seed_aggregated.csv", index=False)

    # ---- Aggregate temporal attention across seeds -------------------------
    attn_df = pd.DataFrame(attn_records)
    attn_agg = (
        attn_df.groupby("Lag")["Weight"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values("Lag")
    )
    attn_agg["std"] = attn_agg["std"].fillna(0.0)

    attn_agg.to_csv(os.path.join(OUTPUT_DIR, "temporal_attention_distribution_seed_aggregated.csv"), index=False)
    attn_agg.to_csv("temporal_attention_distribution_seed_aggregated.csv", index=False)

    # ---- Write multi-seed interpretability report --------------------------
    report_path = os.path.join(OUTPUT_DIR, "tft_explainability_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("      TFT INTERPRETABILITY: CROSS-SEED AGGREGATION (Mean +/- Std)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Seeds aggregated: {processed_seeds}\n\n")

        f.write("1. VARIABLE SELECTION NETWORK (VSN) FEATURE ATTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Feature':<22} | {'Importance (% mean+/-std)':<26} | {'Category'}\n")
        f.write("-" * 80 + "\n")
        for _, r in vsn_agg.iterrows():
            f.write(f"{r['Feature']:<22} | {r['mean']:>10.2f} +/- {r['std']:>5.2f}% | "
                    f"{CATEGORY_MAP.get(r['Feature'], 'Exogenous')}\n")

        f.write("\n\n2. MULTI-HEAD TEMPORAL ATTENTION (top-10 lags)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Lookback Lag':<20} | {'Attention Weight (% mean+/-std)'}\n")
        f.write("-" * 80 + "\n")
        for _, r in attn_agg.head(10).iterrows():
            f.write(f"Lag {int(r['Lag']):>2} day(s) prior        | {100 * r['mean']:>8.2f} +/- {100 * r['std']:>5.2f}%\n")

        f.write("=" * 80 + "\n")
        f.write("NOTE: Low cross-seed std indicates the network's attention\n")
        f.write("structure is a robust structural invariant, not an artifact of\n")
        f.write("a single random initialization.\n")

    print(f"[SUCCESS] Multi-seed explainability report saved to: {report_path}")
    print(f"[SUCCESS] VSN aggregation rows: {len(vsn_agg)}, Attention rows: {len(attn_agg)}")


def run_explainability():
    """Backward-compatible entry point: delegates to multi-seed aggregation."""
    run_multi_seed_explainability()


if __name__ == "__main__":
    run_multi_seed_explainability()
