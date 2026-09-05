import os
import glob
import torch
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

OUTPUT_DIR = "/content/drive/MyDrive/GARCH_TFT_Results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_explainability():
    print("===== EXTRACTING TFT INTERPRETABILITY & ATTENTION WEIGHTS =====")

    # 1. Locate checkpoint
    candidate_ckpts = (
        glob.glob(os.path.join(OUTPUT_DIR, "*champion*.ckpt"))
        + glob.glob(os.path.join(OUTPUT_DIR, "*.ckpt"))
        + glob.glob("*champion*.ckpt")
        + glob.glob("*.ckpt")
    )
    if not candidate_ckpts:
        raise FileNotFoundError("[ERROR] No .ckpt checkpoint found in workspace or Drive.")

    ckpt_path = candidate_ckpts[0]
    print(f"[MODEL] Loading checkpoint: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location=device)
    tft.to(device)
    tft.eval()

    # 2. Load dataset
    master_path = "master_df.csv" if os.path.exists("master_df.csv") else os.path.join(OUTPUT_DIR, "master_df.csv")
    df = pd.read_csv(master_path)
    df["ticker"] = df["ticker"].astype(str)

    max_t = df["time_idx"].max()
    encoder_len = getattr(tft, "max_encoder_length", 30)
    test_df = df[df["time_idx"] >= (max_t - 500 - encoder_len)].copy()

    # 3. Build evaluation dataloader
    eval_dataset = TimeSeriesDataSet.from_parameters(
        tft.dataset_parameters,
        test_df,
        stop_randomization=True
    )
    dataloader = eval_dataset.to_dataloader(batch_size=64, shuffle=False, num_workers=0)

    # 4. Generate raw predictions for interpretation
    print("[INFERENCE] Extracting multi-head attention and VSN tensors across test horizon...")
    with torch.no_grad():
        raw_predictions = tft.predict(dataloader, mode="raw", return_x=True)
        if hasattr(raw_predictions, "output"):
            out = raw_predictions.output
        elif isinstance(raw_predictions, (tuple, list)):
            out = raw_predictions[0]
        else:
            out = raw_predictions

        interpretation = tft.interpret_output(out, reduction="mean")

    # 5. Extract Variable Selection Network (VSN) Feature Importance
    print("\n--- Variable Selection Network (VSN) Importance ---")
    enc_weights = interpretation["encoder_variables"].detach().cpu().numpy()
    if enc_weights.ndim > 1:
        enc_weights = enc_weights.mean(axis=tuple(range(enc_weights.ndim - 1)))
    enc_weights = enc_weights.flatten()

    var_names = list(tft.encoder_variables)
    min_len = min(len(var_names), len(enc_weights))
    var_names = var_names[:min_len]
    enc_weights = enc_weights[:min_len]

    total_w = np.sum(enc_weights)
    pct = (enc_weights / total_w) * 100 if total_w > 0 else np.ones_like(enc_weights) / len(enc_weights) * 100

    vsn_df = pd.DataFrame({
        "Feature": var_names,
        "Percentage": pct
    }).sort_values(by="Percentage", ascending=False)

    for _, row in vsn_df.iterrows():
        print(f"  - {row['Feature']:<22}: {row['Percentage']:.2f}%")

    vsn_df.to_csv("vsn_feature_importance.csv", index=False)
    vsn_df.to_csv(os.path.join(OUTPUT_DIR, "vsn_feature_importance.csv"), index=False)

    # 6. Extract Multi-Head Temporal Attention
    print("\n--- Multi-Head Temporal Attention Lookback Distribution ---")
    attn_weights = interpretation["attention"].detach().cpu().numpy()
    if attn_weights.ndim > 1:
        attn_weights = attn_weights.mean(axis=tuple(range(attn_weights.ndim - 1)))
    attn_weights = attn_weights.flatten()

    norm_attn = attn_weights / np.sum(attn_weights) if np.sum(attn_weights) > 0 else np.ones_like(attn_weights) / len(attn_weights)
    lag_weights = norm_attn[::-1]

    attn_df = pd.DataFrame({
        "Lookback_Lag_Days": np.arange(1, len(lag_weights) + 1),
        "Attention_Weight": lag_weights
    })

    for _, row in attn_df.head(5).iterrows():
        print(f"  - Lag {int(row['Lookback_Lag_Days'])} Day(s) Prior: {row['Attention_Weight']*100:.2f}%")

    attn_df.to_csv("temporal_attention_distribution.csv", index=False)
    attn_df.to_csv(os.path.join(OUTPUT_DIR, "temporal_attention_distribution.csv"), index=False)

    # 7. Write Dedicated Interpretability Report
    report_path = os.path.join(OUTPUT_DIR, "tft_explainability_report.txt")
    categories = {
        "Log_Ret_Lag1": "Autoregressive Shock",
        "Log_Ret_Lag2": "Autoregressive Shock",
        "GK_Vol": "Intraday Range Volatility",
        "GARCH_sigma": "Econometric Volatility Prior",
        "GARCH_resid": "Standardized Innovation",
        "relative_time_idx": "Temporal Indexing",
        "India_VIX_Diff": "Domestic Volatility (India VIX or RV proxy)",
        "US_VIX_Diff": "Cross-Border Macro Spillover",
        "time_idx": "Panel Timeline"
    }

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("        TEMPORAL FUSION TRANSFORMER: MODEL INTERPRETABILITY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. VARIABLE SELECTION NETWORK (VSN) FEATURE ATTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Feature Name':<25} | {'Importance Weight (%)':<25} | {'Category'}\n")
        f.write("-" * 80 + "\n")
        for _, r in vsn_df.iterrows():
            f.write(f"{r['Feature']:<25} | {r['Percentage']:>20.2f}% | {categories.get(r['Feature'], 'Exogenous')}\n")

        f.write("\n\n2. MULTI-HEAD TEMPORAL ATTENTION DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Historical Lookback Horizon':<30} | {'Attention Receptive Field Weight (%)'}\n")
        f.write("-" * 80 + "\n")
        for _, r in attn_df.head(10).iterrows():
            lag = int(r["Lookback_Lag_Days"])
            f.write(f"Lag {lag:>2} Day(s) Prior (t - {lag:>2})          | {r['Attention_Weight']*100:>30.2f}%\n")
        f.write("=" * 80 + "\n")

    print(f"\n[SUCCESS] Master interpretability report written to: {report_path}")

if __name__ == "__main__":
    run_explainability()
