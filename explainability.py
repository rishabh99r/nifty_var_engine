# explainability.py
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer

def export_model_explainability(tft_checkpoint_path, test_dataloader, output_dir="."):
    print("\n[EXPLAINABILITY] === Extracting VSN Weights & Temporal Attention Matrices ===")

    tft = TemporalFusionTransformer.load_from_checkpoint(tft_checkpoint_path)
    tft.eval()

    # Extract raw model interpretations across the test set
    raw_predictions, x = tft.predict(test_dataloader, mode="raw", return_x=True)
    interpretation = tft.interpret_output(raw_predictions, reduction="mean")

    # 1. Export Variable Selection Network (VSN) Feature Importance
    encoder_importance = interpretation["encoder_variables"].numpy()
    feature_names = tft.encoder_variables

    vsn_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Score": encoder_importance
    }).sort_values(by="Importance_Score", ascending=False)

    vsn_df["Percentage"] = (vsn_df["Importance_Score"] / vsn_df["Importance_Score"].sum()) * 100
    vsn_path = f"{output_dir}/vsn_feature_importance.csv"
    vsn_df.to_csv(vsn_path, index=False)

    print(f"[EXPLAINABILITY] VSN Importance successfully saved to {vsn_path}")
    print(vsn_df.to_string(index=False))

    # 2. Export Multi-Head Temporal Attention Lookback Distribution
    # Attention weights show how much importance the network places on lags t-60 to t-1
    attention_weights = interpretation["attention"].numpy() # Shape: [time_steps]
    lags = np.arange(-len(attention_weights), 0)

    attn_df = pd.DataFrame({
        "Lookback_Lag_Days": lags,
        "Attention_Weight": attention_weights
    })

    attn_path = f"{output_dir}/temporal_attention_distribution.csv"
    attn_df.to_csv(attn_path, index=False)
    print(f"[EXPLAINABILITY] Temporal Attention Lookback saved to {attn_path}")
    print(f"  -> Peak Attention Lag: {attn_df.loc[attn_df['Attention_Weight'].idxmax(), 'Lookback_Lag_Days']} Days prior to forecast.")

    return vsn_df, attn_df
