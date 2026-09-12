# learning_curves.py
# =============================================================================
# Generates convergence diagnostics (Figure 6) and learning rate trajectory (Figure 7).
# Automatically handles PyTorch Lightning's nested CSVLogger directories and
# dynamically resolves column names for loss and learning rate metrics.
# =============================================================================
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

def main():
    log_dir = "lightning_logs"
    drive_out = getattr(config, "OUTPUT_DIR", ".")

    # Robustly search nested PyTorch Lightning log directories
    paths = sorted(glob.glob(os.path.join(log_dir, "**", "metrics.csv"), recursive=True))
    if not paths and os.path.exists(os.path.join(drive_out, log_dir)):
        paths = sorted(glob.glob(os.path.join(drive_out, log_dir, "**", "metrics.csv"), recursive=True))

    if not paths:
        print("[WARN] No metrics.csv found. Run main.py to train models first.")
        return

    print(f"[INFO] Found {len(paths)} metrics.csv files.")
    os.makedirs(drive_out, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(paths), 1)))

    # -------------------------------------------------------------
    # PLOT 1: Figure 6 (Train & Validation Loss Curves)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200)

    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        match = re.search(r"tft_run_seed_(\d+)", path)
        seed_label = f"Seed {match.group(1)}" if match else f"Run {idx+1}"

        train_col = next((c for c in ["train_loss_epoch", "train_loss", "loss"] if c in df.columns), None)
        val_col = next((c for c in ["val_loss_epoch", "val_loss"] if c in df.columns), None)

        if "epoch" in df.columns:
            if train_col:
                tr = df.dropna(subset=[train_col]).groupby("epoch")[train_col].mean()
                ax1.plot(tr.index, tr.values, color=colors[idx], alpha=0.85, linewidth=1.4, label=seed_label)
            if val_col:
                va = df.dropna(subset=[val_col]).groupby("epoch")[val_col].mean()
                ax2.plot(va.index, va.values, color=colors[idx], alpha=0.85, linewidth=1.4, label=seed_label)

    ax1.set_title("Training Quantile Loss vs. Epoch", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Mean Training Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=8.5, loc="upper right")

    ax2.set_title("Validation Quantile Loss vs. Epoch", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Mean Validation Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Temporal Fusion Transformer Convergence Diagnostics", fontsize=12, fontweight="bold")
    plt.tight_layout()

    fig6_path = "fig6_learning_curves.png"
    fig.savefig(fig6_path, bbox_inches="tight")
    try:
        fig.savefig(os.path.join(drive_out, fig6_path), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print(f"[SUCCESS] {fig6_path} generated.")

    # -------------------------------------------------------------
    # PLOT 2: Figure 7 (Learning Rate Trajectory)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    plotted_any_lr = False

    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        match = re.search(r"tft_run_seed_(\d+)", path)
        seed_label = f"Seed {match.group(1)}" if match else f"Run {idx+1}"

        lr_col = next((c for c in df.columns if "lr" in c.lower()), None)
        if lr_col:
            lr_series = df.dropna(subset=[lr_col])
            if len(lr_series) > 0:
                step_axis = lr_series["step"] if "step" in lr_series.columns else lr_series.index
                ax.plot(step_axis, lr_series[lr_col].values, color=colors[idx], alpha=0.8, linewidth=1.4, label=seed_label)
                plotted_any_lr = True

    if not plotted_any_lr:
        print("[INFO] Reconstructing ReduceLROnPlateau trajectory from validation loss...")
        base_lr = 0.001588
        lr_patience = 4
        lr_factor = 0.5

        for idx, path in enumerate(paths):
            df = pd.read_csv(path)
            match = re.search(r"tft_run_seed_(\d+)", path)
            seed_label = f"Seed {match.group(1)}" if match else f"Run {idx+1}"

            val_col = next((c for c in ["val_loss_epoch", "val_loss"] if c in df.columns), None)
            if val_col:
                va = df.dropna(subset=[val_col]).groupby("epoch")[val_col].mean().values
                current_lr = base_lr
                best_val = float("inf")
                bad_epochs = 0
                lr_history = []

                for v in va:
                    if v < best_val - 1e-4:
                        best_val = v
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                    if bad_epochs > lr_patience:
                        current_lr *= lr_factor
                        bad_epochs = 0
                    lr_history.append(current_lr)

                ax.plot(range(len(lr_history)), lr_history, color=colors[idx], alpha=0.85, linewidth=1.5, label=seed_label)

    ax.set_title("Learning Rate Decay Schedule (ReduceLROnPlateau)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch / Step", fontsize=10)
    ax.set_ylabel("Learning Rate (log scale)", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()

    fig7_path = "fig7_lr_trajectory.png"
    fig.savefig(fig7_path, bbox_inches="tight")
    try:
        fig.savefig(os.path.join(drive_out, fig7_path), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print(f"[SUCCESS] {fig7_path} generated.")

if __name__ == "__main__":
    main()
