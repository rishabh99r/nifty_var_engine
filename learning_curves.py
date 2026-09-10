# learning_curves.py
# =============================================================================
# RD6: Figure 6 (Learning Curves) + Figure 7 (LR Trajectory).
#
# Reads the per-seed Lightning CSVLogger outputs written by train_tft()
# (lightning_logs/tft_run_seed_*/metrics.csv) and produces:
#   - fig6_learning_curves.png : train_loss & val_loss vs epoch (all seeds)
#   - fig7_lr_trajectory.png   : learning rate vs step (per seed)
#
# Also adds these figures to config.OUTPUT_DIR (GARCH_TFT_Results) so they
# ship with the publication set.
#
# Usage:
#   python learning_curves.py [--log_root lightning_logs]
# =============================================================================
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

OUTPUT_DIR = config.OUTPUT_DIR + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_metrics_csv(path):
    """Read a Lightning metrics.csv; returns DataFrame or None."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


def _lr_column(df):
    """Find the learning-rate column (name varies: 'lr-Adam', 'lr-AdamW', ...)."""
    for c in df.columns:
        if c.lower().startswith("lr"):
            return c
    return None


def plot_learning_curves(log_root="lightning_logs"):
    """Figure 6: train/val loss vs epoch, one line per seed run."""
    paths = sorted(glob.glob(os.path.join(log_root, "tft_run_seed_*", "metrics.csv")))
    if not paths:
        print("[WARN] No CSVLogger metrics found under", log_root)
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(paths)))

    for path, color in zip(paths, colors):
        df = _load_metrics_csv(path)
        if df is None or "val_loss" not in df.columns:
            continue
        # Aggregate per-epoch (CSVLogger may log train/val per batch or epoch).
        tr = df.groupby("epoch")["train_loss"].mean() if "train_loss" in df.columns else None
        va = df.groupby("epoch")["val_loss"].mean()
        label = os.path.basename(os.path.dirname(path)).replace("tft_run_seed_", "seed ")
        if tr is not None:
            ax1.plot(tr.index, tr.values, color=color, alpha=0.7, linewidth=1.2)
        ax2.plot(va.index, va.values, color=color, alpha=0.7, linewidth=1.2, label=label)

    ax1.set_title("Training Loss vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("train_loss")
    ax2.set_title("Validation Loss vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("val_loss")
    ax2.legend(fontsize=8)
    fig.tight_layout()

    out_local = "fig6_learning_curves.png"
    fig.savefig(out_local, dpi=200, bbox_inches="tight")
    try:
        fig.savefig(os.path.join(OUTPUT_DIR, out_local), dpi=200, bbox_inches="tight")
    except Exception as e:
        print(f"  [WARN] Could not save Fig6 to OUTPUT_DIR: {e}")
    plt.close(fig)
    print(f"[SUCCESS] Figure 6 (learning curves) -> {out_local}")
    return out_local


def plot_lr_trajectory(log_root="lightning_logs"):
    """Figure 7: learning rate vs training step, one line per seed run."""
    paths = sorted(glob.glob(os.path.join(log_root, "tft_run_seed_*", "metrics.csv")))
    if not paths:
        print("[WARN] No CSVLogger metrics found under", log_root)
        return None

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = plt.cm.plasma(np.linspace(0, 1, len(paths)))

    for path, color in zip(paths, colors):
        df = _load_metrics_csv(path)
        if df is None:
            continue
        lr_col = _lr_column(df)
        if lr_col is None:
            continue
        step_col = "step" if "step" in df.columns else df.index
        ax.plot(step_col, df[lr_col].values, color=color, alpha=0.75,
                linewidth=1.3, label=os.path.basename(os.path.dirname(path))
                .replace("tft_run_seed_", "seed "))

    ax.set_title("Learning-Rate Trajectory (ReduceLROnPlateau)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_local = "fig7_lr_trajectory.png"
    fig.savefig(out_local, dpi=200, bbox_inches="tight")
    try:
        fig.savefig(os.path.join(OUTPUT_DIR, out_local), dpi=200, bbox_inches="tight")
    except Exception as e:
        print(f"  [WARN] Could not save Fig7 to OUTPUT_DIR: {e}")
    plt.close(fig)
    print(f"[SUCCESS] Figure 7 (LR trajectory) -> {out_local}")
    return out_local


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Learning-curve / LR-trajectory figures.")
    ap.add_argument("--log_root", default="lightning_logs")
    args = ap.parse_args()
    plot_learning_curves(args.log_root)
    plot_lr_trajectory(args.log_root)
