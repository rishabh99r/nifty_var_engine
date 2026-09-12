# render_tables.py
# =============================================================================
# T4: Manuscript table rendering.
#
# Reads the tournament CSVs produced by ablation_runner.py and
# specialist_runner.py and renders them as Markdown tables for the manuscript:
#   - ablation_tournament_results.csv            -> Table 10 (6-arm tournament)
#   - specialist_vs_aggregate_results_seed42.csv -> Specialist vs Aggregate
#
# Usage:  python render_tables.py [--ablation ablation_tournament_results.csv]
#                                  [--specialist specialist_vs_aggregate_results_seed42.csv]
# Output: prints Markdown to stdout and writes manuscript_tables.md
# =============================================================================
import argparse
import os

import pandas as pd


def _fmt(v):
    """NaN-safe formatting for Markdown cells."""
    if v is None:
        return "N/A"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "N/A"
    if abs(f) >= 100:
        return f"{f:.0f}"
    if abs(f) >= 1:
        return f"{f:.3f}"
    return f"{f:.4f}"


def df_to_markdown(df, caption=None):
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    lines = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    cols = list(df.columns)
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        cells = [_fmt(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_ablation_table(path):
    """Table 10: 6-arm ablation tournament (ensemble-level, per asset)."""
    if not os.path.exists(path):
        print(f"[WARN] {path} not found -- run ablation_runner.py first.")
        return ""
    df = pd.read_csv(path)
    # Keep the most informative columns, in a stable order.
    keep = [c for c in ["Model Variant", "Asset", "Breaches", "Total Obs",
                        "Kupiec p-value", "DM Stat (neg=TFT)", "DM p-value",
                        "Mean Loss Diff"] if c in df.columns]
    if not keep:
        keep = list(df.columns)
    return df_to_markdown(df[keep], caption="Table 10. Six-arm ablation tournament "
                                            "(5-seed ensemble, OOS 1% VaR vs GJR-GARCH)")


def render_specialist_table(path):
    """Specialist vs Aggregate vs GJR-GARCH comparison."""
    if not os.path.exists(path):
        print(f"[WARN] {path} not found -- run specialist_runner.py first.")
        return ""
    df = pd.read_csv(path)
    keep = [c for c in ["Model", "Asset", "Breaches", "Kupiec p",
                        "DM Stat (neg=TFT)", "DM p", "Mean Loss Diff"] if c in df.columns]
    if not keep:
        keep = list(df.columns)
    return df_to_markdown(df[keep], caption="Specialist (q0.01-only) vs Aggregate "
                                            "(q0.01/0.50/0.99) vs GJR-GARCH")


def main():
    ap = argparse.ArgumentParser(description="Render tournament CSVs as Markdown tables.")
    ap.add_argument("--ablation", default="ablation_tournament_results.csv")
    ap.add_argument("--specialist", default="specialist_vs_aggregate_results_seed42.csv")
    ap.add_argument("--out", default="manuscript_tables.md")
    args = ap.parse_args()

    blocks = []
    blocks.append(render_ablation_table(args.ablation))
    blocks.append(render_specialist_table(args.specialist))
    md = "\n".join(b for b in blocks if b)

    if md.strip():
        with open(args.out, "w") as f:
            f.write(md)
        print(md)
        print(f"\n[SUCCESS] Markdown tables written to {args.out}")
    else:
        print("[INFO] No tables rendered (input CSVs missing).")


if __name__ == "__main__":
    main()