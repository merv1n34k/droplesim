"""
plot_regime_map.py — Ca-λ regime map figure.

Plots simulation and experimental data points in Ca-λ space,
coloured by regime (Dripping / Transition / Jetting), with the
estimated phase boundary overlaid.

Usage:
    uv run python scripts/plot_regime_map.py --summary results/doe_summary.csv
    uv run python scripts/plot_regime_map.py --summary results/doe_summary.csv \
        --boundary results/effects/regime_boundary.csv --out results/figures/regime_map.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# ── Colour / marker scheme ────────────────────────────────────────────────────

REGIME_STYLE: dict[str, dict] = {
    "Dripping":   {"color": "#2196F3", "marker": "o", "label": "Dripping"},
    "Transition": {"color": "#FF9800", "marker": "s", "label": "Transition"},
    "Jetting":    {"color": "#F44336", "marker": "^", "label": "Jetting"},
    "Unknown":    {"color": "#9E9E9E", "marker": "x", "label": "Unknown"},
}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_regime_map(
    df: pd.DataFrame,
    boundary_df: pd.DataFrame | None = None,
    out_path: Path | None = None,
    source_col: str = "source",
) -> None:
    """
    Plot Ca-λ regime map.

    df must have columns: Ca, viscosity_ratio_lambda, sim_regime (or expt_regime).
    source_col: column distinguishing 'Simulation' vs 'Experiment' (optional).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    regime_col = "sim_regime" if "sim_regime" in df.columns else "expt_regime"
    has_source = source_col in df.columns

    for regime, style in REGIME_STYLE.items():
        sub = df[df[regime_col] == regime]
        if sub.empty:
            continue

        if has_source:
            for src, marker in [("Simulation", "o"), ("Experiment", "D")]:
                s2 = sub[sub[source_col] == src]
                if s2.empty:
                    continue
                ax.scatter(
                    s2["Ca"], s2["viscosity_ratio_lambda"],
                    c=style["color"], marker=marker, s=80,
                    edgecolors="k", linewidths=0.5,
                    zorder=3,
                )
        else:
            ax.scatter(
                sub["Ca"], sub["viscosity_ratio_lambda"],
                c=style["color"], marker=style["marker"], s=90,
                edgecolors="k", linewidths=0.5, label=style["label"],
                zorder=3,
            )

        # Annotate run IDs if present
        if "id" in sub.columns:
            for _, row in sub.iterrows():
                ax.annotate(
                    str(row["id"]),
                    xy=(row["Ca"], row["viscosity_ratio_lambda"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6, color="#333333",
                )

    # Regime boundary
    if boundary_df is not None and not boundary_df.empty:
        bdf = boundary_df.sort_values("lambda")
        ax.plot(bdf["Ca_critical"], bdf["lambda"],
                "k--", lw=1.5, label="Drip/Jet boundary", zorder=2)

    ax.set_xlabel(r"Capillary number  $Ca = \mu_{oil} U_{oil} / \sigma$", fontsize=11)
    ax.set_ylabel(r"Viscosity ratio  $\lambda = \mu_{aq}/\mu_{oil}$", fontsize=11)
    ax.set_title("Flow regime map — Drop-seq flow-focusing generator", fontsize=11)

    # Legend: regimes
    regime_patches = [
        mpatches.Patch(color=s["color"], label=s["label"])
        for s in REGIME_STYLE.values()
        if not df[regime_col].empty
    ]
    if has_source:
        from matplotlib.lines import Line2D
        src_handles = [
            Line2D([0], [0], marker="o", color="k", fillstyle="none", label="Simulation", ms=7),
            Line2D([0], [0], marker="D", color="k", fillstyle="none", label="Experiment", ms=7),
        ]
        ax.legend(handles=regime_patches + src_handles, fontsize=8, loc="upper left")
    else:
        ax.legend(handles=regime_patches, fontsize=9, loc="upper left")

    ax.grid(True, alpha=0.3, ls=":")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ca-λ regime map")
    parser.add_argument("--summary", required=True, help="Path to doe_summary.csv")
    parser.add_argument("--boundary", default="", help="Path to regime_boundary.csv (optional)")
    parser.add_argument("--out", default="", help="Output figure path (pdf/png)")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        return

    df = pd.read_csv(summary_path)

    boundary_df = None
    if args.boundary:
        bp = Path(args.boundary)
        if bp.exists():
            boundary_df = pd.read_csv(bp)

    out_path = Path(args.out) if args.out else summary_path.parent / "figures" / "regime_map.pdf"

    plot_regime_map(df, boundary_df, out_path)


if __name__ == "__main__":
    main()
