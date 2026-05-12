"""
classify_regime.py — VF signal → drip / transition / jet label.

Reads volume fraction (VF) signal extracted at the orifice cross-section
and classifies the flow regime based on:
  - Dripping   : regular oscillations, low CV (<20%), high autocorrelation
  - Transition : irregular oscillations, moderate CV (20-25%)
  - Jetting    : jet present, high CV (>25%) or near-DC signal

The binary filter test (from plan):
  L_break < 1060 µm (straight section length) → Dripping
  L_break ≥ 1060 µm                           → Jetting

Usage:
    uv run python scripts/classify_regime.py --vf results/T1_07/vf_signal.csv
    uv run python scripts/classify_regime.py --vf results/T1_07/vf_signal.csv --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Regime = Literal["Dripping", "Transition", "Jetting", "Unknown"]

STRAIGHT_LENGTH_UM = 1060.0   # µm — binary filter breakup length threshold
DX_UM = 2.5


# ── Signal loading ────────────────────────────────────────────────────────────

def load_vf_signal(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load VF(t) time series.  Returns (time_steps, vf_values)."""
    df = pd.read_csv(csv_path)
    steps = df["step"].to_numpy()
    vf    = df["vf"].to_numpy()
    return steps, vf


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_vf_features(vf: np.ndarray, dt_s: float = 1.4e-7) -> dict:
    """
    Compute CV, dominant frequency, peak-to-peak amplitude, autocorrelation lag.
    """
    mean = vf.mean()
    std  = vf.std()
    cv   = (std / mean * 100.0) if mean > 0 else 0.0

    # Dominant frequency via FFT
    n = len(vf)
    fft = np.abs(np.fft.rfft(vf - mean))
    freqs = np.fft.rfftfreq(n, d=dt_s)
    if len(fft) > 1:
        dominant_idx = np.argmax(fft[1:]) + 1
        dominant_freq_Hz = float(freqs[dominant_idx])
    else:
        dominant_freq_Hz = 0.0

    # Autocorrelation at lag=1 (regularity metric)
    if n > 2:
        ac_lag1 = float(np.corrcoef(vf[:-1], vf[1:])[0, 1])
    else:
        ac_lag1 = 0.0

    return {
        "mean_vf": float(mean),
        "std_vf":  float(std),
        "cv_pct":  float(cv),
        "dominant_freq_Hz": dominant_freq_Hz,
        "ac_lag1": ac_lag1,
        "n_points": n,
    }


def estimate_break_length(
    droplets_csv: Path | None,
    dx_um: float = DX_UM,
) -> float | None:
    """
    Estimate L_break from droplet centroid data (nodes → µm).
    Returns None if data unavailable.
    """
    if droplets_csv is None or not droplets_csv.exists():
        return None

    df = pd.read_csv(droplets_csv)
    if "bbox_xmax" not in df.columns:
        return None

    # L_break = downstream extent of last connected aqueous structure (jet tip)
    # Proxy: max bbox_xmax of droplets/jet fragments
    max_x = df["bbox_xmax"].max()
    return float(max_x * dx_um)


# ── Classification ────────────────────────────────────────────────────────────

def classify(
    features: dict,
    l_break_um: float | None = None,
) -> tuple[Regime, dict]:
    """
    Multi-criterion regime classification.

    Returns (regime_label, evidence_dict).
    """
    cv = features["cv_pct"]
    evidence: dict = {"cv_pct": cv}

    # Binary filter (primary criterion when L_break available)
    if l_break_um is not None:
        evidence["l_break_um"] = l_break_um
        evidence["l_break_threshold_um"] = STRAIGHT_LENGTH_UM
        if l_break_um < STRAIGHT_LENGTH_UM:
            evidence["binary_filter"] = "Dripping"
        else:
            evidence["binary_filter"] = "Jetting"

    # CV-based classification (fallback / secondary)
    if cv < 20.0:
        cv_regime: Regime = "Dripping"
    elif cv < 25.0:
        cv_regime = "Transition"
    else:
        cv_regime = "Jetting"
    evidence["cv_regime"] = cv_regime

    # Dominant regime: prefer binary filter if available
    if l_break_um is not None:
        regime: Regime = evidence["binary_filter"]   # type: ignore[assignment]
        # Upgrade to Transition if CV disagrees significantly
        if regime == "Dripping" and cv > 22.0:
            regime = "Transition"
            evidence["override"] = "CV > 22% despite short break length"
    else:
        regime = cv_regime

    return regime, evidence


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Classify flow regime from VF signal")
    parser.add_argument("--vf", required=True, help="Path to vf_signal.csv")
    parser.add_argument("--droplets", default="", help="Path to droplets.csv (for L_break)")
    parser.add_argument("--dt-s", type=float, default=1.4e-7, help="Physical dt in seconds")
    parser.add_argument("--out", default="", help="Output JSON path")
    parser.add_argument("--plot", action="store_true", help="Save VF signal plot")
    args = parser.parse_args()

    vf_path = Path(args.vf)
    if not vf_path.exists():
        print(f"ERROR: {vf_path} not found")
        return

    steps, vf = load_vf_signal(vf_path)
    features = compute_vf_features(vf, dt_s=args.dt_s)

    droplets_csv = Path(args.droplets) if args.droplets else vf_path.parent / "droplets.csv"
    l_break = estimate_break_length(droplets_csv)

    regime, evidence = classify(features, l_break)

    print(f"Run: {vf_path.parent.name}")
    print(f"  CV        : {features['cv_pct']:.1f}%")
    print(f"  Freq      : {features['dominant_freq_Hz']:.1f} Hz")
    print(f"  L_break   : {l_break:.0f} µm" if l_break else "  L_break   : N/A")
    print(f"  Regime    : {regime}")
    print(f"  Evidence  : {evidence}")

    if args.out:
        import json
        result = {"regime": regime, "features": features, "evidence": evidence}
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"Wrote → {args.out}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 3))
            t_ms = steps * args.dt_s * 1000
            ax.plot(t_ms, vf, lw=0.8)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("VF at orifice")
            ax.set_title(f"{vf_path.parent.name}  →  {regime}  (CV={features['cv_pct']:.1f}%)")
            out_png = vf_path.with_suffix(".png")
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            print(f"Plot → {out_png}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
