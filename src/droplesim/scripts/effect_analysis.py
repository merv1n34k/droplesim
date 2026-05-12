"""
effect_analysis.py — ANOVA + response surface analysis on DoE results.

Reads simulation results (CSV with run metadata + outcomes) and:
  1. One-way ANOVA: decompose CV variance by Q_oil, Q_aq, λ, σ, θ
  2. Main effect plots (CV vs each factor)
  3. Regression response surface: CV = f(Q_oil, Q_aq, λ, σ)
  4. Regime boundary estimation in Ca-λ space

Usage:
    uv run python scripts/effect_analysis.py --results results/doe_summary.csv
    uv run python scripts/effect_analysis.py --results results/doe_summary.csv --out results/effects/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(csv_path: Path) -> pd.DataFrame:
    """
    Load DoE results CSV.  Expected columns (subset):
      run_id, Q_oil, Q_aq, ratio, lambda, sigma, theta, Ca,
      sim_cv_pct, sim_regime, expt_cv_pct, expt_regime
    """
    df = pd.read_csv(csv_path)
    return df


# ── ANOVA ─────────────────────────────────────────────────────────────────────

def one_way_anova(df: pd.DataFrame, factor: str, response: str = "sim_cv_pct") -> dict:
    """
    One-way ANOVA: test if `response` differs across levels of `factor`.
    Returns dict with F-statistic, p-value, eta-squared.
    """
    if factor not in df.columns or response not in df.columns:
        return {"factor": factor, "error": "column missing"}

    groups = [g[response].dropna().values for _, g in df.groupby(factor)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"factor": factor, "error": "insufficient groups"}

    f_stat, p_val = stats.f_oneway(*groups)

    # η² = SS_between / SS_total
    grand_mean = df[response].mean()
    ss_total   = ((df[response] - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "factor":    factor,
        "F_stat":    float(f_stat),
        "p_value":   float(p_val),
        "eta_sq":    float(eta_sq),
        "n_groups":  len(groups),
        "significant": bool(p_val < 0.05),
    }


def run_anova_suite(df: pd.DataFrame, response: str = "sim_cv_pct") -> pd.DataFrame:
    """Run ANOVA for all main factors.  Returns ranked summary DataFrame."""
    factors = ["Q_oil_uL_min", "Q_aq_uL_min", "viscosity_ratio_lambda",
               "sigma_mN_m", "contact_angle_deg"]
    factors = [f for f in factors if f in df.columns]

    results = [one_way_anova(df, f, response) for f in factors]
    result_df = pd.DataFrame(results)
    if "eta_sq" in result_df.columns:
        result_df = result_df.sort_values("eta_sq", ascending=False)
    return result_df


# ── Response surface ──────────────────────────────────────────────────────────

def fit_response_surface(df: pd.DataFrame, response: str = "sim_cv_pct") -> dict:
    """
    Fit quadratic response surface:
      CV = β0 + β1·Q_oil + β2·Q_aq + β3·λ + β4·σ + interactions + ε

    Returns coefficients and R².
    """
    features = ["Q_oil_uL_min", "Q_aq_uL_min", "viscosity_ratio_lambda", "sigma_mN_m"]
    available = [f for f in features if f in df.columns]
    if not available or response not in df.columns:
        return {"error": "insufficient columns"}

    subset = df[available + [response]].dropna()
    if len(subset) < len(available) + 2:
        return {"error": "insufficient data points"}

    X = subset[available].values
    y = subset[response].values

    # Normalize
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xn = (X - X_mean) / X_std

    # Add quadratic terms
    Xq = np.column_stack([Xn, Xn**2])
    Xq = np.column_stack([np.ones(len(Xq)), Xq])

    # Least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(Xq, y, rcond=None)
    y_pred = Xq @ coeffs
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "features": available,
        "coefficients": coeffs.tolist(),
        "R_squared": float(r_sq),
        "n_obs": int(len(y)),
    }


# ── Regime boundary estimation ────────────────────────────────────────────────

def estimate_regime_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find approximate Dripping/Jetting boundary in Ca-λ space.
    Returns DataFrame with boundary points.
    """
    if "Ca" not in df.columns or "viscosity_ratio_lambda" not in df.columns:
        return pd.DataFrame()
    if "sim_regime" not in df.columns and "expt_regime" not in df.columns:
        return pd.DataFrame()

    regime_col = "sim_regime" if "sim_regime" in df.columns else "expt_regime"
    df = df.copy()
    df["is_jetting"] = df[regime_col].isin(["Jetting", "Transition"])

    boundary = df.groupby("viscosity_ratio_lambda").apply(
        lambda g: g.sort_values("Ca")
        .query("is_jetting")
        ["Ca"].min() if g["is_jetting"].any() else float("nan")
    ).reset_index()
    boundary.columns = ["lambda", "Ca_critical"]
    boundary = boundary.dropna()

    return boundary


# ── Classification accuracy ───────────────────────────────────────────────────

def regime_accuracy(df: pd.DataFrame) -> dict:
    """Compare sim_regime to expt_regime.  Returns accuracy metrics."""
    if "sim_regime" not in df.columns or "expt_regime" not in df.columns:
        return {"error": "regime columns missing"}

    sub = df[["sim_regime", "expt_regime"]].dropna()
    correct = (sub["sim_regime"] == sub["expt_regime"]).sum()
    n = len(sub)
    return {
        "n_runs": int(n),
        "n_correct": int(correct),
        "accuracy": float(correct / n) if n > 0 else 0.0,
        "target": 7.0 / 9.0,
        "passes": bool(correct / n >= 7.0 / 9.0) if n > 0 else False,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ANOVA + response surface on DoE results")
    parser.add_argument("--results", required=True, help="Path to doe_summary.csv")
    parser.add_argument("--out", default="results/effects", help="Output directory")
    parser.add_argument("--response", default="sim_cv_pct", help="Response variable")
    args = parser.parse_args()

    csv_path = Path(args.results)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")

    # ANOVA
    print("\n=== ANOVA ===")
    anova_df = run_anova_suite(df, args.response)
    print(anova_df.to_string(index=False))
    anova_df.to_csv(out_dir / "anova.csv", index=False)

    # Response surface
    print("\n=== Response Surface ===")
    rs = fit_response_surface(df, args.response)
    if "error" not in rs:
        print(f"  R² = {rs['R_squared']:.3f}  (n={rs['n_obs']})")
        import json
        (out_dir / "response_surface.json").write_text(json.dumps(rs, indent=2))
    else:
        print(f"  {rs}")

    # Regime accuracy
    print("\n=== Regime Classification Accuracy ===")
    acc = regime_accuracy(df)
    if "error" not in acc:
        print(f"  {acc['n_correct']}/{acc['n_runs']}  accuracy={acc['accuracy']:.0%}  "
              f"passes_target={acc['passes']}")
    else:
        print(f"  {acc}")

    # Regime boundary
    boundary = estimate_regime_boundary(df)
    if not boundary.empty:
        boundary.to_csv(out_dir / "regime_boundary.csv", index=False)
        print(f"\nRegime boundary → {out_dir}/regime_boundary.csv")

    print(f"\nResults → {out_dir}/")


if __name__ == "__main__":
    main()
