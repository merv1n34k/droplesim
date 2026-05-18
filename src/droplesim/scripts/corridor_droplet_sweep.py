"""Run droplet corridor scenarios and write VF figures.

Usage:
    uv run python -m droplesim.scripts.corridor_droplet_sweep
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from droplesim.solver.geometry2d import BC_OUTLET, BCSpec, Geometry2D, build_sparse_maps
from droplesim.solver.sim import PhysParams, TwoPhaseSim


@dataclass(frozen=True)
class CorridorCase:
    name: str
    shape: tuple[int, int]
    points: tuple[tuple[float, float], ...]
    pressure_mbar: float = 1000.0


CASES = (
    CorridorCase(
        name="straight",
        shape=(56, 180),
        points=((14.0, 28.0), (166.0, 28.0)),
    ),
    CorridorCase(
        name="l_junction",
        shape=(136, 136),
        points=((18.0, 32.0), (92.0, 32.0), (92.0, 112.0), (122.0, 112.0)),
    ),
    CorridorCase(
        name="zig_zag",
        shape=(128, 360),
        points=(
            (40.0, 14.0), (40.0, 82.0), (58.0, 104.0),
            (88.0, 104.0), (112.0, 86.0), (142.0, 86.0),
            (166.0, 104.0), (196.0, 104.0), (220.0, 86.0),
            (250.0, 86.0), (274.0, 104.0), (304.0, 104.0),
            (326.0, 86.0), (342.0, 86.0),
        ),
    ),
)


def _segment_distance(xx: np.ndarray, yy: np.ndarray, p0, p1) -> np.ndarray:
    x0, y0 = p0
    x1, y1 = p1
    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy
    t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / denom, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    return np.hypot(xx - px, yy - py)


def _path_lengths(points: tuple[tuple[float, float], ...]) -> np.ndarray:
    lengths = [0.0]
    for p0, p1 in zip(points[:-1], points[1:]):
        lengths.append(lengths[-1] + float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
    return np.array(lengths)


def _point_at_fraction(points: tuple[tuple[float, float], ...], fraction: float) -> tuple[float, float]:
    lengths = _path_lengths(points)
    target = float(np.clip(fraction, 0.0, 1.0)) * lengths[-1]
    idx = int(np.searchsorted(lengths, target, side="right") - 1)
    idx = min(idx, len(points) - 2)
    p0 = np.array(points[idx])
    p1 = np.array(points[idx + 1])
    seg_len = lengths[idx + 1] - lengths[idx]
    alpha = 0.0 if seg_len == 0.0 else (target - lengths[idx]) / seg_len
    point = p0 + alpha * (p1 - p0)
    return float(point[0]), float(point[1])


def _cross_mask(shape: tuple[int, int], point: tuple[float, float], radius: float = 2.0) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    return np.hypot(xx - point[0], yy - point[1]) <= radius


def build_corridor(case: CorridorCase, width: float = 18.0) -> Geometry2D:
    ny, nx = case.shape
    yy, xx = np.mgrid[:ny, :nx]
    dist = np.full(case.shape, np.inf, dtype=np.float64)
    for p0, p1 in zip(case.points[:-1], case.points[1:]):
        dist = np.minimum(dist, _segment_distance(xx, yy, p0, p1))

    fluid = dist <= width / 2.0
    solid_mask = ~fluid
    bc_map = np.zeros(case.shape, dtype=np.uint8)

    inlet_mask = _cross_mask(case.shape, case.points[0], radius=width / 2.0) & fluid
    outlet_mask = _cross_mask(case.shape, case.points[-1], radius=width / 2.0) & fluid
    bc_map[inlet_mask] = 1
    bc_map[outlet_mask] = BC_OUTLET

    inlet = BCSpec("oil_inlet", "inlet", 0.0, 0.0, 1.0, 1.0,
                   phi=1.0, pressure_mbar=case.pressure_mbar)
    inlet.type_id = 1
    outlet = BCSpec("pressure_outlet", "outlet", 0.0, 0.0, 1.0, 1.0,
                    outlet_bc="pressure")
    outlet.type_id = BC_OUTLET
    return Geometry2D(
        solid_mask=solid_mask,
        bc_map=bc_map,
        specs=[inlet, outlet],
        dx_um=2.5,
        origin_um=(0.0, 0.0),
        sparse=build_sparse_maps(solid_mask, bc_map),
    )


def seed_droplets(geom: Geometry2D, points: tuple[tuple[float, float], ...],
                  radius: float = 5.0, interface: float = 1.15) -> np.ndarray:
    ny, nx = geom.shape
    yy, xx = np.mgrid[:ny, :nx]
    phi = np.ones((ny, nx), dtype=np.float64)
    for fraction in (0.08, 0.23, 0.38):
        cx, cy = _point_at_fraction(points, fraction)
        dist = np.hypot(xx - cx, yy - cy)
        aqueous = 0.5 * (1.0 - np.tanh((dist - radius) / interface))
        phi = np.minimum(phi, 1.0 - aqueous)
    phi[geom.solid_mask] = 1.0
    return phi


def dense_phi(sim: TwoPhaseSim, phi) -> np.ndarray:
    out = np.ones(sim.geom.shape, dtype=np.float64)
    out[sim.geom.solid_mask] = np.nan
    out[np.asarray(sim.fluid_y), np.asarray(sim.fluid_x)] = np.asarray(phi)
    return out


def aqueous_vf(phi_dense: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(phi_dense) if mask is None else (mask & np.isfinite(phi_dense))
    if not valid.any():
        return 0.0
    return float(np.mean(1.0 - phi_dense[valid]))


def run_case(
    case: CorridorCase,
    out_dir: Path,
    steps: int,
    emit_interval: int,
    delta_rho_max: float,
    dpi: int,
) -> dict:
    geom = build_corridor(case)
    phys = PhysParams(
        mu_c=1.24e-3,
        mu_d=1.2e-3,
        rho_c=1614.0,
        rho_d=1015.0,
        sigma=6e-3,
        contact_angle_deg=150.0,
        D_s=2e-10,
        D_bulk=4e-10,
        psi_inf=3e-6,
        E0=0.22,
        k_a=0.2,
        k_d=0.01,
        C_inlet=1e-3,
        sigma_floor=2e-3,
        surfactant_initial_coverage=0.85,
    )
    sim = TwoPhaseSim(geom, phys, delta_rho_max=delta_rho_max)
    f, phi, psi, C = sim.init_state(phi_init=seed_droplets(geom, case.points))

    snapshot_steps = {0, steps // 3, 2 * steps // 3, steps}
    snapshots = {0: dense_phi(sim, phi)}
    cross_masks = [
        _cross_mask(geom.shape, _point_at_fraction(case.points, fraction), radius=5.0)
        & ~geom.solid_mask
        for fraction in (0.25, 0.45, 0.65, 0.85)
    ]
    vf_rows = []

    for step in range(1, steps + 1):
        f, phi, psi, C = sim.step(f, phi, psi, C)
        if step % emit_interval == 0 or step in snapshot_steps:
            frame = dense_phi(sim, phi)
            row = {"step": step, "vf_total": aqueous_vf(frame)}
            for idx, mask in enumerate(cross_masks, start=1):
                row[f"vf_cross_{idx}"] = aqueous_vf(frame, mask)
            vf_rows.append(row)
            if step in snapshot_steps:
                snapshots[step] = frame

    out_dir.mkdir(parents=True, exist_ok=True)
    vf_panel = out_dir / f"{case.name}_vf.png"
    vf_csv = out_dir / f"{case.name}_vf.csv"
    _write_vf_panels(case, geom, snapshots, vf_panel, dpi)
    np.savetxt(
        vf_csv,
        np.array([[row["step"], row["vf_total"], row["vf_cross_1"], row["vf_cross_2"],
                   row["vf_cross_3"], row["vf_cross_4"]] for row in vf_rows]),
        delimiter=",",
        header="step,vf_total,vf_cross_1,vf_cross_2,vf_cross_3,vf_cross_4",
        comments="",
    )
    return {
        "name": case.name,
        "steps": steps,
        "pressure_mbar": case.pressure_mbar,
        "pressure_scale": sim.units.pressure_scale,
        "surfactant_initial_coverage": phys.surfactant_initial_coverage,
        "vf_rows": vf_rows,
        "outputs": {
            "vf_panel": str(vf_panel),
            "vf_csv": str(vf_csv),
        },
    }


def _vf_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "oil_water_vf",
        ["#ffffff", "#d8f1ff", "#9edcff", "#63bdff"],
    )


def _write_vf_panels(case: CorridorCase, geom: Geometry2D, snapshots: dict[int, np.ndarray],
                     path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.0), constrained_layout=True)
    cmap = _vf_cmap()
    cmap.set_bad("#333333")
    for ax, step in zip(axes, sorted(snapshots)):
        vf = 1.0 - snapshots[step]
        ax.imshow(vf, vmin=0.0, vmax=1.0, cmap=cmap, origin="lower")
        ax.contour(vf, levels=[0.5], colors="#1b75bb", linewidths=0.8,
                   origin="lower")
        ax.contour(~geom.solid_mask, levels=[0.5], colors="#222222", linewidths=0.5,
                   origin="lower")
        ax.set_title(f"{case.name}  step {step}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("VF = 1 - phi   |   oil = white, water = light blue", fontsize=16)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _write_combined_vf_graph(summaries: list[dict], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for summary in summaries:
        rows = summary["vf_rows"]
        steps = np.array([row["step"] for row in rows])
        ax.plot(steps, [row["vf_total"] for row in rows], label=summary["name"], lw=2.0)
        for idx, alpha in zip(range(1, 5), (0.24, 0.28, 0.32, 0.36)):
            ax.plot(
                steps,
                [row[f"vf_cross_{idx}"] for row in rows],
                color=ax.lines[-1].get_color(),
                alpha=alpha,
                lw=1.0,
            )
    ax.set_title("Aqueous volume fraction through corridors")
    ax.set_xlabel("Step")
    ax.set_ylabel("Aqueous VF")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run straight/L/zig-zag droplet corridor sweep")
    parser.add_argument("--out", default="results/corridor_sweep", help="Output directory")
    parser.add_argument("--steps", type=int, default=3000, help="Simulation steps per case")
    parser.add_argument("--emit-interval", type=int, default=20, help="VF sampling interval")
    parser.add_argument("--dpi", type=int, default=420, help="Output image DPI")
    parser.add_argument(
        "--delta-rho-max",
        type=float,
        default=0.05,
        help="LBM pressure cap for visualization; requested inlet pressure stays 1000 mbar",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        run_case(case, out_dir, args.steps, args.emit_interval, args.delta_rho_max, args.dpi)
        for case in CASES
    ]
    _write_combined_vf_graph(summary, out_dir / "vf_graph.png", args.dpi)
    clean_summary = [{k: v for k, v in item.items() if k != "vf_rows"} for item in summary]
    (out_dir / "summary.json").write_text(json.dumps(clean_summary, indent=2) + "\n")
    print(f"Wrote corridor sweep to {out_dir}")


if __name__ == "__main__":
    main()
