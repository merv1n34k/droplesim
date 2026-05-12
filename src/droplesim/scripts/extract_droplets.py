"""
extract_droplets.py — Connected-component tracking from HDF5 φ snapshots.

For each snapshot in a run's results directory:
  1. Load φ field from HDF5
  2. Threshold at φ < 0.5 to identify aqueous droplets
  3. 3D connected-component labeling (scipy.ndimage)
  4. Extract per-droplet properties: volume, centroid, bounding box
  5. Track droplets across frames (nearest-centroid matching)
  6. Output: droplets.csv with columns: step, droplet_id, volume_pL, cx, cy, cz

Usage:
    uv run python scripts/extract_droplets.py --results results/T1_07
    uv run python scripts/extract_droplets.py --results results/T1_07 --phi-threshold 0.5
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DropletFrame:
    step: int
    droplet_id: int
    volume_pL: float
    cx: float          # centroid x [nodes]
    cy: float
    cz: float
    bbox_xmin: int
    bbox_xmax: int


# ── HDF5 loading ──────────────────────────────────────────────────────────────

def load_phi_snapshot(h5_path: Path, dx_um: float = 2.5) -> tuple[np.ndarray, int]:
    """
    Load φ field from a snapshot HDF5 file.
    Returns (phi_array, step_number).
    """
    with h5py.File(h5_path, "r") as f:
        phi = f["phi"][:]
        step = int(f.attrs.get("step", 0))
    return phi, step


def parse_step_from_filename(name: str) -> int:
    """Extract step number from filename like 'snapshot_00500.h5'."""
    m = re.search(r"_(\d+)\.h5$", name)
    return int(m.group(1)) if m else 0


# ── Connected-component analysis ─────────────────────────────────────────────

def extract_droplets_from_phi(
    phi: np.ndarray,
    dx_um: float,
    phi_threshold: float = 0.5,
    min_volume_pL: float = 1.0,
) -> list[dict]:
    """
    Label connected components in φ < threshold region.
    Returns list of droplet property dicts.

    Assumes phi shape: (ny, nx, nz) or (nx, ny, nz).
    """
    mask = phi < phi_threshold
    labeled, n_labels = ndi.label(mask)

    dx_m = dx_um * 1e-6
    dx_pL_factor = (dx_m ** 3) * 1e12   # node volume in pL

    droplets = []
    for label_id in range(1, n_labels + 1):
        component = labeled == label_id
        n_nodes = component.sum()
        vol_pL = n_nodes * dx_pL_factor

        if vol_pL < min_volume_pL:
            continue

        # Centroid
        indices = np.argwhere(component)
        centroid = indices.mean(axis=0)

        # Bounding box along x (flow direction = axis 1)
        x_coords = indices[:, 1]

        droplets.append({
            "volume_pL": float(vol_pL),
            "n_nodes": int(n_nodes),
            "cx": float(centroid[0]),
            "cy": float(centroid[1]),
            "cz": float(centroid[2]),
            "bbox_xmin": int(x_coords.min()),
            "bbox_xmax": int(x_coords.max()),
        })

    # Sort by centroid x position (downstream first)
    droplets.sort(key=lambda d: d["cy"], reverse=True)
    return droplets


# ── Droplet tracking ──────────────────────────────────────────────────────────

def track_droplets(
    frames: list[tuple[int, list[dict]]],
    max_displacement: float = 30.0,
) -> list[DropletFrame]:
    """
    Nearest-centroid tracking across frames.
    Returns flat list of DropletFrame records with consistent droplet_id.
    """
    next_id = 0
    prev_centroids: dict[int, tuple[float, float, float]] = {}
    records: list[DropletFrame] = []

    for step, droplets in frames:
        # Match each droplet to nearest previous centroid
        new_centroids: dict[int, tuple[float, float, float]] = {}
        assigned_ids: set[int] = set()

        for d in droplets:
            cx, cy, cz = d["cx"], d["cy"], d["cz"]
            best_id = None
            best_dist = max_displacement

            for did, (px, py, pz) in prev_centroids.items():
                if did in assigned_ids:
                    continue
                dist = ((cx - px)**2 + (cy - py)**2 + (cz - pz)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = did

            if best_id is None:
                best_id = next_id
                next_id += 1

            assigned_ids.add(best_id)
            new_centroids[best_id] = (cx, cy, cz)

            records.append(DropletFrame(
                step=step,
                droplet_id=best_id,
                volume_pL=d["volume_pL"],
                cx=cx, cy=cy, cz=cz,
                bbox_xmin=d["bbox_xmin"],
                bbox_xmax=d["bbox_xmax"],
            ))

        prev_centroids = new_centroids

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract droplet properties from HDF5 snapshots")
    parser.add_argument("--results", required=True, help="Run results directory")
    parser.add_argument("--phi-threshold", type=float, default=0.5)
    parser.add_argument("--min-vol-pL", type=float, default=1.0, help="Min droplet volume filter")
    parser.add_argument("--dx-um", type=float, default=2.5)
    parser.add_argument("--out", default="", help="Output CSV (default: <results>/droplets.csv)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_csv = Path(args.out) if args.out else results_dir / "droplets.csv"

    snapshots = sorted(results_dir.glob("snapshot_*.h5"))
    if not snapshots:
        print(f"No snapshot_*.h5 files found in {results_dir}")
        return

    print(f"Processing {len(snapshots)} snapshots from {results_dir}...")

    frames: list[tuple[int, list[dict]]] = []
    for snap in snapshots:
        phi, step = load_phi_snapshot(snap, args.dx_um)
        if step == 0:
            step = parse_step_from_filename(snap.name)
        droplets = extract_droplets_from_phi(phi, args.dx_um, args.phi_threshold, args.min_vol_pL)
        frames.append((step, droplets))
        print(f"  step {step:6d}: {len(droplets)} droplets")

    print("Tracking droplets across frames...")
    records = track_droplets(frames)

    df = pd.DataFrame([vars(r) for r in records])
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} records → {out_csv}")

    if not df.empty:
        vol_stats = df.groupby("droplet_id")["volume_pL"].mean()
        print(f"Mean droplet volume: {vol_stats.mean():.1f} ± {vol_stats.std():.1f} pL")
        print(f"Unique droplets tracked: {df['droplet_id'].nunique()}")


if __name__ == "__main__":
    main()
