"""
DXF → 3D boolean voxel mask → HDF5.

Reads generator.dxf, removes stray entity 62, rasterizes the 2D channel
footprint onto a dx=2.5 µm grid, then extrudes 50 nodes in Z (125 µm depth).

Usage:
    uv run python geometry/dxf_to_voxel.py --dxf ../generator.dxf --out geometry/geometry.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ezdxf
import h5py
import matplotlib.path as mpath
import numpy as np

# ── Physical constants ───────────────────────────────────────────────────────

SCALE = 0.125 / 3.175       # DXF units → mm  (main channel = 3.175 DXF = 0.125 mm)
DX_MM = 0.0025              # 2.5 µm in mm
DEPTH_NODES = 50            # 125 µm / 2.5 µm


# ── DXF loading ──────────────────────────────────────────────────────────────

def load_and_clean_dxf(dxf_path: Path) -> list[list[tuple[float, float]]]:
    """
    Read DXF, drop entity index 62 (stray dangling line), collect polyline
    vertices.  Returns a list of closed polygon loops (mm coordinates).
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    entities = list(msp)
    print(f"  DXF entities: {len(entities)}")

    # Remove stray entity 62 (known from dxf_to_step.py analysis)
    if len(entities) > 62:
        msp.delete_entity(entities[62])
        entities = list(msp)
        print(f"  After cleaning: {len(entities)} entities")

    polygons: list[list[tuple[float, float]]] = []
    for ent in entities:
        pts: list[tuple[float, float]] = []
        if ent.dxftype() in ("LWPOLYLINE", "POLYLINE"):
            pts = [(p[0] * SCALE, p[1] * SCALE) for p in ent.get_points()]
        elif ent.dxftype() == "LINE":
            p0 = ent.dxf.start
            p1 = ent.dxf.end
            pts = [(p0.x * SCALE, p0.y * SCALE), (p1.x * SCALE, p1.y * SCALE)]
        if pts:
            polygons.append(pts)

    if not polygons:
        sys.exit("ERROR: No geometry found in DXF after cleaning.")

    return polygons


# ── Rasterization ─────────────────────────────────────────────────────────────

def rasterize(
    polygons: list[list[tuple[float, float]]],
    dx: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Point-in-polygon rasterization of 2D channel footprint.

    Returns:
        mask2d : bool array (ny, nx), True = fluid (inside channel)
        origin  : (x0, y0) in mm of the [0,0] grid corner
    """
    all_pts = np.array([p for poly in polygons for p in poly])
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    # Add one-cell padding
    x_min -= dx
    y_min -= dx
    x_max += dx
    y_max += dx

    nx = int(np.ceil((x_max - x_min) / dx))
    ny = int(np.ceil((y_max - y_min) / dx))
    print(f"  Grid: {nx} × {ny} (x × y)")

    # Cell centres
    xs = x_min + (np.arange(nx) + 0.5) * dx
    ys = y_min + (np.arange(ny) + 0.5) * dx
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    pts_grid = np.column_stack([gx.ravel(), gy.ravel()])

    mask2d = np.zeros((ny, nx), dtype=bool)
    for poly in polygons:
        path = mpath.Path(poly)
        inside = path.contains_points(pts_grid)
        mask2d |= inside.reshape(ny, nx)

    return mask2d, (x_min, y_min)


# ── Extrusion ─────────────────────────────────────────────────────────────────

def extrude(mask2d: np.ndarray, nz: int) -> np.ndarray:
    """Extrude 2D mask into 3D (ny, nx, nz).  All z-layers identical."""
    return np.repeat(mask2d[:, :, np.newaxis], nz, axis=2)


# ── HDF5 output ───────────────────────────────────────────────────────────────

def write_hdf5(
    out_path: Path,
    solid_mask: np.ndarray,
    dx: float,
    origin: tuple[float, float],
) -> None:
    """
    Write geometry HDF5.

    Datasets:
        solid_mask  — bool (ny, nx, nz), True = solid wall
    Attributes:
        dx_mm       — grid spacing in mm
        dx_um       — grid spacing in µm
        origin_mm   — (x0, y0) corner of grid in mm
        shape       — (ny, nx, nz)
    """
    # solid = NOT fluid
    walls = ~solid_mask

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        ds = f.create_dataset(
            "solid_mask",
            data=walls,
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )
        ds.attrs["dx_mm"] = dx
        ds.attrs["dx_um"] = dx * 1000.0
        ds.attrs["origin_mm"] = list(origin)
        ds.attrs["shape"] = list(walls.shape)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Wrote {out_path}  ({size_mb:.1f} MB)")
    print(f"  solid_mask shape: {walls.shape}  (ny, nx, nz)")
    fluid_pct = solid_mask.mean() * 100
    print(f"  Fluid fraction: {fluid_pct:.1f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DXF → 3D voxel HDF5")
    parser.add_argument("--dxf", default="../generator.dxf", help="Path to generator.dxf")
    parser.add_argument("--out", default="geometry/geometry.h5", help="Output HDF5 path")
    parser.add_argument("--dx-um", type=float, default=2.5, help="Grid spacing in µm")
    parser.add_argument("--depth-nodes", type=int, default=DEPTH_NODES, help="Z nodes")
    args = parser.parse_args()

    dxf_path = Path(args.dxf)
    out_path = Path(args.out)
    dx = args.dx_um / 1000.0  # mm

    if not dxf_path.exists():
        sys.exit(f"ERROR: DXF file not found: {dxf_path}")

    print("=" * 60)
    print("DXF → Voxel Converter")
    print("=" * 60)
    print(f"  Input:  {dxf_path}")
    print(f"  Output: {out_path}")
    print(f"  dx = {dx * 1000:.2f} µm,  depth = {args.depth_nodes} nodes "
          f"({args.depth_nodes * dx * 1000:.0f} µm)")

    print("\n[1] Loading DXF...")
    polygons = load_and_clean_dxf(dxf_path)

    print("\n[2] Rasterizing 2D footprint...")
    mask2d, origin = rasterize(polygons, dx)

    print("\n[3] Extruding in Z...")
    mask3d = extrude(mask2d, args.depth_nodes)

    print("\n[4] Writing HDF5...")
    write_hdf5(out_path, mask3d, dx, origin)

    print("\nDone.")


if __name__ == "__main__":
    main()
