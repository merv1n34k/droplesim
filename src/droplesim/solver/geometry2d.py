"""
2D geometry: DXF → rasterised solid mask + boundary-condition assignment.

BC assignment rule
------------------
Any fluid node whose cell-centre lies *completely inside* a BCSpec's
bounding box gets that spec's type_id written into bc_map.
Multiple inlets (oil, aq, or any other fluid) are supported — just add
more BCSpec entries.  Specs are applied in list order; later specs win
on overlap.

BC kind codes stored in bc_map
-------------------------------
0   bulk fluid
1…N inlet   (one per BCSpec with kind="inlet")
254 outlet  (Neumann / zero-gradient)
255 solid   (informational only — separate from solid_mask)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

import ezdxf
import matplotlib.path as mpath
import numpy as np

_SCALE = 0.125 / 3.175   # DXF units → mm

BC_OUTLET: int = 254


@dataclass
class EdgeSpec:
    """One wall edge segment with optional BC assignment."""
    name: str
    kind: Literal["wall", "inlet", "outlet"]
    points_um: list[tuple[float, float]]
    phi: float = 1.0
    pressure_mbar: float = 0.0  # inlet gauge pressure [mbar]
    outlet_bc: str = "pressure"  # "neumann" or "pressure"
    rho_target: float = 1.0
    type_id: int = field(default=0, init=False)


@dataclass
class BCSpec:
    """One boundary region.  Coordinates in physical µm."""
    name: str
    kind: Literal["inlet", "outlet"]
    x1_um: float
    y1_um: float
    x2_um: float
    y2_um: float
    # inlet-only
    phi: float = 1.0            # phase value: 1.0 = oil, 0.0 = aqueous
    pressure_mbar: float = 0.0  # inlet gauge pressure [mbar]
    # outlet-only
    outlet_bc: str = "pressure"  # "neumann" or "pressure"
    rho_target: float = 1.0      # target density for pressure outlet BC
    # assigned automatically by assign_bcs()
    type_id: int = field(default=0, init=False)


@dataclass
class SparseIndex:
    """Pre-computed index arrays for sparse (fluid-only) LBM."""
    n_fluid: int                     # number of fluid cells
    fluid_yx: np.ndarray             # (n_fluid, 2) int — (row, col) of each fluid cell
    index_map: np.ndarray            # (ny, nx) int32 — fluid index at each cell, -1 for solid
    pull_src: np.ndarray             # (9, n_fluid) int32 — streaming source fluid index
    pull_bb: np.ndarray              # (9, n_fluid) bool — True = bounce-back (neighbor is solid)
    nbr4: np.ndarray                 # (4, n_fluid) int32 — E,W,N,S neighbor fluid indices
    nbr4_solid: np.ndarray           # (4, n_fluid) bool — True if that neighbor is solid
    nbr8: np.ndarray                 # (8, n_fluid) int32 — E,W,N,S,NE,NW,SE,SW neighbors
    nbr8_solid: np.ndarray           # (8, n_fluid) bool — True if that neighbor is solid
    bc_map_fluid: np.ndarray         # (n_fluid,) uint8 — BC type per fluid cell
    outlet_mask: np.ndarray          # (n_fluid,) bool — True for outlet cells
    outlet_upstream: np.ndarray      # (n_fluid,) int32 — fluid idx of upstream cell (valid only at outlets)
    phi_wall_nbr8: np.ndarray | None = None  # optional per-link wall φ for wetting overrides


@dataclass
class Geometry2D:
    solid_mask: np.ndarray          # bool (ny, nx)  True = wall
    bc_map: np.ndarray              # uint8 (ny, nx)  see codes above
    specs: list[BCSpec]             # ordered list with type_ids filled in
    dx_um: float
    origin_um: tuple[float, float]  # (x0, y0) of cell [0,0] centre in µm
    sparse: SparseIndex | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.solid_mask.shape   # (ny, nx)

    @property
    def size_um(self) -> tuple[float, float]:
        ny, nx = self.shape
        return nx * self.dx_um, ny * self.dx_um

    def inlet_specs(self) -> list[BCSpec]:
        return [s for s in self.specs if s.kind == "inlet"]

    def outlet_specs(self) -> list[BCSpec]:
        return [s for s in self.specs if s.kind == "outlet"]


# ── Sparse index construction ────────────────────────────────────────────────

# D2Q9 lattice velocities (must match sim.py)
_EX9 = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
_EY9 = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
_OPP9 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


def build_sparse_maps(solid_mask: np.ndarray, bc_map: np.ndarray) -> SparseIndex:
    """Build all index arrays for sparse (fluid-only) LBM.

    Parameters
    ----------
    solid_mask : (ny, nx) bool — True = wall
    bc_map     : (ny, nx) uint8 — boundary-condition codes
    """
    ny, nx = solid_mask.shape

    # Fluid cell coordinates: (row, col) ordered by row-major scan
    fluid_yx = np.argwhere(~solid_mask).astype(np.int32)  # (n_fluid, 2)
    n_fluid = len(fluid_yx)

    # index_map: dense→sparse lookup.  -1 for solid.
    index_map = np.full((ny, nx), -1, dtype=np.int32)
    index_map[fluid_yx[:, 0], fluid_yx[:, 1]] = np.arange(n_fluid, dtype=np.int32)

    # Pull-based streaming: for each direction i and fluid cell j,
    # the source is at (y - ey[i], x - ex[i]).
    pull_src = np.zeros((9, n_fluid), dtype=np.int32)
    pull_bb = np.zeros((9, n_fluid), dtype=bool)

    fy = fluid_yx[:, 0]
    fx = fluid_yx[:, 1]

    for i in range(9):
        src_y = fy - _EY9[i]
        src_x = fx - _EX9[i]
        # Out-of-bounds → treat as solid (bounce-back)
        oob = (src_y < 0) | (src_y >= ny) | (src_x < 0) | (src_x >= nx)
        # Clamp for safe indexing (values at OOB will be overridden)
        sy = np.clip(src_y, 0, ny - 1)
        sx = np.clip(src_x, 0, nx - 1)
        src_idx = index_map[sy, sx]
        is_solid = oob | (src_idx < 0)

        pull_src[i] = np.where(is_solid, np.arange(n_fluid), src_idx)
        pull_bb[i] = is_solid

    # 4-neighbor indices: E(+x), W(-x), N(+y), S(-y)
    nbr_dx = np.array([1, -1, 0, 0], dtype=np.int32)
    nbr_dy = np.array([0, 0, 1, -1], dtype=np.int32)
    nbr4 = np.zeros((4, n_fluid), dtype=np.int32)
    nbr4_solid = np.zeros((4, n_fluid), dtype=bool)

    for d in range(4):
        ny_ = fy + nbr_dy[d]
        nx_ = fx + nbr_dx[d]
        oob = (ny_ < 0) | (ny_ >= ny) | (nx_ < 0) | (nx_ >= nx)
        cy = np.clip(ny_, 0, ny - 1)
        cx = np.clip(nx_, 0, nx - 1)
        nidx = index_map[cy, cx]
        is_solid = oob | (nidx < 0)
        nbr4[d] = np.where(is_solid, np.arange(n_fluid), nidx)
        nbr4_solid[d] = is_solid

    # 8-neighbor indices: E,W,N,S,NE,NW,SE,SW (for isotropic stencils)
    nbr8_dx = np.array([1, -1, 0, 0, 1, -1, 1, -1], dtype=np.int32)
    nbr8_dy = np.array([0, 0, 1, -1, 1, 1, -1, -1], dtype=np.int32)
    nbr8 = np.zeros((8, n_fluid), dtype=np.int32)
    nbr8_solid = np.zeros((8, n_fluid), dtype=bool)

    for d in range(8):
        ny_ = fy + nbr8_dy[d]
        nx_ = fx + nbr8_dx[d]
        oob = (ny_ < 0) | (ny_ >= ny) | (nx_ < 0) | (nx_ >= nx)
        cy = np.clip(ny_, 0, ny - 1)
        cx = np.clip(nx_, 0, nx - 1)
        nidx = index_map[cy, cx]
        is_solid = oob | (nidx < 0)
        nbr8[d] = np.where(is_solid, np.arange(n_fluid), nidx)
        nbr8_solid[d] = is_solid

    # BC map for fluid cells
    bc_map_fluid = bc_map[fy, fx]

    # Outlet: find upstream neighbor (interior fluid, not outlet).
    # Auto-detect direction — works regardless of chip orientation.
    outlet_mask = (bc_map_fluid == BC_OUTLET)
    outlet_upstream = np.arange(n_fluid, dtype=np.int32)  # default: self
    not_found = outlet_mask.copy()

    # First pass: prefer non-outlet fluid neighbors (true interior cells)
    for d in range(4):
        ny_ = fy + nbr_dy[d]
        nx_ = fx + nbr_dx[d]
        oob = (ny_ < 0) | (ny_ >= ny) | (nx_ < 0) | (nx_ >= nx)
        cy = np.clip(ny_, 0, ny - 1)
        cx = np.clip(nx_, 0, nx - 1)
        nidx = index_map[cy, cx]
        is_fluid = ~oob & (nidx >= 0)
        is_not_outlet = bc_map[cy, cx] != BC_OUTLET
        use = not_found & is_fluid & is_not_outlet
        outlet_upstream[use] = nidx[use]
        not_found &= ~use

    # Second pass: accept any fluid neighbor for remaining outlet cells
    # (e.g. multi-layer outlet strips where inner layer chains through)
    for d in range(4):
        if not not_found.any():
            break
        ny_ = fy + nbr_dy[d]
        nx_ = fx + nbr_dx[d]
        oob = (ny_ < 0) | (ny_ >= ny) | (nx_ < 0) | (nx_ >= nx)
        cy = np.clip(ny_, 0, ny - 1)
        cx = np.clip(nx_, 0, nx - 1)
        nidx = index_map[cy, cx]
        is_fluid = ~oob & (nidx >= 0)
        use = not_found & is_fluid
        outlet_upstream[use] = nidx[use]
        not_found &= ~use

    return SparseIndex(
        n_fluid=n_fluid,
        fluid_yx=fluid_yx,
        index_map=index_map,
        pull_src=pull_src,
        pull_bb=pull_bb,
        nbr4=nbr4,
        nbr4_solid=nbr4_solid,
        nbr8=nbr8,
        nbr8_solid=nbr8_solid,
        bc_map_fluid=bc_map_fluid.astype(np.uint8),
        outlet_mask=outlet_mask,
        outlet_upstream=outlet_upstream,
    )


# ── DXF loading ───────────────────────────────────────────────────────────────

_ARC_SEGMENTS = 32  # points per arc for discretisation


def _read_dxf(source: Path | str | IO):
    """Read a DXF from file path, BytesIO, or text stream."""
    if isinstance(source, (str, Path)):
        return ezdxf.readfile(str(source))
    # Streamlit uploader gives BytesIO — save to temp file for ezdxf
    import tempfile
    raw = source.read()
    if isinstance(raw, str):
        raw = raw.encode("ascii")
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmp.write(raw)
        tmp.flush()
        return ezdxf.readfile(tmp.name)


def _arc_points(ent) -> list[tuple[float, float]]:
    """Sample an ARC entity into line-segment points."""
    cx, cy = ent.dxf.center.x, ent.dxf.center.y
    r = ent.dxf.radius
    a0 = np.radians(ent.dxf.start_angle)
    a1 = np.radians(ent.dxf.end_angle)
    if a1 <= a0:
        a1 += 2.0 * np.pi
    angles = np.linspace(a0, a1, _ARC_SEGMENTS + 1)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]


def _extract_segments(doc) -> list[list[tuple[float, float]]]:
    """Extract ordered-point segments from all LINE/ARC/POLYLINE entities."""
    msp = doc.modelspace()

    segments: list[list[tuple[float, float]]] = []
    for ent in msp:
        t = ent.dxftype()
        if t in ("LWPOLYLINE", "POLYLINE"):
            pts = [(p[0], p[1]) for p in ent.get_points()]
            if len(pts) >= 2:
                segments.append(pts)
        elif t == "LINE":
            p0, p1 = ent.dxf.start, ent.dxf.end
            segments.append([(p0.x, p0.y), (p1.x, p1.y)])
        elif t == "ARC":
            pts = _arc_points(ent)
            if len(pts) >= 2:
                segments.append(pts)
    return segments


def _chain_segments(
    segments: list[list[tuple[float, float]]],
    tol: float = 1e-4,
) -> list[list[tuple[float, float]]]:
    """Chain connected segments into contour polylines."""
    if not segments:
        return []

    # Each segment is a list of (x,y) points. Build chains by matching endpoints.
    remaining = list(range(len(segments)))
    chains: list[list[tuple[float, float]]] = []

    def dist(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    while remaining:
        chain = list(segments[remaining.pop(0)])
        changed = True
        while changed:
            changed = False
            for j in list(remaining):
                seg = segments[j]
                head, tail = chain[0], chain[-1]
                s0, s1 = seg[0], seg[-1]
                if dist(tail, s0) < tol:
                    chain.extend(seg[1:])
                    remaining.remove(j)
                    changed = True
                elif dist(tail, s1) < tol:
                    chain.extend(reversed(seg[:-1]))
                    remaining.remove(j)
                    changed = True
                elif dist(head, s1) < tol:
                    chain = list(seg[:-1]) + chain
                    remaining.remove(j)
                    changed = True
                elif dist(head, s0) < tol:
                    chain = list(reversed(seg[1:])) + chain
                    remaining.remove(j)
                    changed = True
        chains.append(chain)

    return chains


def _is_closed_chain(chain: list[tuple[float, float]], tol: float = 1e-2) -> bool:
    if len(chain) < 3:
        return False
    a = chain[0]
    b = chain[-1]
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5 < tol


def _load_chains(source: Path | str | IO) -> list[list[tuple[float, float]]]:
    doc = _read_dxf(source)
    segments = _extract_segments(doc)

    if not segments:
        raise ValueError("No geometry entities found in DXF.")

    # Scale DXF units → mm
    scaled = [
        [(x * _SCALE, y * _SCALE) for x, y in seg]
        for seg in segments
    ]

    return _chain_segments(scaled)


def _load_polygons(source: Path | str | IO) -> list[list[tuple[float, float]]]:
    polygons = [chain for chain in _load_chains(source) if _is_closed_chain(chain)]

    if not polygons:
        raise ValueError(
            "Could not chain DXF entities into closed polygons. "
            "Check DXF file for stray entities."
        )
    return polygons


def _rasterize(
    polygons: list[list[tuple[float, float]]],
    dx_mm: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    all_pts = np.array([p for poly in polygons for p in poly])
    x_min, y_min = all_pts.min(axis=0) - dx_mm
    x_max, y_max = all_pts.max(axis=0) + dx_mm

    nx = int(np.ceil((x_max - x_min) / dx_mm))
    ny = int(np.ceil((y_max - y_min) / dx_mm))

    xs = x_min + (np.arange(nx) + 0.5) * dx_mm
    ys = y_min + (np.arange(ny) + 0.5) * dx_mm
    gx, gy = np.meshgrid(xs, ys)
    pts_grid = np.column_stack([gx.ravel(), gy.ravel()])

    fluid = np.zeros((ny, nx), dtype=bool)
    for poly in polygons:
        path = mpath.Path(poly)
        fluid ^= path.contains_points(pts_grid).reshape(ny, nx)

    return fluid, (x_min, y_min)


def _burn_contour_walls(
    solid_mask: np.ndarray,
    contours: list[list[tuple[float, float]]],
    origin_mm: tuple[float, float],
    dx_mm: float,
) -> None:
    """Rasterize contour polylines into solid cells (1-cell thickness)."""
    if not contours:
        return

    ny, nx = solid_mask.shape
    x0, y0 = origin_mm

    def xy_to_ij(x: float, y: float) -> tuple[int, int]:
        ix = int(round((x - x0) / dx_mm - 0.5))
        iy = int(round((y - y0) / dx_mm - 0.5))
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)
        return iy, ix

    for contour in contours:
        if len(contour) < 2:
            continue
        for k in range(len(contour) - 1):
            x0s, y0s = contour[k]
            x1s, y1s = contour[k + 1]
            iy0, ix0 = xy_to_ij(x0s, y0s)
            iy1, ix1 = xy_to_ij(x1s, y1s)
            n = max(abs(ix1 - ix0), abs(iy1 - iy0))
            if n == 0:
                solid_mask[iy0, ix0] = True
                continue
            xs = np.linspace(ix0, ix1, n + 1)
            ys = np.linspace(iy0, iy1, n + 1)
            ixs = np.rint(xs).astype(np.int32)
            iys = np.rint(ys).astype(np.int32)
            solid_mask[iys, ixs] = True


# ── BC assignment ─────────────────────────────────────────────────────────────

def assign_bcs(
    solid_mask: np.ndarray,
    specs: list[BCSpec],
    dx_um: float,
    origin_um: tuple[float, float],
    bc_map: np.ndarray | None = None,
    inlet_counter: int = 1,
) -> tuple[np.ndarray, list[BCSpec]]:
    """
    Mark fluid nodes inside each spec's box.

    Inlets get sequential type_ids starting from *inlet_counter*.
    Outlets get type_id = BC_OUTLET (254).
    If *bc_map* is provided it is modified in-place (area BCs layer on
    top of edge BCs); otherwise a fresh zero map is created.
    Returns (bc_map, specs_with_ids_filled).
    """
    ny, nx = solid_mask.shape
    ox, oy = origin_um

    xs = ox + (np.arange(nx) + 0.5) * dx_um
    ys = oy + (np.arange(ny) + 0.5) * dx_um
    XX, YY = np.meshgrid(xs, ys)   # (ny, nx)

    if bc_map is None:
        bc_map = np.zeros((ny, nx), dtype=np.uint8)

    for spec in specs:
        inside = (
            (XX >= spec.x1_um) & (XX <= spec.x2_um) &
            (YY >= spec.y1_um) & (YY <= spec.y2_um) &
            (~solid_mask)
        )
        if spec.kind == "inlet":
            spec.type_id = inlet_counter
            inlet_counter += 1
        else:
            spec.type_id = BC_OUTLET
        bc_map[inside] = spec.type_id

    return bc_map, specs


# ── Public entry point ────────────────────────────────────────────────────────

def load_geometry(
    source: Path | str | IO,
    specs: list[BCSpec],
    dx_um: float = 2.5,
) -> Geometry2D:
    """
    Load a DXF, rasterise it, and apply all BCSpecs.

    Parameters
    ----------
    source       : file path or file-like object (Streamlit uploader)
    specs        : list of BCSpec — any number of inlets + outlets
    dx_um        : grid spacing in µm
    """
    polygons = _load_polygons(source)
    fluid_mask, (ox_mm, oy_mm) = _rasterize(polygons, dx_um / 1000.0)
    solid_mask = ~fluid_mask
    origin_um = (ox_mm * 1000.0, oy_mm * 1000.0)

    bc_map, specs = assign_bcs(solid_mask, specs, dx_um, origin_um)
    sparse = build_sparse_maps(solid_mask, bc_map)
    return Geometry2D(
        solid_mask=solid_mask,
        bc_map=bc_map,
        specs=specs,
        dx_um=dx_um,
        origin_um=origin_um,
        sparse=sparse,
    )


# ── Public polygon loading ───────────────────────────────────────────────────

def load_polygons(
    source: Path | str | IO,
) -> tuple[list[list[tuple[float, float]]], float]:
    """Load DXF and return (polygons_mm, scale).

    Public wrapper around _load_polygons for GUI edge extraction.
    """
    return _load_polygons(source), _SCALE


def load_contours(
    source: Path | str | IO,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]], float]:
    """Load DXF and return (closed_polygons_mm, all_contours_mm, scale)."""
    chains = _load_chains(source)
    polygons = [chain for chain in chains if _is_closed_chain(chain)]
    if not polygons:
        raise ValueError(
            "Could not chain DXF entities into closed polygons. "
            "Check DXF file for stray entities."
        )
    return polygons, chains, _SCALE


def rasterize_polygons(
    polygons: list[list[tuple[float, float]]],
    dx_um: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Rasterize polygons to solid mask.

    Returns (solid_mask, origin_um).
    """
    fluid_mask, (ox_mm, oy_mm) = _rasterize(polygons, dx_um / 1000.0)
    solid_mask = ~fluid_mask
    origin_um = (ox_mm * 1000.0, oy_mm * 1000.0)
    return solid_mask, origin_um


def rasterize_contours(
    polygons: list[list[tuple[float, float]]],
    contours: list[list[tuple[float, float]]],
    dx_um: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Rasterize polygons and burn contours as solid divider walls.

    This preserves internal/open CAD lines as barriers in the voxel mask.
    """
    fluid_mask, (ox_mm, oy_mm) = _rasterize(polygons, dx_um / 1000.0)
    solid_mask = ~fluid_mask
    open_contours = [c for c in contours if not _is_closed_chain(c)]
    _burn_contour_walls(solid_mask, open_contours, (ox_mm, oy_mm), dx_um / 1000.0)
    origin_um = (ox_mm * 1000.0, oy_mm * 1000.0)
    return solid_mask, origin_um


# ── Edge extraction ──────────────────────────────────────────────────────────

def extract_edges(
    polygons: list[list[tuple[float, float]]],
    angle_threshold_deg: float = 30.0,
) -> list[list[tuple[float, float]]]:
    """Split polygon contours at corners into clickable edge segments.

    Parameters
    ----------
    polygons : list of polylines in mm coordinates (from load_polygons)
    angle_threshold_deg : split where angle change exceeds this

    Returns list of polylines in mm coordinates. Each polyline is a
    straight or smoothly curved wall section between two corners.
    """
    threshold = np.radians(angle_threshold_deg)
    all_edges: list[list[tuple[float, float]]] = []

    for poly in polygons:
        pts = np.array(poly)
        n = len(pts)
        if n < 3:
            all_edges.append(poly)
            continue

        # Compute vectors between consecutive points
        vecs = np.diff(pts, axis=0)  # (n-1, 2)
        # Compute angles of each vector
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])

        # Find corners: where angle change > threshold
        split_indices = [0]
        for i in range(1, len(angles)):
            da = abs(angles[i] - angles[i - 1])
            da = min(da, 2 * np.pi - da)
            if da > threshold:
                split_indices.append(i)

        # Build edge segments between split points
        split_indices.append(n - 1)
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]
            if end > start:
                edge_pts = [tuple(pts[j]) for j in range(start, end + 1)]
                if len(edge_pts) >= 2:
                    all_edges.append(edge_pts)

    return all_edges


def assign_edge_bcs(
    solid_mask: np.ndarray,
    edges: list[EdgeSpec],
    dx_um: float,
    origin_um: tuple[float, float],
) -> tuple[np.ndarray, list[EdgeSpec]]:
    """Assign BC types based on labeled edges.

    For each edge labeled inlet/outlet:
    1. Find wall pixels closest to the edge polyline (within dx_um)
    2. Find fluid nodes 4-adjacent to those wall pixels
    3. Set those fluid nodes in bc_map
    """
    from scipy.ndimage import binary_dilation

    ny, nx = solid_mask.shape
    ox, oy = origin_um
    bc_map = np.zeros((ny, nx), dtype=np.uint8)

    # Build coordinate arrays (cell centres in µm)
    xs = ox + (np.arange(nx) + 0.5) * dx_um
    ys = oy + (np.arange(ny) + 0.5) * dx_um

    inlet_counter = 1

    for edge in edges:
        if edge.kind == "wall":
            continue

        # Convert edge polyline from µm to pixel coords
        edge_pts = np.array(edge.points_um)
        # Find wall pixels near the edge polyline
        wall_mask = np.zeros((ny, nx), dtype=bool)
        for k in range(len(edge_pts) - 1):
            p0 = edge_pts[k]
            p1 = edge_pts[k + 1]
            seg_vec = p1 - p0
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-12:
                continue
            seg_dir = seg_vec / seg_len

            # For each grid cell, compute distance to this line segment
            for iy in range(ny):
                for ix in range(nx):
                    pt = np.array([xs[ix], ys[iy]])
                    v = pt - p0
                    t = np.clip(np.dot(v, seg_dir), 0, seg_len)
                    closest = p0 + t * seg_dir
                    dist = np.linalg.norm(pt - closest)
                    if dist < 1.5 * dx_um and solid_mask[iy, ix]:
                        wall_mask[iy, ix] = True

        # Find fluid nodes adjacent to those wall pixels (4-connected dilation)
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        dilated = binary_dilation(wall_mask, structure=cross)
        adjacent_fluid = dilated & (~solid_mask) & (~wall_mask)

        if edge.kind == "inlet":
            edge.type_id = inlet_counter
            inlet_counter += 1
        else:
            edge.type_id = BC_OUTLET
        bc_map[adjacent_fluid] = edge.type_id

    return bc_map, edges
