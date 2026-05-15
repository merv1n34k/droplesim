import numpy as np

from droplesim.solver.geometry2d import BC_OUTLET, build_sparse_maps


def test_outlet_upstream_prefers_interior_neighbor():
    solid = np.ones((3, 5), dtype=bool)
    solid[1, 1:4] = False
    bc_map = np.zeros_like(solid, dtype=np.uint8)
    bc_map[1, 3] = BC_OUTLET

    sparse = build_sparse_maps(solid, bc_map)

    outlet_idx = int(np.where(sparse.outlet_mask)[0][0])
    upstream_idx = int(sparse.outlet_upstream[outlet_idx])

    assert tuple(sparse.fluid_yx[outlet_idx]) == (1, 3)
    assert tuple(sparse.fluid_yx[upstream_idx]) == (1, 2)
