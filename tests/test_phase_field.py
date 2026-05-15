import numpy as np

from droplesim.solver.geometry2d import Geometry2D, build_sparse_maps
from droplesim.solver.sim import PhysParams, TwoPhaseSim


def test_static_droplet_preserves_phase_mass_after_clipping():
    ny, nx = 32, 32
    solid_mask = np.zeros((ny, nx), dtype=bool)
    bc_map = np.zeros((ny, nx), dtype=np.uint8)
    geom = Geometry2D(
        solid_mask=solid_mask,
        bc_map=bc_map,
        specs=[],
        dx_um=2.5,
        origin_um=(0.0, 0.0),
        sparse=build_sparse_maps(solid_mask, bc_map),
    )
    phys = PhysParams(
        mu_c=1.24e-3,
        mu_d=1.2e-3,
        rho_c=1614.0,
        rho_d=1015.0,
        sigma=6e-3,
    )
    sim = TwoPhaseSim(geom, phys, mobility=0.1)

    yy, xx = np.mgrid[:ny, :nx]
    radius = np.sqrt((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2)
    phi_init = np.ones((ny, nx), dtype=np.float64)
    phi_init[radius < 6] = 0.0

    f, phi = sim.init_state(phi_init=phi_init)
    initial_mass = float(np.asarray(phi).sum())
    for _ in range(100):
        f, phi = sim.step(f, phi)

    phi_np = np.asarray(phi)
    assert abs(float(phi_np.sum()) - initial_mass) < 1e-8
    assert float(phi_np.min()) < 0.1
    assert float(phi_np.max()) == 1.0
