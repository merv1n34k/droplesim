import numpy as np
from scipy import ndimage as ndi

from droplesim.solver.geometry2d import BC_OUTLET, BCSpec, Geometry2D, build_sparse_maps
from droplesim.solver.sim import PhysParams, TwoPhaseSim


def test_aqueous_droplet_integrity_in_pressure_driven_oil_channel():
    nx, ny = 180, 48
    solid_mask = np.ones((ny, nx), dtype=bool)
    solid_mask[1:-1, 1:-1] = False
    bc_map = np.zeros((ny, nx), dtype=np.uint8)
    bc_map[1:-1, 1] = 1
    bc_map[1:-1, -2] = BC_OUTLET

    inlet = BCSpec("oil_inlet", "inlet", 0.0, 0.0, 5.0, ny, phi=1.0, pressure_mbar=5.0)
    inlet.type_id = 1
    outlet = BCSpec("outlet", "outlet", nx - 5.0, 0.0, nx, ny, outlet_bc="pressure")
    outlet.type_id = BC_OUTLET
    geom = Geometry2D(
        solid_mask=solid_mask,
        bc_map=bc_map,
        specs=[inlet, outlet],
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
        contact_angle_deg=150.0,
    )
    sim = TwoPhaseSim(geom, phys)

    yy, xx = np.mgrid[:ny, :nx]
    phi_init = np.ones((ny, nx), dtype=np.float64)
    droplet = ((xx - 45) / 8.0) ** 2 + ((yy - ny / 2) / 8.0) ** 2 <= 1.0
    phi_init[droplet & ~solid_mask] = 0.0

    f, phi = sim.init_state(phi_init=phi_init)
    initial_aq_mass = float((1.0 - np.asarray(phi)).sum())
    for _ in range(1000):
        f, phi = sim.step(f, phi)

    phi_dense = np.ones((ny, nx), dtype=np.float64)
    phi_dense[solid_mask] = np.nan
    phi_dense[np.asarray(sim.fluid_y), np.asarray(sim.fluid_x)] = np.asarray(phi)
    aq_mask = np.zeros((ny, nx), dtype=bool)
    aq_mask[~solid_mask] = phi_dense[~solid_mask] < 0.5

    labels, n_components = ndi.label(aq_mask)
    sizes = ndi.sum(aq_mask, labels, index=np.arange(1, n_components + 1))
    final_aq_mass = float((1.0 - phi_dense[~solid_mask]).sum())
    mass_loss = (initial_aq_mass - final_aq_mass) / initial_aq_mass

    assert n_components == 1
    assert int(sizes.max()) >= 0.85 * int(droplet.sum())
    assert mass_loss < 0.03
    assert float(np.nanmin(phi_dense)) == 0.0
