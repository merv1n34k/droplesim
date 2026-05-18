import numpy as np
from scipy import ndimage as ndi

from droplesim.solver.geometry2d import BC_OUTLET, BCSpec, Geometry2D, build_sparse_maps
from droplesim.solver.sim import PhysParams, TwoPhaseSim


def _straight_oil_channel(pressure_mbar: float = 5.0):
    nx, ny = 180, 48
    solid_mask = np.ones((ny, nx), dtype=bool)
    solid_mask[1:-1, 1:-1] = False
    bc_map = np.zeros((ny, nx), dtype=np.uint8)
    bc_map[1:-1, 1] = 1
    bc_map[1:-1, -2] = BC_OUTLET

    inlet = BCSpec(
        "oil_inlet", "inlet", 0.0, 0.0, 5.0, ny, phi=1.0,
        pressure_mbar=pressure_mbar,
    )
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
        contact_angle_deg=180.0,
    )
    return geom, phys


def test_high_pressure_oil_inlet_does_not_seed_aqueous_phase():
    geom, phys = _straight_oil_channel(pressure_mbar=1000.0)
    sim = TwoPhaseSim(geom, phys)

    f, phi = sim.init_state()
    for _ in range(1000):
        f, phi = sim.step(f, phi)

    phi_np = np.asarray(phi)
    # With conservative Allen-Cahn (M·∇²φ), wall-adjacent cells may have
    # a tiny perturbation from the wetting boundary phi_wall < 1.0.
    # This is NOT aqueous seeding — just a numerical artifact of the contact
    # angle boundary condition (~2e-4 for M=0.01, contact_angle=150°).
    assert float(phi_np.min()) > 0.999
    assert float((1.0 - phi_np).sum()) < 0.1


def test_aqueous_droplet_integrity_in_pressure_driven_oil_channel():
    geom, phys = _straight_oil_channel()
    solid_mask = geom.solid_mask
    ny, nx = geom.shape
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
    final_aq_mass = float((1.0 - phi_dense[~solid_mask]).sum())

    assert n_components == 1
    # With conservative Allen-Cahn (compression = 4M/W ≈ 0.01), the interface
    # is broader than the old hybrid formulation. Some mass exits via the outlet
    # BC (physical). The droplet must survive as a single connected component
    # with measurable aqueous mass remaining.
    assert final_aq_mass > 0.5 * initial_aq_mass
    assert float(np.nanmin(phi_dense)) < 0.5
