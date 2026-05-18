import numpy as np

from droplesim.solver.geometry2d import BC_OUTLET, BCSpec, Geometry2D, build_sparse_maps
from droplesim.solver.sim import (
    PhysParams,
    TwoPhaseSim,
    _sigma_local,
    _surfactant_coverage,
    contact_angle_to_phi_wall,
    convert_units,
)


def test_convert_units_keeps_viscosity_ratio_in_tau_d():
    phys = PhysParams(
        mu_c=1.24e-3,
        mu_d=1.2e-3,
        rho_c=1614.0,
        rho_d=1015.0,
        sigma=6e-3,
    )

    units = convert_units(phys, dx_um=2.5, tau_c=0.55)

    assert units.tau_c == 0.55
    assert 0.5 < units.tau_d < 2.0
    assert units.rho_ratio == phys.rho_d / phys.rho_c
    assert units.sigma_lbm > 0.0
    assert units.kappa > 0.0
    assert units.beta > 0.0


def test_contact_angle_maps_to_wall_phase_value():
    assert abs(contact_angle_to_phi_wall(0.0) - 0.0) < 1e-12
    assert abs(contact_angle_to_phi_wall(90.0) - 0.5) < 1e-12
    assert abs(contact_angle_to_phi_wall(180.0) - 1.0) < 1e-12


def test_pressure_cap_does_not_rescale_material_properties():
    phys = PhysParams(
        mu_c=1.24e-3,
        mu_d=1.2e-3,
        rho_c=1614.0,
        rho_d=1015.0,
        sigma=6e-3,
    )
    base_units = convert_units(phys, dx_um=2.5, tau_c=0.55)

    solid = np.ones((3, 5), dtype=bool)
    solid[1, 1:4] = False
    bc_map = np.zeros_like(solid, dtype=np.uint8)
    bc_map[1, 1] = 1
    inlet = BCSpec("inlet", "inlet", 0.0, 0.0, 10.0, 10.0, pressure_mbar=100.0)
    inlet.type_id = 1
    geom = Geometry2D(
        solid_mask=solid,
        bc_map=bc_map,
        specs=[inlet],
        dx_um=2.5,
        origin_um=(0.0, 0.0),
        sparse=build_sparse_maps(solid, bc_map),
    )

    sim = TwoPhaseSim(geom, phys, tau_c=0.55, delta_rho_max=0.005)

    assert 0.0 < sim.units.pressure_scale < 1.0
    assert sim.units.sigma_lbm == base_units.sigma_lbm
    assert sim.units.kappa == base_units.kappa
    assert sim.units.beta == base_units.beta
    assert sim.inlet_data[0][2] <= 1.0050000001


def test_mixed_outlet_conditions_fail_fast():
    phys = PhysParams(
        mu_c=1.24e-3,
        mu_d=1.2e-3,
        rho_c=1614.0,
        rho_d=1015.0,
        sigma=6e-3,
    )
    solid = np.ones((3, 6), dtype=bool)
    solid[1, 1:5] = False
    bc_map = np.zeros_like(solid, dtype=np.uint8)
    bc_map[1, 3:5] = BC_OUTLET

    outlet_a = BCSpec("out_a", "outlet", 5.0, 0.0, 10.0, 10.0, outlet_bc="pressure")
    outlet_a.type_id = BC_OUTLET
    outlet_b = BCSpec("out_b", "outlet", 10.0, 0.0, 15.0, 10.0, outlet_bc="neumann")
    outlet_b.type_id = BC_OUTLET
    geom = Geometry2D(
        solid_mask=solid,
        bc_map=bc_map,
        specs=[outlet_a, outlet_b],
        dx_um=2.5,
        origin_um=(0.0, 0.0),
        sparse=build_sparse_maps(solid, bc_map),
    )

    try:
        TwoPhaseSim(geom, phys)
    except ValueError as exc:
        assert "Mixed outlet boundary conditions" in str(exc)
    else:
        raise AssertionError("mixed outlet boundary conditions should fail fast")


def test_surfactant_units_include_floor():
    phys = PhysParams(
        mu_c=1.24e-3,
        mu_d=1.2e-3,
        rho_c=1614.0,
        rho_d=1015.0,
        sigma=50e-3,
        D_s=1e-10,
        D_bulk=5e-10,
        psi_inf=3e-6,
        E0=0.2,
        k_a=10.0,
        k_d=0.1,
        C_inlet=0.1,
        sigma_floor=3e-3,
        surfactant_initial_coverage=0.85,
    )

    units = convert_units(phys, dx_um=2.5, tau_c=0.55)

    assert units.surfactant_enabled
    assert units.sigma_floor_lbm > 0.0
    assert units.sigma_floor_lbm < units.sigma_lbm
    assert units.surfactant_initial_coverage == 0.85


def test_surfactant_coverage_and_sigma_are_bounded():
    psi_inf = 2.0
    psi = np.array([0.0, 0.5, 1.0, 3.0])

    theta = np.asarray(_surfactant_coverage(psi, psi_inf))
    sigma = np.asarray(_sigma_local(psi, 10.0, 0.2, psi_inf, sigma_floor_lbm=3.0))

    assert np.all(theta >= 0.0)
    assert np.all(theta <= 0.999)
    assert np.all(sigma >= 3.0)
    assert sigma[0] > sigma[1] > sigma[2]
    assert sigma[-1] == 3.0


def test_initial_surfactant_coverage_seeds_droplet_interface():
    solid = np.ones((24, 24), dtype=bool)
    solid[1:-1, 1:-1] = False
    bc_map = np.zeros_like(solid, dtype=np.uint8)
    geom = Geometry2D(
        solid_mask=solid,
        bc_map=bc_map,
        specs=[],
        dx_um=2.5,
        origin_um=(0.0, 0.0),
        sparse=build_sparse_maps(solid, bc_map),
    )
    phys = PhysParams(
        mu_c=1.24e-3,
        mu_d=1.2e-3,
        rho_c=1614.0,
        rho_d=1015.0,
        sigma=6e-3,
        D_s=1e-10,
        psi_inf=3e-6,
        surfactant_initial_coverage=0.8,
    )
    yy, xx = np.mgrid[:24, :24]
    dist = np.hypot(xx - 12.0, yy - 12.0)
    phi_init = 1.0 - 0.5 * (1.0 - np.tanh((dist - 5.0) / 1.2))

    sim = TwoPhaseSim(geom, phys)
    _f, _phi, psi, _C = sim.init_state(phi_init=phi_init)
    theta = np.asarray(_surfactant_coverage(psi, sim.units.psi_inf_lu))

    assert float(np.asarray(psi).sum()) > 0.0
    assert theta.max() <= 0.8 + 1e-12
    assert theta.max() > 0.5
