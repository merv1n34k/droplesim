import numpy as np

from droplesim.solver.geometry2d import BC_OUTLET, BCSpec, Geometry2D, build_sparse_maps
from droplesim.solver.sim import PhysParams, TwoPhaseSim, convert_units


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
