from droplesim.solver.sim import PhysParams, convert_units


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
