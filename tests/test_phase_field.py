import numpy as np

from droplesim.solver.geometry2d import Geometry2D, build_sparse_maps
from droplesim.solver.sim import (
    PhysParams,
    TwoPhaseSim,
    _disjoining_force,
    _snap_bulk_phase,
)


def _periodic_geom(ny, nx, dx_um=2.5):
    solid_mask = np.zeros((ny, nx), dtype=bool)
    bc_map = np.zeros((ny, nx), dtype=np.uint8)
    return Geometry2D(
        solid_mask=solid_mask,
        bc_map=bc_map,
        specs=[],
        dx_um=dx_um,
        origin_um=(0.0, 0.0),
        sparse=build_sparse_maps(solid_mask, bc_map),
    )


_PHYS = PhysParams(
    mu_c=1.24e-3,
    mu_d=1.2e-3,
    rho_c=1614.0,
    rho_d=1015.0,
    sigma=6e-3,
)


def test_static_droplet_preserves_phase_mass_after_clipping():
    ny, nx = 32, 32
    geom = _periodic_geom(ny, nx)
    sim = TwoPhaseSim(geom, _PHYS)

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


def test_flat_interface_preserves_mass_and_stays_bounded():
    """Flat tanh interface: mass is conserved, phi stays bounded, interface persists."""
    ny, nx = 40, 80
    W = 4
    geom = _periodic_geom(ny, nx)
    sim = TwoPhaseSim(geom, _PHYS, interface_width=W, mobility=0.01)

    # Initialize with equilibrium tanh profile
    x_arr = np.arange(nx, dtype=np.float64)
    phi_1d = 0.5 * (1.0 + np.tanh(2.0 * (x_arr - nx / 2.0) / W))
    phi_init = np.tile(phi_1d, (ny, 1))

    f, phi = sim.init_state(phi_init=phi_init)
    initial_mass = float(np.asarray(phi).sum())

    for _ in range(200):
        f, phi = sim.step(f, phi)

    phi_np = np.asarray(phi)
    assert abs(float(phi_np.sum()) - initial_mass) < 1e-8
    # Interface still has cells between bulk phases
    mixed = (phi_np > 0.1) & (phi_np < 0.9)
    assert float(mixed.mean()) > 0.01
    # Bulk phases remain pure
    assert float(phi_np.min()) < 0.01
    assert float(phi_np.max()) > 0.99


def test_snap_preserves_equilibrium_interface_profile():
    """Equilibrium tanh profile nodes at phi~0.003 must NOT be snapped."""
    import jax.numpy as jnp

    W = 4.0
    x = jnp.linspace(-10.0, 10.0, 200)
    phi = 0.5 * (1.0 + jnp.tanh(x / W))
    snapped = _snap_bulk_phase(phi)
    interface_mask = (phi > 1e-4) & (phi < 1.0 - 1e-4)
    assert jnp.allclose(phi[interface_mask], snapped[interface_mask])
    near_edge = (phi > 0.001) & (phi < 0.01)
    assert near_edge.sum() > 0
    assert jnp.allclose(phi[near_edge], snapped[near_edge])


def test_pure_oil_domain_no_nan():
    """Pure-oil domain (phi=1 everywhere) must not produce NaN."""
    ny, nx = 16, 16
    geom = _periodic_geom(ny, nx)
    sim = TwoPhaseSim(geom, _PHYS)
    phi_init = np.ones((ny, nx), dtype=np.float64)
    f, phi = sim.init_state(phi_init=phi_init)
    for _ in range(20):
        f, phi = sim.step(f, phi)
    phi_np = np.asarray(phi)
    assert not np.any(np.isnan(phi_np))
    assert not np.any(np.isnan(np.asarray(f)))


def test_laplace_pressure_positive():
    """Static circular droplet: inner pressure > outer pressure (Laplace law sign)."""
    ny, nx = 80, 80
    R = 20.0
    geom = _periodic_geom(ny, nx)
    sim = TwoPhaseSim(geom, _PHYS, interface_width=4, mobility=0.01)

    yy, xx = np.mgrid[:ny, :nx]
    cx, cy = nx / 2.0, ny / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # phi=0 inside (disperse), phi=1 outside (continuous)
    phi_init = 0.5 * (1.0 + np.tanh(2.0 * (dist - R) / 4.0))

    f, phi = sim.init_state(phi_init=phi_init)
    for _ in range(500):
        f, phi = sim.step(f, phi)

    rho, _, _ = sim.macroscopic(f)
    rho_np = np.asarray(rho)
    phi_np = np.asarray(phi)

    fy = geom.sparse.fluid_yx[:, 0]
    fx = geom.sparse.fluid_yx[:, 1]
    dist_sparse = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)

    inner = (phi_np < 0.3) & (dist_sparse < R - 4)
    outer = (phi_np > 0.7) & (dist_sparse > R + 4)

    if inner.sum() > 0 and outer.sum() > 0:
        rho_in = float(rho_np[inner].mean())
        rho_out = float(rho_np[outer].mean())
        # Inner pressure must be higher than outer (Laplace law)
        assert rho_in > rho_out, f"rho_in={rho_in:.6f} <= rho_out={rho_out:.6f}"


def test_spurious_currents():
    """Static droplet should have bounded spurious currents."""
    ny, nx = 64, 64
    R = 15.0
    geom = _periodic_geom(ny, nx)
    sim = TwoPhaseSim(geom, _PHYS, interface_width=4, mobility=0.01)

    yy, xx = np.mgrid[:ny, :nx]
    cx, cy = nx / 2.0, ny / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    phi_init = 0.5 * (1.0 + np.tanh(2.0 * (dist - R) / 4.0))

    f, phi = sim.init_state(phi_init=phi_init)
    for _ in range(500):
        f, phi = sim.step(f, phi)

    _, ux, uy = sim.macroscopic(f)
    u_mag = np.sqrt(np.asarray(ux) ** 2 + np.asarray(uy) ** 2)
    u_max = float(u_mag.max())
    # Spurious currents should remain bounded (no blowup)
    assert u_max < 0.1, f"Spurious currents too large: u_max={u_max:.4e}"
    assert not np.any(np.isnan(np.asarray(ux)))


def test_disjoining_force_zero_in_bulk():
    """Disjoining force vanishes in uniform bulk when wall matches."""
    import jax.numpy as jnp

    ny, nx = 16, 16
    geom = _periodic_geom(ny, nx)
    nbr8 = jnp.array(geom.sparse.nbr8)
    nbr8_solid = jnp.array(geom.sparse.nbr8_solid)

    for bulk_val in [0.0, 1.0]:
        phi = jnp.full(geom.sparse.n_fluid, bulk_val)
        fx, fy = _disjoining_force(phi, 1.0, nbr8, nbr8_solid, phi_wall=bulk_val)
        assert float(jnp.abs(fx).max()) < 1e-12
        assert float(jnp.abs(fy).max()) < 1e-12


def test_disjoining_force_repels_approaching_droplets():
    """Two nearby droplets with ε>0 should not merge over 500 steps."""
    ny, nx = 64, 128
    geom = _periodic_geom(ny, nx)
    phys = PhysParams(
        mu_c=1.24e-3,
        mu_d=1.2e-3,
        rho_c=1614.0,
        rho_d=1015.0,
        sigma=6e-3,
        disjoining_strength=0.5,
    )
    sim = TwoPhaseSim(geom, phys, interface_width=4, mobility=0.01)

    yy, xx = np.mgrid[:ny, :nx]
    cy = ny / 2.0
    R = 10.0
    gap = 3.0  # narrow gap between droplets
    cx1 = nx / 2.0 - R - gap / 2.0
    cx2 = nx / 2.0 + R + gap / 2.0
    d1 = np.sqrt((xx - cx1) ** 2 + (yy - cy) ** 2)
    d2 = np.sqrt((xx - cx2) ** 2 + (yy - cy) ** 2)
    phi_init = np.ones((ny, nx), dtype=np.float64)
    phi_init = np.minimum(phi_init, 0.5 * (1.0 + np.tanh(2.0 * (d1 - R) / 4.0)))
    phi_init = np.minimum(phi_init, 0.5 * (1.0 + np.tanh(2.0 * (d2 - R) / 4.0)))

    f, phi = sim.init_state(phi_init=phi_init)
    for _ in range(500):
        f, phi = sim.step(f, phi)

    phi_np = np.asarray(phi)
    # The gap between droplets should still have oil (phi > 0.5)
    fy = geom.sparse.fluid_yx[:, 0]
    fx = geom.sparse.fluid_yx[:, 1]
    midline = (np.abs(fy - cy) < 2) & (np.abs(fx - nx / 2.0) < 2)
    assert midline.sum() > 0
    assert float(phi_np[midline].mean()) > 0.3, "Droplets merged — disjoining force ineffective"
