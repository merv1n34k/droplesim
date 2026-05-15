"""
Pure-JAX two-phase LBM solver (D2Q9 + Allen-Cahn phase field).
Sparse / indirect-addressing: stores and computes ONLY fluid cells.

Physics
-------
f_i  — 9 distribution functions for Navier-Stokes (BGK + Guo forcing)
φ    — phase-field order parameter: 1 = continuous, 0 = disperse
τ(φ) — spatially varying relaxation time (linear interpolation)
F_s  — surface-tension body force = μ ∇φ  (chemical-potential model)

Allen-Cahn equation solved via finite-difference advection-diffusion
with a double-well reaction term.

Unit conversion
---------------
Anchor: τ_c = 0.55  →  ν_c_LU = (τ_c - 0.5) / 3
dx_phys given (e.g. 2.5 µm).  dt derived so that ν_c matches.
All other LBM parameters (τ_d, u_inlet, σ) follow from dt/dx.

Sparse addressing
-----------------
All field arrays are (n_fluid,) instead of (ny, nx).
Distribution arrays are (9, n_fluid) instead of (9, ny, nx).
Streaming and neighbor access use pre-computed index arrays from SparseIndex.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from droplesim.solver.geometry2d import Geometry2D  # noqa: E402

# ── D2Q9 lattice ─────────────────────────────────────────────────────────────

#            0      1      2       3       4      5       6        7       8
_EX = jnp.array([0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=jnp.int32)
_EY = jnp.array([0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=jnp.int32)
_W  = jnp.array([4/9,
                  1/9, 1/9, 1/9, 1/9,
                  1/36, 1/36, 1/36, 1/36], dtype=jnp.float64)
_OPP = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=jnp.int32)
_CS2 = 1.0 / 3.0
_Q = 9


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PhysParams:
    mu_c: float         # continuous phase dynamic viscosity, Pa·s
    mu_d: float         # disperse phase dynamic viscosity, Pa·s
    rho_c: float        # continuous phase density, kg/m³
    rho_d: float        # disperse phase density, kg/m³
    sigma: float        # interfacial tension, N/m
    contact_angle_deg: float = 150.0
    # Surfactant (all None = disabled)
    D_s: float | None = None        # interfacial diffusivity, m²/s
    D_bulk: float | None = None     # bulk diffusivity, m²/s
    psi_inf: float | None = None    # max interfacial concentration, mol/m²
    E0: float | None = None         # elasticity number (dimensionless)
    k_a: float | None = None        # adsorption rate, m³/(mol·s)
    k_d: float | None = None        # desorption rate, 1/s
    C_inlet: float | None = None    # inlet bulk concentration, mol/m³
    # Viscoelastic (all None = disabled, disperse phase only)
    lambda_p: float | None = None   # polymer relaxation time, s
    mu_p: float | None = None       # polymer viscosity contribution, Pa·s
    kappa_ve: float | None = None   # artificial diffusion for A_ij stability


@dataclass
class LBMUnits:
    tau_c: float
    tau_d: float
    dx: float           # physical length per node [m]
    dt: float           # physical time per step [s]
    rho_ratio: float    # rho_d / rho_c in LBM (density contrast)
    sigma_lbm: float
    kappa: float        # gradient energy coefficient
    beta: float         # bulk free-energy coefficient
    mobility: float
    interface_width: int  # nodes
    # Surfactant LU (0.0 when disabled)
    D_s_lu: float = 0.0
    D_bulk_lu: float = 0.0
    psi_inf_lu: float = 0.0
    E0: float = 0.0
    k_a_lu: float = 0.0
    k_d_lu: float = 0.0
    C_inlet_lu: float = 0.0
    surfactant_enabled: bool = False
    # Viscoelastic LU (0.0 when disabled)
    lambda_p_lu: float = 0.0
    mu_p_lu: float = 0.0
    beta_visc: float = 1.0       # solvent fraction mu_s / mu_total
    kappa_ve_lu: float = 0.0
    tau_d_solvent: float = 0.0   # τ_d using solvent-only viscosity
    viscoelastic_enabled: bool = False
    pressure_scale: float = 1.0  # 1.0 = requested pressure maps directly to LU


def convert_units(
    phys: PhysParams,
    dx_um: float,
    tau_c: float = 0.55,
    interface_width: int = 4,
    mobility: float = 0.1,
) -> LBMUnits:
    dx = dx_um * 1e-6  # m
    nu_c = phys.mu_c / phys.rho_c        # kinematic viscosity [m²/s]
    nu_c_lu = (tau_c - 0.5) / 3.0
    dt = nu_c_lu * dx**2 / nu_c          # time step [s]

    nu_d = phys.mu_d / phys.rho_d
    nu_d_lu = nu_d * dt / dx**2
    tau_d = 3.0 * nu_d_lu + 0.5

    rho_ratio = phys.rho_d / phys.rho_c

    # Allen-Cahn interface parameters
    # Free energy: F(φ) = β·φ²·(1-φ)²
    # Chemical potential: μ = dF/dφ - κ∇²φ = 2β·φ(1-φ)(1-2φ) - κ∇²φ
    # Equilibrium profile: φ(x) = 0.5·(1 + tanh(2x/W))
    # Relations: W = √(8κ/β),  σ = 2κ/(3W)
    # Solving:   κ = 3σW/2,    β = 12σ/W

    W_lu = float(interface_width)
    sigma_lbm = phys.sigma * dt**2 / (phys.rho_c * dx**3)
    kappa_lu = 1.5 * sigma_lbm * W_lu       # 3σW/2
    beta_lu = 12.0 * sigma_lbm / W_lu       # 12σ/W

    # Surfactant unit conversion
    surf_enabled = phys.D_s is not None and phys.psi_inf is not None
    D_s_lu = 0.0
    D_bulk_lu = 0.0
    psi_inf_lu = 0.0
    E0 = 0.0
    k_a_lu = 0.0
    k_d_lu = 0.0
    C_inlet_lu = 0.0
    if surf_enabled:
        D_s_lu = phys.D_s * dt / dx**2
        D_bulk_lu = (phys.D_bulk or phys.D_s) * dt / dx**2
        psi_inf_lu = phys.psi_inf * dx**2
        E0 = phys.E0 or 0.0
        if phys.k_a is not None:
            # k_a [m³/(mol·s)] → LU: k_a * (dt / dx³) * (mol_scale)
            # With psi_inf_lu = psi_inf * dx², concentration C [mol/m³]:
            # C_lu = C * dx³, so k_a_lu = k_a * C_lu_scale * dt / psi_lu_scale
            k_a_lu = phys.k_a * dt / dx
        if phys.k_d is not None:
            k_d_lu = phys.k_d * dt
        if phys.C_inlet is not None:
            # C [mol/m³] → LU: C * dx³ (concentration per lattice volume)
            C_inlet_lu = phys.C_inlet * dx**3

    # Viscoelastic unit conversion
    ve_enabled = phys.lambda_p is not None and phys.mu_p is not None
    lambda_p_lu = 0.0
    mu_p_lu = 0.0
    beta_visc = 1.0
    kappa_ve_lu = 0.0
    tau_d_solvent = tau_d
    if ve_enabled:
        lambda_p_lu = phys.lambda_p / dt
        mu_p_lu = phys.mu_p * dt / (phys.rho_d * dx**2)
        # Solvent fraction: mu_s = mu_d - mu_p
        mu_s = phys.mu_d - phys.mu_p
        beta_visc = mu_s / phys.mu_d
        # Solvent-only tau for disperse phase
        nu_s = mu_s / phys.rho_d
        nu_s_lu = nu_s * dt / dx**2
        tau_d_solvent = 3.0 * nu_s_lu + 0.5
        kappa_ve_lu = (phys.kappa_ve or 0.0) * dt / dx**2

    return LBMUnits(
        tau_c=tau_c,
        tau_d=tau_d,
        dx=dx,
        dt=dt,
        rho_ratio=rho_ratio,
        sigma_lbm=sigma_lbm,
        kappa=kappa_lu,
        beta=beta_lu,
        mobility=mobility,
        interface_width=interface_width,
        D_s_lu=D_s_lu,
        D_bulk_lu=D_bulk_lu,
        psi_inf_lu=psi_inf_lu,
        E0=E0,
        k_a_lu=k_a_lu,
        k_d_lu=k_d_lu,
        C_inlet_lu=C_inlet_lu,
        surfactant_enabled=surf_enabled,
        lambda_p_lu=lambda_p_lu,
        mu_p_lu=mu_p_lu,
        beta_visc=beta_visc,
        kappa_ve_lu=kappa_ve_lu,
        tau_d_solvent=tau_d_solvent,
        viscoelastic_enabled=ve_enabled,
    )


# ── Sparse equilibrium ──────────────────────────────────────────────────────

def _equilibrium(rho, ux, uy):
    """Compute f_eq.  Shapes: (n_fluid,) → (9, n_fluid)."""
    eu = _EX[:, None] * ux[None] + _EY[:, None] * uy[None]
    usq = ux**2 + uy**2
    feq = _W[:, None] * rho[None] * (
        1.0 + eu / _CS2 + 0.5 * eu**2 / _CS2**2 - 0.5 * usq[None] / _CS2
    )
    return feq


# ── Sparse macroscopic fields ───────────────────────────────────────────────

def _macroscopic(f):
    """rho = Σf_i, u = Σ(e_i f_i) / rho.  f: (9, n_fluid)."""
    rho = f.sum(axis=0)
    ux = (f * _EX[:, None]).sum(axis=0) / rho
    uy = (f * _EY[:, None]).sum(axis=0) / rho
    return rho, ux, uy


# ── Sparse collision (BGK + Guo force) ──────────────────────────────────────

def _collision(f, rho, ux, uy, tau, fx, fy):
    """BGK collision with Guo forcing. All arrays (n_fluid,) or (9, n_fluid)."""
    feq = _equilibrium(rho, ux, uy)
    omega = 1.0 / tau

    eu = _EX[:, None] * ux[None] + _EY[:, None] * uy[None]
    term_x = (_EX[:, None] - ux[None]) / _CS2 + eu * _EX[:, None] / _CS2**2
    term_y = (_EY[:, None] - uy[None]) / _CS2 + eu * _EY[:, None] / _CS2**2
    Si = (1.0 - 0.5 * omega[None]) * _W[:, None] * (
        term_x * fx[None] + term_y * fy[None]
    )

    return f - omega[None] * (f - feq) + Si


# ── Sparse streaming + bounce-back (merged) ─────────────────────────────────

def _stream_bb(f_coll, pull_src, pull_bb):
    """Pull-based streaming with inline bounce-back.

    f_coll: (9, n_fluid) — post-collision distributions
    pull_src: (9, n_fluid) int — source fluid index per direction
    pull_bb: (9, n_fluid) bool — True where source is solid → bounce-back
    """
    # Pull from post-collision at source nodes
    pulled = jnp.stack([f_coll[i][pull_src[i]] for i in range(9)])
    # Bounce-back on link: particle heading opp(i) from current cell
    # hit the wall and returned as direction i → use f_coll[opp(i)](self)
    bounced = f_coll[_OPP]
    return jnp.where(pull_bb, bounced, pulled)


# ── Sparse phase-field operators ─────────────────────────────────────────────

def _laplacian(field, nbr8, nbr8_solid, wall_value=1.0):
    """Isotropic ∇² using D2Q9-weighted 8-neighbor stencil.

    nbr8 order: [0]=E, [1]=W, [2]=N, [3]=S, [4]=NE, [5]=NW, [6]=SE, [7]=SW
    wall_value: value to substitute for solid neighbors.
        phi → 1.0 (solid = oil), mu → 0.0 (phi=1 is double-well minimum)
    """
    f_nbr = field[nbr8]  # (8, n_fluid)
    f_nbr = jnp.where(nbr8_solid, wall_value, f_nbr)
    # Cardinal (w=1/9): weight 2/3 each, diagonal (w=1/36): weight 1/6 each
    # ∇²φ = Σ 2·w_i·[φ_i - φ]/cs² = 6·Σ w_i·[φ_i - φ]
    card = f_nbr[:4].sum(axis=0)  # E+W+N+S
    diag = f_nbr[4:].sum(axis=0)  # NE+NW+SE+SW
    return (2.0 / 3.0) * card + (1.0 / 6.0) * diag - (10.0 / 3.0) * field


def _grad(field, nbr8, nbr8_solid, wall_value=1.0):
    """Isotropic gradient using D2Q9-weighted 8-neighbor stencil.

    nbr8 order: [0]=E, [1]=W, [2]=N, [3]=S, [4]=NE, [5]=NW, [6]=SE, [7]=SW
    wall_value: value to substitute for solid neighbors.
    """
    f_nbr = field[nbr8]  # (8, n_fluid)
    f_nbr = jnp.where(nbr8_solid, wall_value, f_nbr)
    # ∇_x φ = Σ w_i·e_xi·φ_i / cs² = 3·Σ w_i·e_xi·φ_i
    #        = 1/3·(E-W) + 1/12·(NE+SE-NW-SW)
    dfdx = (1.0 / 3.0) * (f_nbr[0] - f_nbr[1]) + \
           (1.0 / 12.0) * (f_nbr[4] + f_nbr[6] - f_nbr[5] - f_nbr[7])
    # ∇_y φ = 1/3·(N-S) + 1/12·(NE+NW-SE-SW)
    dfdy = (1.0 / 3.0) * (f_nbr[2] - f_nbr[3]) + \
           (1.0 / 12.0) * (f_nbr[4] + f_nbr[5] - f_nbr[6] - f_nbr[7])
    return dfdx, dfdy


def _divergence(fx, fy, nbr8, nbr8_solid, wall_value_x=0.0, wall_value_y=0.0):
    """∇·F = ∂Fx/∂x + ∂Fy/∂y using D2Q9-weighted stencil."""
    fx_nbr = fx[nbr8]
    fx_nbr = jnp.where(nbr8_solid, wall_value_x, fx_nbr)
    fy_nbr = fy[nbr8]
    fy_nbr = jnp.where(nbr8_solid, wall_value_y, fy_nbr)

    dFx_dx = (1.0 / 3.0) * (fx_nbr[0] - fx_nbr[1]) + \
             (1.0 / 12.0) * (fx_nbr[4] + fx_nbr[6] - fx_nbr[5] - fx_nbr[7])
    dFy_dy = (1.0 / 3.0) * (fy_nbr[2] - fy_nbr[3]) + \
             (1.0 / 12.0) * (fy_nbr[4] + fy_nbr[5] - fy_nbr[6] - fy_nbr[7])

    return dFx_dx + dFy_dy


def _chemical_potential(phi, kappa, beta, nbr8, nbr8_solid, phi_wall=1.0):
    """μ = 2βφ(1-φ)(1-2φ) - κ∇²φ  (Allen-Cahn double-well, F=βφ²(1-φ)²)."""
    mu = 2.0 * beta * phi * (1.0 - phi) * (1.0 - 2.0 * phi) - kappa * _laplacian(phi, nbr8, nbr8_solid, phi_wall)
    return mu


def _allen_cahn_step(phi, ux, uy, mu, mobility, nbr8, nbr8_solid,
                     phi_wall=1.0, interface_width=4):
    """Conservative Allen-Cahn step.

    ∂φ/∂t + ∇·(φu) = M∇²μ

    The explicit update is clipped for boundedness and then corrected for the
    clipping drift before inlet/outlet phase BCs are applied.
    """
    # Conservative advection: ∇·(φu)
    adv = _divergence(phi * ux, phi * uy, nbr8, nbr8_solid)

    # Chemical-potential diffusion (mu wall_value=0: φ=1 double-well minimum).
    diff = mobility * _laplacian(mu, nbr8, nbr8_solid, wall_value=0.0)

    phi_new = jnp.clip(phi - adv + diff, 0.0, 1.0)

    # Correct numerical clipping drift before inlet/outlet phase BCs are applied.
    mass_err = phi.sum() - phi_new.sum()
    add_capacity = jnp.clip(1.0 - phi_new, 0.0, None)
    remove_capacity = jnp.clip(phi_new, 0.0, None)
    capacity = jnp.where(mass_err > 0.0, add_capacity, remove_capacity)
    capacity_sum = jnp.clip(capacity.sum(), 1e-30, None)
    phi_new = phi_new + mass_err * capacity / capacity_sum
    return jnp.clip(phi_new, 0.0, 1.0)


def _surface_tension_force(phi, mu, nbr8, nbr8_solid, phi_wall=1.0):
    """Surface tension body force F = μ ∇φ."""
    dphidx, dphidy = _grad(phi, nbr8, nbr8_solid, wall_value=phi_wall)
    return mu * dphidx, mu * dphidy


# ── Surfactant transport + Marangoni ───────────────────────────────────────

def _sigma_local(phi, psi, sigma_lbm, E0, psi_inf_lu):
    """Langmuir equation of state: σ(ψ) = σ₀·[1 + E₀·ln(1 - ψ/ψ_inf)].
    Returns per-node σ. Active only at interface (|∇φ|>0 region)."""
    ratio = jnp.clip(psi / psi_inf_lu, 0.0, 0.999)
    sigma = sigma_lbm * (1.0 + E0 * jnp.log(1.0 - ratio))
    return jnp.clip(sigma, 0.0, sigma_lbm * 10.0)


def _surfactant_step(psi, ux, uy, phi, D_s_lu, nbr8, nbr8_solid, phi_wall=1.0):
    """Interfacial surfactant transport (FD, explicit Euler).
    ∂ψ/∂t + u·∇ψ = D_s·∇²ψ + sharpening flux.
    Sharpening confines ψ to the diffuse interface."""
    # Advection
    dpsi_dx, dpsi_dy = _grad(psi, nbr8, nbr8_solid, wall_value=0.0)
    advection = ux * dpsi_dx + uy * dpsi_dy

    # Diffusion along interface
    diffusion = D_s_lu * _laplacian(psi, nbr8, nbr8_solid, wall_value=0.0)

    # Sharpening: confine ψ to interface region using |∇φ|
    dphi_dx, dphi_dy = _grad(phi, nbr8, nbr8_solid, wall_value=phi_wall)
    grad_phi_mag = jnp.sqrt(dphi_dx**2 + dphi_dy**2 + 1e-30)
    # Interface indicator: peaks at diffuse interface
    interface_w = 4.0 * phi * (1.0 - phi)
    # Anti-sharpening term pushes ψ away from bulk toward interface
    psi_target = psi * interface_w / jnp.clip(interface_w.mean(), 1e-30, None)
    sharpen = 0.1 * D_s_lu * (psi_target - psi) * grad_phi_mag

    psi_new = psi - advection + diffusion + sharpen
    return jnp.clip(psi_new, 0.0, None)


def _bulk_surfactant_step(C, ux, uy, D_bulk_lu, nbr8, nbr8_solid):
    """Bulk surfactant advection-diffusion: ∂C/∂t + u·∇C = D_bulk·∇²C."""
    dC_dx, dC_dy = _grad(C, nbr8, nbr8_solid, wall_value=0.0)
    advection = ux * dC_dx + uy * dC_dy
    diffusion = D_bulk_lu * _laplacian(C, nbr8, nbr8_solid, wall_value=0.0)
    C_new = C - advection + diffusion
    return jnp.clip(C_new, 0.0, None)


def _adsorption_desorption(psi, C, phi, k_a_lu, k_d_lu, psi_inf_lu,
                           nbr8, nbr8_solid, phi_wall=1.0):
    """Langmuir kinetics: j = k_a·C_int·(ψ_inf - ψ) - k_d·ψ.
    Couples interfacial ψ and bulk C at the diffuse interface."""
    dphi_dx, dphi_dy = _grad(phi, nbr8, nbr8_solid, wall_value=phi_wall)
    grad_phi_mag = jnp.sqrt(dphi_dx**2 + dphi_dy**2 + 1e-30)
    # Interface indicator: significant |∇φ| marks interface cells
    interface_mask = grad_phi_mag / jnp.clip(grad_phi_mag.max(), 1e-30, None)

    # Source term (active at interface)
    j = (k_a_lu * C * (psi_inf_lu - psi) - k_d_lu * psi) * interface_mask

    psi_new = jnp.clip(psi + j, 0.0, psi_inf_lu)
    # Mass conservation: bulk loses what interface gains
    C_new = jnp.clip(C - j, 0.0, None)
    return psi_new, C_new


def _marangoni_force(phi, sigma_local, nbr8, nbr8_solid, phi_wall=1.0):
    """Marangoni force: tangential gradient of σ at interface.
    F_ma = [∇σ - (∇σ·n)n] · |∇φ|  where n = ∇φ/|∇φ|."""
    dsig_dx, dsig_dy = _grad(sigma_local, nbr8, nbr8_solid, wall_value=sigma_local.mean())
    dphi_dx, dphi_dy = _grad(phi, nbr8, nbr8_solid, wall_value=phi_wall)
    grad_phi_mag = jnp.sqrt(dphi_dx**2 + dphi_dy**2 + 1e-30)

    # Unit normal
    nx = dphi_dx / grad_phi_mag
    ny = dphi_dy / grad_phi_mag

    # Project out normal component of ∇σ → tangential ∇_s(σ)
    dsig_n = dsig_dx * nx + dsig_dy * ny
    fx_ma = (dsig_dx - dsig_n * nx) * grad_phi_mag
    fy_ma = (dsig_dy - dsig_n * ny) * grad_phi_mag

    return fx_ma, fy_ma


def _apply_surfactant_bc(psi, C, bc_map_fluid, inlet_data, outlet_mask, outlet_upstream,
                         C_inlet=0.0):
    """Surfactant BCs: inlet C=C_inlet, ψ=0; outlet: Neumann."""
    for type_id, _phi_val, _rho_in in inlet_data:
        mask = (bc_map_fluid == type_id)
        psi = jnp.where(mask, 0.0, psi)
        C = jnp.where(mask, C_inlet, C)

    # Outlet: Neumann (copy from upstream)
    psi = jnp.where(outlet_mask, psi[outlet_upstream], psi)
    C = jnp.where(outlet_mask, C[outlet_upstream], C)

    return psi, C


# ── Viscoelasticity (Oldroyd-B, disperse phase) ───────────────────────────

def _conformation_step(A_xx, A_xy, A_yy, ux, uy, phi,
                       lambda_p_lu, kappa_ve_lu, nbr8, nbr8_solid):
    """Explicit Euler for upper-convected Maxwell: conformation tensor A.
    ∂A/∂t + u·∇A = A·∇u + (∇u)ᵀ·A - (1/λ)(A-I) + κ_ve·∇²A.
    Active in disperse phase (phi→0). Wall BC: A=I."""
    # Velocity gradients
    dux_dx, dux_dy = _grad(ux, nbr8, nbr8_solid, wall_value=0.0)
    duy_dx, duy_dy = _grad(uy, nbr8, nbr8_solid, wall_value=0.0)

    # Advection of A components
    dAxx_dx, dAxx_dy = _grad(A_xx, nbr8, nbr8_solid, wall_value=1.0)
    dAxy_dx, dAxy_dy = _grad(A_xy, nbr8, nbr8_solid, wall_value=0.0)
    dAyy_dx, dAyy_dy = _grad(A_yy, nbr8, nbr8_solid, wall_value=1.0)

    adv_xx = ux * dAxx_dx + uy * dAxx_dy
    adv_xy = ux * dAxy_dx + uy * dAxy_dy
    adv_yy = ux * dAyy_dx + uy * dAyy_dy

    # Upper-convected derivative: A·∇u + (∇u)ᵀ·A
    stretch_xx = 2.0 * (A_xx * dux_dx + A_xy * dux_dy)
    stretch_xy = A_xx * duy_dx + A_xy * duy_dy + A_xy * dux_dx + A_yy * dux_dy
    stretch_yy = 2.0 * (A_xy * duy_dx + A_yy * duy_dy)

    # Relaxation: -(1/λ)(A - I)
    inv_lambda = 1.0 / lambda_p_lu
    relax_xx = -inv_lambda * (A_xx - 1.0)
    relax_xy = -inv_lambda * A_xy
    relax_yy = -inv_lambda * (A_yy - 1.0)

    # Artificial diffusion for stability
    diff_xx = kappa_ve_lu * _laplacian(A_xx, nbr8, nbr8_solid, wall_value=1.0)
    diff_xy = kappa_ve_lu * _laplacian(A_xy, nbr8, nbr8_solid, wall_value=0.0)
    diff_yy = kappa_ve_lu * _laplacian(A_yy, nbr8, nbr8_solid, wall_value=1.0)

    # Disperse-phase mask: (1-phi) → 1 in disperse, 0 in continuous
    disp = 1.0 - phi

    A_xx_new = A_xx + disp * (-adv_xx + stretch_xx + relax_xx + diff_xx)
    A_xy_new = A_xy + disp * (-adv_xy + stretch_xy + relax_xy + diff_xy)
    A_yy_new = A_yy + disp * (-adv_yy + stretch_yy + relax_yy + diff_yy)

    # Reset to identity in continuous phase
    A_xx_new = jnp.where(phi > 0.99, 1.0, A_xx_new)
    A_xy_new = jnp.where(phi > 0.99, 0.0, A_xy_new)
    A_yy_new = jnp.where(phi > 0.99, 0.0, A_yy_new)

    return A_xx_new, A_xy_new, A_yy_new


def _polymer_stress_force(A_xx, A_xy, A_yy, phi, mu_p_lu, lambda_p_lu,
                          nbr8, nbr8_solid):
    """div(τ_p) as body force, masked to disperse phase.
    τ_p = (μ_p/λ_p)(A - I) · (1-φ)."""
    coeff = mu_p_lu / lambda_p_lu
    disp = 1.0 - phi

    tau_xx = coeff * (A_xx - 1.0) * disp
    tau_xy = coeff * A_xy * disp
    tau_yy = coeff * (A_yy - 1.0) * disp

    dtxx_dx, _ = _grad(tau_xx, nbr8, nbr8_solid, wall_value=0.0)
    dtxy_dx, dtxy_dy = _grad(tau_xy, nbr8, nbr8_solid, wall_value=0.0)
    _, dtyy_dy = _grad(tau_yy, nbr8, nbr8_solid, wall_value=0.0)

    fx_p = dtxx_dx + dtxy_dy
    fy_p = dtxy_dx + dtyy_dy

    return fx_p, fy_p


def _apply_conformation_bc(A_xx, A_xy, A_yy, bc_map_fluid, inlet_data,
                           outlet_mask, outlet_upstream):
    """Conformation BCs: inlet A=I; outlet: Neumann."""
    for type_id, _phi_val, _rho_in in inlet_data:
        mask = (bc_map_fluid == type_id)
        A_xx = jnp.where(mask, 1.0, A_xx)
        A_xy = jnp.where(mask, 0.0, A_xy)
        A_yy = jnp.where(mask, 1.0, A_yy)

    A_xx = jnp.where(outlet_mask, A_xx[outlet_upstream], A_xx)
    A_xy = jnp.where(outlet_mask, A_xy[outlet_upstream], A_xy)
    A_yy = jnp.where(outlet_mask, A_yy[outlet_upstream], A_yy)

    return A_xx, A_xy, A_yy


# ── Sparse inlet/outlet BCs ────────────────────────────────────────────────

def _apply_f_bc(f, bc_map_fluid, inlet_data, outlet_mask, outlet_upstream,
                outlet_pressure=False, rho_target=1.0):
    """Apply inlet equilibrium and outlet BCs to distributions only.

    Inlets: pressure-driven — fixed rho (from user pressure), velocity from
    current flow state.  Mirror of the pressure outlet approach.
    """
    n = f.shape[1]

    # Compute current macroscopic fields once for all inlets
    rho_cur, ux_cur, uy_cur = _macroscopic(f)

    for type_id, _phi_val, rho_in in inlet_data:
        mask = (bc_map_fluid == type_id)
        rho_bc = jnp.where(mask, rho_in, rho_cur)
        feq = _equilibrium(rho_bc, ux_cur, uy_cur)
        f = jnp.where(mask[None], feq, f)

    if outlet_pressure:
        # Pressure outlet: equilibrium with target density, velocity from upstream
        rho_up, ux_up, uy_up = _macroscopic(f)
        ux_out = ux_up[outlet_upstream]
        uy_out = uy_up[outlet_upstream]
        rho_out = jnp.full(n, rho_target, dtype=jnp.float64)
        feq_out = _equilibrium(rho_out, ux_out, uy_out)
        f = jnp.where(outlet_mask[None], feq_out, f)
    else:
        # Neumann: copy from upstream neighbor
        f_up = f[:, outlet_upstream]
        f = jnp.where(outlet_mask[None], f_up, f)

    return f


def _apply_phi_bc(phi, bc_map_fluid, inlet_data, outlet_mask, outlet_upstream):
    """Enforce phi at inlet/outlet cells. Must be called AFTER Allen-Cahn step."""
    for type_id, phi_val, _rho_in in inlet_data:
        mask = (bc_map_fluid == type_id)
        phi = jnp.where(mask, phi_val, phi)

    # Outlet: copy from upstream neighbor
    phi_up = phi[outlet_upstream]
    phi = jnp.where(outlet_mask, phi_up, phi)

    return phi


# ── Simulation class ─────────────────────────────────────────────────────────

class TwoPhaseSim:
    def __init__(
        self,
        geometry: Geometry2D,
        phys: PhysParams,
        tau_c: float = 0.55,
        interface_width: int = 4,
        mobility: float = 0.1,
        delta_rho_max: float = 0.005,
    ):
        self.geom = geometry
        self.phys = phys
        self.units = convert_units(phys, geometry.dx_um, tau_c, interface_width, mobility)

        sp = geometry.sparse
        assert sp is not None, "Geometry2D must have sparse index maps (call build_sparse_maps)"

        self.n_fluid = sp.n_fluid
        self.ny, self.nx = geometry.shape
        self.fluid_y = jnp.array(sp.fluid_yx[:, 0])
        self.fluid_x = jnp.array(sp.fluid_yx[:, 1])

        # Sparse index arrays → JAX
        self.pull_src = jnp.array(sp.pull_src)
        self.pull_bb = jnp.array(sp.pull_bb)
        self.nbr4 = jnp.array(sp.nbr4)
        self.nbr4_solid = jnp.array(sp.nbr4_solid)
        self.nbr8 = jnp.array(sp.nbr8)
        self.nbr8_solid = jnp.array(sp.nbr8_solid)
        self.bc_map_fluid = jnp.array(sp.bc_map_fluid)
        self.outlet_mask = jnp.array(sp.outlet_mask)
        self.outlet_upstream = jnp.array(sp.outlet_upstream)

        # Pre-compute inlet data in LU: pressure → rho_lbm
        # Direct: delta_rho = 3 * P_Pa / (rho_phys * (dx/dt)^2)
        # Physical pressures (hundreds of mbar) give delta_rho >> 0.01,
        # violating the low-Mach constraint.  We cap only the lattice pressure
        # drive; material properties stay tied to the physical parameters.
        dx_m = self.units.dx
        dt_s = self.units.dt
        rho_phys = phys.rho_c
        lattice_v2 = (dx_m / dt_s) ** 2
        inlet_specs = geometry.inlet_specs()
        self.inlet_data = []
        # Compute physical delta_rho for each inlet
        drho_phys = []
        for spec in inlet_specs:
            p_pa = spec.pressure_mbar * 100.0
            drho_phys.append(3.0 * p_pa / (rho_phys * lattice_v2))
        drho_max = max(drho_phys) if drho_phys else 0.0
        if drho_max > delta_rho_max:
            alpha = delta_rho_max / drho_max  # < 1
            self.units = replace(self.units, pressure_scale=alpha)
        else:
            alpha = 1.0
            self.units = replace(self.units, pressure_scale=alpha)
        for spec, drho in zip(inlet_specs, drho_phys):
            rho_in = 1.0 + drho * alpha
            self.inlet_data.append((spec.type_id, spec.phi, rho_in))

        # Outlet BC mode (use first outlet spec's settings, default to Neumann)
        outlet_specs = geometry.outlet_specs()
        if outlet_specs and outlet_specs[0].outlet_bc == "pressure":
            self.outlet_pressure = True
            self.rho_target = outlet_specs[0].rho_target
        else:
            self.outlet_pressure = False
            self.rho_target = 1.0

        # Wall wetting BC: phi_wall from contact angle (Ding & Spelt 2007)
        # θ measured through aqueous phase: 150° = hydrophobic (oil-wet)
        # θ=180° → phi_wall=1 (fully oil-wet), θ=90° → 0.5, θ=0° → 0 (aqueous-wet)
        theta_rad = np.radians(phys.contact_angle_deg)
        self.phi_wall = 0.5 * (1.0 - np.cos(theta_rad))

        # Surfactant config
        self.surfactant_enabled = self.units.surfactant_enabled
        self.C_inlet = self.units.C_inlet_lu

        # Viscoelastic config
        self.viscoelastic_enabled = self.units.viscoelastic_enabled

        # JIT-compile step function once
        self._jit_step = jax.jit(self._step)

    def _tau_field(self, phi):
        """Linear interpolation of relaxation time: τ(φ) = τ_c·φ + τ_d·(1-φ).
        When viscoelastic: τ_d uses solvent-only viscosity."""
        tau_d_eff = (self.units.tau_d_solvent if self.viscoelastic_enabled
                     else self.units.tau_d)
        return self.units.tau_c * phi + tau_d_eff * (1.0 - phi)

    def _init_state(self):
        """Initialize f = equilibrium(rho=1, u=0) and φ = 1 (oil everywhere)."""
        n = self.n_fluid
        rho0 = jnp.ones(n, dtype=jnp.float64)
        ux0 = jnp.zeros(n, dtype=jnp.float64)
        uy0 = jnp.zeros(n, dtype=jnp.float64)
        f = _equilibrium(rho0, ux0, uy0)
        phi = jnp.ones(n, dtype=jnp.float64)
        psi = jnp.zeros(n, dtype=jnp.float64)
        C = jnp.zeros(n, dtype=jnp.float64)
        # Conformation tensor: A = I (identity)
        A_xx = jnp.ones(n, dtype=jnp.float64)
        A_xy = jnp.zeros(n, dtype=jnp.float64)
        A_yy = jnp.ones(n, dtype=jnp.float64)
        return f, phi, psi, C, A_xx, A_xy, A_yy

    def init_state(self, phi_init: np.ndarray | None = None):
        """Initialize state. Returns tuple of active fields."""
        f, phi, psi, C, A_xx, A_xy, A_yy = self._init_state()
        if phi_init is not None:
            phi_dense = np.asarray(phi_init, dtype=np.float64)
            phi_sparse = phi_dense[
                np.asarray(self.fluid_y), np.asarray(self.fluid_x)
            ]
            phi = jnp.array(phi_sparse)
        surf = self.surfactant_enabled
        ve = self.viscoelastic_enabled
        if surf and ve:
            return f, phi, psi, C, A_xx, A_xy, A_yy
        elif surf:
            return f, phi, psi, C
        elif ve:
            return f, phi, A_xx, A_xy, A_yy
        return f, phi

    def _step(self, f, phi, psi, C, A_xx, A_xy, A_yy):
        """One LBM time step (sparse), with optional extensions.

        Step ordering (Liang et al. 2018 + surfactant + viscoelastic):
         1.  Macroscopic fields from f
         2.  σ_local from ψ (Langmuir EOS)                      [surfactant]
         3.  Local κ(σ), β(σ) from σ_local                      [surfactant]
         4.  Chemical potential with (local) κ, β
         5.  Capillary force F = μ∇φ
         6.  Marangoni force F_ma = ∇_s(σ)·|∇φ|                 [surfactant]
         7.  Polymer stress divergence → f_polymer               [viscoelastic]
         8.  Total force
         9.  Force-corrected velocity
        10.  BGK collision with Guo forcing
        11.  Streaming + bounce-back
        12.  Distribution BCs
        13.  Allen-Cahn phase update
        14.  Phase-field BCs
        15.  Surfactant transport (FD)                           [surfactant]
        16.  Conformation tensor evolution                       [viscoelastic]
        17.  Conformation BCs                                    [viscoelastic]
        """
        u = self.units

        # 1. Macroscopic (bare velocity from distributions)
        rho, ux, uy = _macroscopic(f)

        # 2-3. Local σ → local κ, β (surfactant modulates interface energy)
        if u.surfactant_enabled:
            sig_loc = _sigma_local(psi, psi, u.sigma_lbm, u.E0, u.psi_inf_lu)
            W_lu = float(u.interface_width)
            kappa_loc = 1.5 * sig_loc * W_lu
            beta_loc = 12.0 * sig_loc / W_lu
        else:
            kappa_loc = u.kappa
            beta_loc = u.beta

        # 4. Chemical potential
        mu = _chemical_potential(phi, kappa_loc, beta_loc,
                                 self.nbr8, self.nbr8_solid, self.phi_wall)

        # 5. Capillary force
        fx, fy = _surface_tension_force(phi, mu, self.nbr8, self.nbr8_solid, self.phi_wall)

        # 6. Marangoni force
        if u.surfactant_enabled:
            fx_ma, fy_ma = _marangoni_force(phi, sig_loc,
                                            self.nbr8, self.nbr8_solid, self.phi_wall)
            fx = fx + fx_ma
            fy = fy + fy_ma

        # 7. Polymer stress divergence
        if u.viscoelastic_enabled:
            fx_p, fy_p = _polymer_stress_force(A_xx, A_xy, A_yy, phi,
                                               u.mu_p_lu, u.lambda_p_lu,
                                               self.nbr8, self.nbr8_solid)
            fx = fx + fx_p
            fy = fy + fy_p

        # 9. Force-corrected velocity (Guo scheme: u_phys = u_bare + 0.5·F/ρ)
        ux_c = ux + 0.5 * fx / rho
        uy_c = uy + 0.5 * fy / rho

        # 10. Collision (uses bare velocity — Guo Si term handles the correction)
        tau = self._tau_field(phi)
        f = _collision(f, rho, ux, uy, tau, fx, fy)

        # 11. Streaming + bounce-back
        f = _stream_bb(f, self.pull_src, self.pull_bb)

        # 12. Distribution BCs
        f = _apply_f_bc(
            f, self.bc_map_fluid, self.inlet_data,
            self.outlet_mask, self.outlet_upstream,
            outlet_pressure=self.outlet_pressure,
            rho_target=self.rho_target,
        )

        # 13. Allen-Cahn phase-field update (recompute μ with potentially local κ,β)
        mu = _chemical_potential(phi, kappa_loc, beta_loc,
                                 self.nbr8, self.nbr8_solid, self.phi_wall)
        phi = _allen_cahn_step(phi, ux_c, uy_c, mu, u.mobility,
                               self.nbr8, self.nbr8_solid, self.phi_wall,
                               interface_width=u.interface_width)

        # 14. Phase-field BCs
        phi = _apply_phi_bc(
            phi, self.bc_map_fluid, self.inlet_data,
            self.outlet_mask, self.outlet_upstream,
        )

        # 15. Surfactant transport
        if u.surfactant_enabled:
            psi = _surfactant_step(psi, ux_c, uy_c, phi, u.D_s_lu,
                                   self.nbr8, self.nbr8_solid, self.phi_wall)
            C = _bulk_surfactant_step(C, ux_c, uy_c, u.D_bulk_lu,
                                      self.nbr8, self.nbr8_solid)
            psi, C = _adsorption_desorption(psi, C, phi, u.k_a_lu, u.k_d_lu,
                                            u.psi_inf_lu, self.nbr8,
                                            self.nbr8_solid, self.phi_wall)
            psi, C = _apply_surfactant_bc(psi, C, self.bc_map_fluid,
                                          self.inlet_data, self.outlet_mask,
                                          self.outlet_upstream, self.C_inlet)

        # 16-17. Conformation tensor evolution
        if u.viscoelastic_enabled:
            A_xx, A_xy, A_yy = _conformation_step(
                A_xx, A_xy, A_yy, ux_c, uy_c, phi,
                u.lambda_p_lu, u.kappa_ve_lu,
                self.nbr8, self.nbr8_solid,
            )
            A_xx, A_xy, A_yy = _apply_conformation_bc(
                A_xx, A_xy, A_yy, self.bc_map_fluid,
                self.inlet_data, self.outlet_mask, self.outlet_upstream,
            )

        return f, phi, psi, C, A_xx, A_xy, A_yy

    def step(self, f, phi, psi=None, C=None, A_xx=None, A_xy=None, A_yy=None):
        """One JIT-compiled LBM time step. Returns active fields matching input."""
        n = self.n_fluid
        z = jnp.zeros(n, dtype=jnp.float64)
        o = jnp.ones(n, dtype=jnp.float64)
        if psi is None:
            psi = z
        if C is None:
            C = z
        if A_xx is None:
            A_xx = o
        if A_xy is None:
            A_xy = z
        if A_yy is None:
            A_yy = o
        f, phi, psi, C, A_xx, A_xy, A_yy = self._jit_step(
            f, phi, psi, C, A_xx, A_xy, A_yy
        )
        surf = self.surfactant_enabled
        ve = self.viscoelastic_enabled
        if surf and ve:
            return f, phi, psi, C, A_xx, A_xy, A_yy
        elif surf:
            return f, phi, psi, C
        elif ve:
            return f, phi, A_xx, A_xy, A_yy
        return f, phi

    @staticmethod
    def macroscopic(f):
        """Extract (rho, ux, uy) from distributions."""
        return _macroscopic(f)

    def run(
        self,
        n_steps: int,
        callback: Callable | None = None,
        callback_interval: int = 100,
    ) -> dict:
        """Run the simulation.

        callback(step, phi, rho, ux, uy) is called every callback_interval steps.
        Returns dict with final state arrays (as numpy).
        """
        f, phi, psi, C, A_xx, A_xy, A_yy = self._init_state()

        for t in range(n_steps):
            f, phi, psi, C, A_xx, A_xy, A_yy = self._jit_step(
                f, phi, psi, C, A_xx, A_xy, A_yy
            )

            if callback and (t + 1) % callback_interval == 0:
                rho, ux, uy = _macroscopic(f)
                callback(
                    t + 1,
                    np.asarray(phi),
                    np.asarray(rho),
                    np.asarray(ux),
                    np.asarray(uy),
                )

        rho, ux, uy = _macroscopic(f)
        result = {
            "phi": np.asarray(phi),
            "rho": np.asarray(rho),
            "ux": np.asarray(ux),
            "uy": np.asarray(uy),
            "f": np.asarray(f),
            "units": self.units,
        }
        if self.surfactant_enabled:
            result["psi"] = np.asarray(psi)
            result["C"] = np.asarray(C)
        if self.viscoelastic_enabled:
            result["A_xx"] = np.asarray(A_xx)
            result["A_xy"] = np.asarray(A_xy)
            result["A_yy"] = np.asarray(A_yy)
        return result
