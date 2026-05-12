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

from dataclasses import dataclass
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


def _chemical_potential(phi, kappa, beta, nbr8, nbr8_solid, phi_wall=1.0):
    """μ = 2βφ(1-φ)(1-2φ) - κ∇²φ  (Allen-Cahn double-well, F=βφ²(1-φ)²)."""
    mu = 2.0 * beta * phi * (1.0 - phi) * (1.0 - 2.0 * phi) - kappa * _laplacian(phi, nbr8, nbr8_solid, phi_wall)
    return mu


def _allen_cahn_step(phi, ux, uy, mu, mobility, nbr8, nbr8_solid, phi_wall=1.0):
    """Explicit Euler step for Allen-Cahn: ∂φ/∂t + u·∇φ = M ∇²μ."""
    dphidx, dphidy = _grad(phi, nbr8, nbr8_solid, wall_value=phi_wall)
    advection = ux * dphidx + uy * dphidy
    # mu wall_value=0: at phi=1 (solid) the double-well gives mu≈0
    diffusion = mobility * _laplacian(mu, nbr8, nbr8_solid, wall_value=0.0)
    phi_new = phi - advection + diffusion
    return jnp.clip(phi_new, 0.0, 1.0)


def _surface_tension_force(phi, mu, nbr8, nbr8_solid, phi_wall=1.0):
    """Surface tension body force F = μ ∇φ."""
    dphidx, dphidy = _grad(phi, nbr8, nbr8_solid, wall_value=phi_wall)
    return mu * dphidx, mu * dphidy


# ── Sparse inlet/outlet BCs ────────────────────────────────────────────────

def _apply_f_bc(f, bc_map_fluid, inlet_data, outlet_mask, outlet_upstream,
                outlet_pressure=False, rho_target=1.0):
    """Apply inlet equilibrium and outlet BCs to distributions only."""
    n = f.shape[1]

    for type_id, _phi_val, ux_lu, uy_lu in inlet_data:
        mask = (bc_map_fluid == type_id)
        rho_in = jnp.ones(n, dtype=jnp.float64)
        feq = _equilibrium(rho_in, ux_lu * jnp.ones(n, dtype=jnp.float64),
                           uy_lu * jnp.ones(n, dtype=jnp.float64))
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
    for type_id, phi_val, _ux_lu, _uy_lu in inlet_data:
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

        # Pre-compute inlet data in LU
        self.inlet_data = []
        for spec in geometry.inlet_specs():
            u_scale = self.units.dt / self.units.dx
            ux_lu = spec.ux * u_scale
            uy_lu = spec.uy * u_scale
            self.inlet_data.append((spec.type_id, spec.phi, ux_lu, uy_lu))

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

        # JIT-compile step function once
        self._jit_step = jax.jit(self._step)

    def _tau_field(self, phi):
        """Linear interpolation of relaxation time: τ(φ) = τ_c·φ + τ_d·(1-φ)."""
        return self.units.tau_c * phi + self.units.tau_d * (1.0 - phi)

    def _init_state(self):
        """Initialize f = equilibrium(rho=1, u=0) and φ = 1 (oil everywhere)."""
        n = self.n_fluid
        rho0 = jnp.ones(n, dtype=jnp.float64)
        ux0 = jnp.zeros(n, dtype=jnp.float64)
        uy0 = jnp.zeros(n, dtype=jnp.float64)
        f = _equilibrium(rho0, ux0, uy0)
        phi = jnp.ones(n, dtype=jnp.float64)
        return f, phi

    def init_state(self, phi_init: np.ndarray | None = None):
        """Initialize (f, phi). If phi_init given (dense ny,nx), extract fluid cells."""
        f, phi = self._init_state()
        if phi_init is not None:
            phi_dense = np.asarray(phi_init, dtype=np.float64)
            phi_sparse = phi_dense[
                np.asarray(self.fluid_y), np.asarray(self.fluid_x)
            ]
            phi = jnp.array(phi_sparse)
        return f, phi

    def _step(self, f, phi):
        """One LBM time step (sparse).

        Step ordering (Liang et al. 2018 style):
        1. Macroscopic fields from f
        2. Chemical potential + surface tension force
        3. Force-corrected velocity for phase advection
        4. BGK collision with Guo forcing
        5. Streaming + bounce-back
        6. Distribution BCs (inlet equilibrium, outlet Neumann)
        7. Allen-Cahn phase update (FD, uses corrected velocity)
        8. Phase-field BCs (re-enforce phi at inlets/outlets)
        """
        # 1. Macroscopic (bare velocity from distributions)
        rho, ux, uy = _macroscopic(f)

        # 2. Phase field forces
        mu = _chemical_potential(phi, self.units.kappa, self.units.beta,
                                 self.nbr8, self.nbr8_solid, self.phi_wall)
        fx, fy = _surface_tension_force(phi, mu, self.nbr8, self.nbr8_solid, self.phi_wall)

        # 3. Force-corrected velocity (Guo scheme: u_phys = u_bare + 0.5·F/ρ)
        ux_c = ux + 0.5 * fx / rho
        uy_c = uy + 0.5 * fy / rho

        # 4. Collision (uses bare velocity — Guo Si term handles the correction)
        tau = self._tau_field(phi)
        f = _collision(f, rho, ux, uy, tau, fx, fy)

        # 5. Streaming + bounce-back
        f = _stream_bb(f, self.pull_src, self.pull_bb)

        # 6. Distribution BCs
        f = _apply_f_bc(
            f, self.bc_map_fluid, self.inlet_data,
            self.outlet_mask, self.outlet_upstream,
            outlet_pressure=self.outlet_pressure,
            rho_target=self.rho_target,
        )

        # 7. Allen-Cahn phase-field update
        mu = _chemical_potential(phi, self.units.kappa, self.units.beta,
                                 self.nbr8, self.nbr8_solid, self.phi_wall)
        phi = _allen_cahn_step(phi, ux_c, uy_c, mu, self.units.mobility,
                               self.nbr8, self.nbr8_solid, self.phi_wall)

        # 8. Phase-field BCs (must be AFTER Allen-Cahn to prevent diffusion leak)
        phi = _apply_phi_bc(
            phi, self.bc_map_fluid, self.inlet_data,
            self.outlet_mask, self.outlet_upstream,
        )

        return f, phi

    def step(self, f, phi):
        """One JIT-compiled LBM time step. Returns (f_new, phi_new)."""
        return self._jit_step(f, phi)

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
        f, phi = self._init_state()

        for t in range(n_steps):
            f, phi = self._jit_step(f, phi)

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
        return {
            "phi": np.asarray(phi),
            "rho": np.asarray(rho),
            "ux": np.asarray(ux),
            "uy": np.asarray(uy),
            "f": np.asarray(f),
            "units": self.units,
        }
