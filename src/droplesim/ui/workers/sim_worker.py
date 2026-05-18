"""QThread simulation loop with pause/resume support."""

from __future__ import annotations

import logging
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from droplesim.solver.sim import TwoPhaseSim

log = logging.getLogger(__name__)


def _np(x):
    return np.asarray(x) if x is not None else None


class SimWorker(QThread):
    # step, phi, rho, ux, uy, elapsed, mlups, extra_fields_dict
    frame_ready = Signal(int, object, object, object, object, float, float, object)
    # step, state_dict (all fields as numpy)
    state_saved = Signal(int, object)
    # emitted when simulation diverges (step, field_name)
    diverged = Signal(int, str)

    def __init__(
        self,
        sim: TwoPhaseSim,
        phi_init: np.ndarray | None = None,
        f_resume: object = None,
        phi_resume: object = None,
        psi_resume: object = None,
        C_resume: object = None,
        A_xx_resume: object = None,
        A_xy_resume: object = None,
        A_yy_resume: object = None,
        start_step: int = 0,
        emit_interval: int = 50,
    ):
        super().__init__()
        self._sim = sim
        self._phi_init = phi_init
        self._f_resume = f_resume
        self._phi_resume = phi_resume
        self._psi_resume = psi_resume
        self._C_resume = C_resume
        self._A_xx_resume = A_xx_resume
        self._A_xy_resume = A_xy_resume
        self._A_yy_resume = A_yy_resume
        self._start_step = start_step
        self._emit_interval = emit_interval
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        log.info("SimWorker started (emit every %d steps, from step %d)",
                 self._emit_interval, self._start_step)
        sim = self._sim
        surf = sim.surfactant_enabled
        ve = sim.viscoelastic_enabled

        if self._f_resume is not None:
            f, phi = self._f_resume, self._phi_resume
            psi, C = self._psi_resume, self._C_resume
            A_xx, A_xy, A_yy = self._A_xx_resume, self._A_xy_resume, self._A_yy_resume
        else:
            state = sim.init_state(phi_init=self._phi_init)
            # Unpack based on enabled extensions
            if surf and ve:
                f, phi, psi, C, A_xx, A_xy, A_yy = state
            elif surf:
                f, phi, psi, C = state
                A_xx, A_xy, A_yy = None, None, None
            elif ve:
                f, phi, A_xx, A_xy, A_yy = state
                psi, C = None, None
            else:
                f, phi = state
                psi, C = None, None
                A_xx, A_xy, A_yy = None, None, None

        step = self._start_step
        t0 = time.perf_counter()

        while not self._stop_requested:
            if surf and ve:
                f, phi, psi, C, A_xx, A_xy, A_yy = sim.step(
                    f, phi, psi, C, A_xx, A_xy, A_yy
                )
            elif surf:
                f, phi, psi, C = sim.step(f, phi, psi, C)
            elif ve:
                f, phi, A_xx, A_xy, A_yy = sim.step(
                    f, phi, A_xx=A_xx, A_xy=A_xy, A_yy=A_yy
                )
            else:
                f, phi = sim.step(f, phi)
            step += 1

            if step % self._emit_interval == 0:
                rho, ux, uy = sim.macroscopic(f)
                rho_np = np.asarray(rho)
                ux_np = np.asarray(ux)
                uy_np = np.asarray(uy)
                phi_np = np.asarray(phi)

                # Divergence check
                diverged_field = None
                if not np.all(np.isfinite(rho_np)):
                    diverged_field = "rho"
                elif not np.all(np.isfinite(ux_np)):
                    diverged_field = "ux"
                elif not np.all(np.isfinite(uy_np)):
                    diverged_field = "uy"
                elif not np.all(np.isfinite(phi_np)):
                    diverged_field = "phi"
                if diverged_field is not None:
                    log.error(
                        "Simulation diverged at step %d (NaN/Inf in %s)",
                        step, diverged_field,
                    )
                    self.diverged.emit(step, diverged_field)
                    break

                elapsed = time.perf_counter() - t0
                total_steps = step - self._start_step
                n_cells = sim.n_fluid
                mlups = total_steps * n_cells / elapsed / 1e6 if elapsed > 0 else 0.0
                extra = {}
                if psi is not None:
                    extra["psi"] = np.asarray(psi)
                    theta, sigma_local = sim.surfactant_fields(psi)
                    extra["theta"] = np.asarray(theta)
                    extra["sigma_local"] = np.asarray(sigma_local)
                if A_xx is not None:
                    extra["A_xx"] = np.asarray(A_xx)
                    extra["A_xy"] = np.asarray(A_xy)
                    extra["A_yy"] = np.asarray(A_yy)
                self.frame_ready.emit(
                    step, phi_np, rho_np, ux_np, uy_np,
                    elapsed, mlups,
                    extra if extra else None,
                )
                if step % (self._emit_interval * 20) == 0:
                    log.info("Step %d  MLUPS=%.1f  elapsed=%.1fs", step, mlups, elapsed)

        elapsed = time.perf_counter() - t0
        log.info("SimWorker stopped after %d steps (%.1fs)", step, elapsed)
        saved = {
            "f": np.asarray(f),
            "phi": np.asarray(phi),
            "psi": _np(psi),
            "C": _np(C),
            "A_xx": _np(A_xx),
            "A_xy": _np(A_xy),
            "A_yy": _np(A_yy),
        }
        self.state_saved.emit(step, saved)
