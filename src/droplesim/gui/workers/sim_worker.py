"""QThread simulation loop with pause/resume support."""

from __future__ import annotations

import logging
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from droplesim.solver.sim import TwoPhaseSim

log = logging.getLogger(__name__)


class SimWorker(QThread):
    frame_ready = Signal(int, object, object, object, object, float, float)
    # Emitted right before run() exits so MainWindow can save state for resume
    state_saved = Signal(int, object, object)  # step, f, phi

    def __init__(
        self,
        sim: TwoPhaseSim,
        phi_init: np.ndarray | None = None,
        f_resume: object = None,
        phi_resume: object = None,
        start_step: int = 0,
        emit_interval: int = 50,
    ):
        super().__init__()
        self._sim = sim
        self._phi_init = phi_init
        self._f_resume = f_resume
        self._phi_resume = phi_resume
        self._start_step = start_step
        self._emit_interval = emit_interval
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        log.info("SimWorker started (emit every %d steps, from step %d)",
                 self._emit_interval, self._start_step)

        if self._f_resume is not None:
            f, phi = self._f_resume, self._phi_resume
        else:
            f, phi = self._sim.init_state(phi_init=self._phi_init)

        step = self._start_step
        t0 = time.perf_counter()

        while not self._stop_requested:
            f, phi = self._sim.step(f, phi)
            step += 1
            if step % self._emit_interval == 0:
                rho, ux, uy = self._sim.macroscopic(f)
                elapsed = time.perf_counter() - t0
                total_steps = step - self._start_step
                n_cells = self._sim.n_fluid
                mlups = total_steps * n_cells / elapsed / 1e6 if elapsed > 0 else 0.0
                self.frame_ready.emit(
                    step,
                    np.asarray(phi),
                    np.asarray(rho),
                    np.asarray(ux),
                    np.asarray(uy),
                    elapsed,
                    mlups,
                )
                if step % (self._emit_interval * 20) == 0:
                    log.info("Step %d  MLUPS=%.1f  elapsed=%.1fs", step, mlups, elapsed)

        elapsed = time.perf_counter() - t0
        log.info("SimWorker stopped after %d steps (%.1fs)", step, elapsed)
        # Save state for resume before thread exits
        self.state_saved.emit(step, np.asarray(f), np.asarray(phi))
