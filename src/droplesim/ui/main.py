"""QApplication + MainWindow with 4-tab workflow."""

from __future__ import annotations

import logging
import signal
import sys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from droplesim.ui.panels.params_panel import ParamsPanel
from droplesim.ui.state import SessionState
from droplesim.ui.theme import GLOBAL_QSS, configure_pyqtgraph
from droplesim.ui.views.edge_view import EdgeView
from droplesim.ui.views.geometry_view import GeometryView
from droplesim.ui.views.phase_view import PhaseView
from droplesim.ui.views.sim_view import SimView
from droplesim.ui.workers.sim_worker import SimWorker
from droplesim.solver.geometry2d import (
    EdgeSpec,
    assign_edge_bcs,
    build_sparse_maps,
    extract_edges,
    load_polygons,
    rasterize_polygons,
)
from droplesim.solver.sim import PhysParams, TwoPhaseSim

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("droplesim")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 700)

        self._solid_mask = None
        self._origin_um = (0.0, 0.0)
        self._dx_um = 2.5
        self._polygons_mm = None
        self._edge_polylines_mm = None
        self._worker: SimWorker | None = None
        # Resume state: kept between stop→start
        self._current_sim: TwoPhaseSim | None = None
        self._saved_step = 0
        self._saved_f = None
        self._saved_phi = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -- Body: horizontal splitter (params | right panel) --
        body = QSplitter()

        self._params = ParamsPanel()
        self._params.load_geometry_requested.connect(self._on_load_geometry)
        self._params.channel_depth_changed.connect(self._on_channel_depth_changed)
        self._params.save_config_requested.connect(self._on_save_config)
        self._params.load_config_requested.connect(self._on_load_config)
        body.addWidget(self._params)

        # Right panel: stage buttons + stacked views
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        from PySide6.QtWidgets import QSizePolicy

        stage_bar = QWidget()
        stage_bar.setFixedHeight(150)
        stage_row = QHBoxLayout(stage_bar)
        stage_row.setContentsMargins(0, 0, 0, 0)
        stage_row.setSpacing(1)

        self._stage_btns: list[QPushButton] = []
        stage_labels = ["1. Geometry", "2. Edges", "3. Phase", "4. Simulate"]
        for i, label in enumerate(stage_labels):
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._on_stage_clicked(idx))
            stage_row.addWidget(btn, 1)
            self._stage_btns.append(btn)

        right_layout.addWidget(stage_bar)

        self._stack = QStackedWidget()
        right_layout.addWidget(self._stack, stretch=1)

        body.addWidget(right)

        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)

        # Views
        self._geometry_view = GeometryView()
        self._stack.addWidget(self._geometry_view)

        self._edge_view = EdgeView()
        self._edge_view.edges_changed.connect(self._on_edges_changed)
        self._stack.addWidget(self._edge_view)

        self._phase_view = PhaseView()
        self._phase_view.phase_changed.connect(self._on_phase_changed)
        self._stack.addWidget(self._phase_view)

        self._sim_view = SimView()
        self._sim_view.start_requested.connect(self._on_start)
        self._sim_view.stop_requested.connect(self._on_stop)
        self._sim_view.reset_requested.connect(self._on_reset)
        self._stack.addWidget(self._sim_view)

        # Disable stages 2-4 until geometry is loaded
        for i in range(1, 4):
            self._stage_btns[i].setEnabled(False)
        self._stage_btns[0].setChecked(True)
        self._update_stage_styles()

        root.addWidget(body, stretch=1)

        # -- Status bar --
        self._status_bar = self.statusBar()
        self._status_bar.showMessage("Load a DXF file to begin")

        # Timer so Python signal handlers fire
        self._keep_alive = QTimer()
        self._keep_alive.timeout.connect(lambda: None)
        self._keep_alive.start(500)

        log.info("MainWindow initialized")

    def _set_status(self, msg: str, color: str = "#888888"):
        self._status_bar.setStyleSheet(f"color: {color};")
        self._status_bar.showMessage(msg)

    def _on_stage_clicked(self, idx: int):
        if not self._stage_btns[idx].isEnabled():
            return
        self._stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._stage_btns):
            btn.setChecked(i == idx)
        self._update_stage_styles()

    def _update_stage_styles(self):
        from droplesim.ui.theme import Theme
        for i, btn in enumerate(self._stage_btns):
            if btn.isChecked():
                btn.setStyleSheet(
                    f"QPushButton {{ background-color: {Theme.ACCENT}; color: {Theme.TEXT_WHITE};"
                    f" border: none; border-radius: 0; padding: 8px 10px;"
                    f" font-weight: bold; font-size: {Theme.FONT_SIZE_BODY}px; }}"
                )
            elif not btn.isEnabled():
                btn.setStyleSheet(
                    f"QPushButton {{ background-color: {Theme.BG_MEDIUM}; color: {Theme.TEXT_DISABLED};"
                    f" border: none; border-radius: 0; padding: 8px 10px;"
                    f" font-size: {Theme.FONT_SIZE_BODY}px; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ background-color: {Theme.BG_CONTROL}; color: {Theme.TEXT_WHITE};"
                    f" border: none; border-radius: 0; padding: 8px 10px;"
                    f" font-size: {Theme.FONT_SIZE_BODY}px; }}"
                    f"QPushButton:hover {{ background-color: {Theme.BG_CONTROL_HOVER}; }}"
                )

    def _on_load_geometry(self, path: str, dx_um: float):
        log.info("Loading geometry: %s (dx=%.1f µm)", path, dx_um)
        try:
            self._dx_um = dx_um
            polygons_mm, _ = load_polygons(path)
            self._polygons_mm = polygons_mm
            solid_mask, origin_um = rasterize_polygons(polygons_mm, dx_um)
            self._solid_mask = solid_mask
            self._origin_um = origin_um

            edge_polylines_mm = extract_edges(polygons_mm)
            self._edge_polylines_mm = edge_polylines_mm

            self._geometry_view.set_geometry(solid_mask, dx_um, origin_um)
            self._edge_view.set_geometry(solid_mask, dx_um, origin_um, edge_polylines_mm)
            self._phase_view.set_geometry(solid_mask, dx_um, origin_um, edge_polylines_mm)

            # Compute fluid cell indices for sparse view
            n_fluid = int((~solid_mask).sum())
            fluid_yx = np.argwhere(~solid_mask).astype(np.int32)
            self._fluid_yx = fluid_yx
            self._sim_view.set_geometry_info(dx_um, origin_um, solid_mask, fluid_yx)

            ny, nx = solid_mask.shape
            n_total = ny * nx
            pct = 100.0 * n_fluid / n_total
            n_edges = len(edge_polylines_mm)
            msg = (f"{nx}x{ny} grid  |  {n_fluid:,} fluid cells ({pct:.1f}%)"
                   f"  |  {n_edges} edges")
            log.info("Geometry loaded: %s", msg)
            self._set_status(msg, "#27ae60")

            for i in range(1, 4):
                self._stage_btns[i].setEnabled(True)
            self._on_stage_clicked(1)

        except Exception as e:
            log.exception("Failed to load geometry")
            QMessageBox.critical(self, "Error", f"Failed to load geometry:\n{e}")
            self._set_status(f"Error: {e}", "#e74c3c")

    def _on_channel_depth_changed(self, depth_um: float):
        self._edge_view.set_channel_depth(depth_um)

    def _on_edges_changed(self):
        n_bc = sum(1 for e in self._edge_view.get_edges() if e["kind"] != "wall")
        log.info("Edges updated: %d BC edges", n_bc)

    def _on_phase_changed(self):
        n = len(self._phase_view.get_regions())
        log.info("Phase regions updated: %d regions", n)

    def _build_geometry(self):
        from droplesim.solver.geometry2d import BCSpec, Geometry2D

        edges_data = self._edge_view.get_edges()
        edge_specs = []
        for e in edges_data:
            if e["kind"] != "wall":
                spec = EdgeSpec(
                    name=e["name"],
                    kind=e["kind"],
                    points_um=e["points_um"],
                    phi=e.get("phi", 1.0),
                    ux=e.get("ux", 0.0),
                    uy=e.get("uy", 0.0),
                    outlet_bc=e.get("outlet_bc", "pressure"),
                    rho_target=e.get("rho_target", 1.0),
                )
                edge_specs.append(spec)

        bc_map, edge_specs = assign_edge_bcs(
            self._solid_mask, edge_specs, self._dx_um, self._origin_um
        )

        bc_specs = []
        for es in edge_specs:
            pts = np.array(es.points_um)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            bs = BCSpec(
                name=es.name,
                kind=es.kind,
                x1_um=x1, y1_um=y1,
                x2_um=x2, y2_um=y2,
                phi=es.phi,
                ux=es.ux,
                uy=es.uy,
                outlet_bc=es.outlet_bc,
                rho_target=es.rho_target,
            )
            bs.type_id = es.type_id
            bc_specs.append(bs)

        sparse = build_sparse_maps(self._solid_mask, bc_map)

        return Geometry2D(
            solid_mask=self._solid_mask,
            bc_map=bc_map,
            specs=bc_specs,
            dx_um=self._dx_um,
            origin_um=self._origin_um,
            sparse=sparse,
        )

    def _on_start(self):
        if self._solid_mask is None:
            QMessageBox.warning(self, "Warning", "Load geometry first.")
            return

        try:
            s = self._params.simulation_dict()

            if self._saved_f is not None and self._current_sim is not None:
                # Resume from saved state
                log.info("Resuming simulation from step %d", self._saved_step)
                self._worker = SimWorker(
                    self._current_sim,
                    f_resume=self._saved_f,
                    phi_resume=self._saved_phi,
                    start_step=self._saved_step,
                    emit_interval=s["emit_interval"],
                )
            else:
                # Fresh start
                geom = self._build_geometry()
                p = self._params.physics_dict()
                phys = PhysParams(
                    mu_oil=p["continuous"]["mu_mPas"] * 1e-3,
                    mu_aq=p["disperse"]["mu_mPas"] * 1e-3,
                    rho=p["continuous"]["rho_kg_m3"],
                    sigma=p["interface"]["sigma_mNm"] * 1e-3,
                    contact_angle_deg=p["interface"]["contact_angle_deg"],
                )
                sim = TwoPhaseSim(
                    geom, phys,
                    tau_oil=s["tau_oil"],
                    interface_width=s["interface_width"],
                    mobility=s["mobility"],
                )
                self._current_sim = sim
                phi_init = self._phase_view.build_phi_init()

                log.info(
                    "Starting simulation: tau_oil=%.3f, tau_aq=%.3f, W=%d, M=%.3f, emit=%d",
                    sim.units.tau_oil, sim.units.tau_aq,
                    s["interface_width"], s["mobility"], s["emit_interval"],
                )
                log.info("  dt=%.3e s, dx=%.3e m, sigma_lbm=%.3e, kappa=%.3e, beta=%.3e",
                         sim.units.dt, sim.units.dx, sim.units.sigma_lbm,
                         sim.units.kappa, sim.units.beta)
                u_scale = sim.units.dt / sim.units.dx
                log.info("  u_scale=%.4e, phi_wall=%.4f (contact_angle=%.1f°)",
                         u_scale, sim.phi_wall, phys.contact_angle_deg)
                sp = geom.sparse
                for tid, phi_val, ux_lu, uy_lu in sim.inlet_data:
                    n_cells = int((sp.bc_map_fluid == tid).sum())
                    log.info("  inlet type=%d: phi=%.1f ux_lu=%.4e uy_lu=%.4e  (%d cells)",
                             tid, phi_val, ux_lu, uy_lu, n_cells)
                n_out = int(sp.outlet_mask.sum())
                if n_out > 0:
                    # Show upstream direction for first outlet cell to verify detection
                    out_indices = np.where(sp.outlet_mask)[0]
                    first_out = out_indices[0]
                    out_y = sp.fluid_yx[first_out, 0]
                    out_x = sp.fluid_yx[first_out, 1]
                    up_idx = sp.outlet_upstream[first_out]
                    up_y = sp.fluid_yx[up_idx, 0]
                    up_x = sp.fluid_yx[up_idx, 1]
                    dy, dx_ = up_y - out_y, up_x - out_x
                    log.info("  outlet: %d cells, upstream direction: dy=%d dx=%d (first cell [%d,%d]→[%d,%d])",
                             n_out, dy, dx_, out_y, out_x, up_y, up_x)
                    # Count how many outlet cells found upstream vs self-referencing
                    self_ref = int((sp.outlet_upstream[out_indices] == out_indices).sum())
                    if self_ref > 0:
                        log.warning("  outlet: %d cells have NO upstream neighbor (self-ref)", self_ref)
                else:
                    log.info("  outlet: %d cells", n_out)
                log.info("  outlet BC: %s (rho_target=%.4f)",
                         "pressure" if sim.outlet_pressure else "neumann",
                         sim.rho_target)
                log.info("  n_fluid=%d (%.1f%%)", sp.n_fluid,
                         100.0 * sp.n_fluid / (self._solid_mask.size))
                self._worker = SimWorker(
                    sim, phi_init=phi_init, emit_interval=s["emit_interval"],
                )

            self._worker.frame_ready.connect(self._on_frame)
            self._worker.state_saved.connect(self._on_state_saved)
            self._worker.finished.connect(self._on_sim_finished)
            self._worker.start()
            self._sim_view.set_running(True)
            self._set_status("Simulation running...", "#27ae60")

        except Exception as e:
            log.exception("Failed to start simulation")
            QMessageBox.critical(self, "Error", f"Failed to start simulation:\n{e}")
            self._set_status(f"Error: {e}", "#e74c3c")

    def _on_stop(self):
        if self._worker is not None:
            log.info("Stopping simulation")
            self._worker.request_stop()

    def _on_state_saved(self, step, f, phi):
        self._saved_step = step
        self._saved_f = f
        self._saved_phi = phi
        log.info("Simulation state saved at step %d", step)

    def _on_sim_finished(self):
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        self._sim_view.set_running(False)
        self._sim_view.set_has_saved_state(self._saved_f is not None)
        log.info("Simulation stopped")
        self._set_status("Simulation paused — press Resume to continue", "#888888")

    def _on_reset(self):
        self._saved_f = None
        self._saved_phi = None
        self._saved_step = 0
        self._current_sim = None
        self._sim_view.set_has_saved_state(False)
        log.info("Simulation reset")
        self._set_status("Simulation reset", "#888888")

    def _on_frame(self, step, phi, rho, ux, uy, elapsed, mlups):
        self._sim_view.update_frame(step, phi, rho, ux, uy, elapsed, mlups)
        self._set_status(
            f"Step: {step}  |  MLUPS: {mlups:.1f}  |  Elapsed: {elapsed:.1f}s",
            "#27ae60",
        )

    def _build_session_state(self) -> SessionState:
        edges = self._edge_view.get_edges()
        serialized_edges = []
        for e in edges:
            se = {
                "name": e["name"],
                "kind": e["kind"],
                "phi": e.get("phi", 1.0),
                "ux": e.get("ux", 0.0),
                "uy": e.get("uy", 0.0),
                "flow_rate": e.get("flow_rate", 0.0),
            }
            if e.get("normal_flipped", False):
                se["normal_flipped"] = True
            if e.get("contact_angle_deg") is not None:
                se["contact_angle_deg"] = e["contact_angle_deg"]
            outlet_bc = e.get("outlet_bc", "pressure")
            if outlet_bc != "pressure" or e.get("rho_target", 1.0) != 1.0:
                se["outlet_bc"] = outlet_bc
                se["rho_target"] = e.get("rho_target", 1.0)
            serialized_edges.append(se)

        return SessionState(
            dxf_path=self._params.dxf_path,
            dx_um=self._params.dx_um,
            edges=serialized_edges,
            phase_regions=self._phase_view.get_regions(),
            physics=self._params.physics_dict(),
            simulation=self._params.simulation_dict(),
        )

    def _on_save_config(self):
        state = self._build_session_state()
        path = state.save()
        log.info("Config saved: %s", path)
        self._set_status(f"Config saved: {path}", "#27ae60")

    def _on_load_config(self, path: str):
        try:
            state = SessionState.load(path)
            self._params.set_from_state(state)
            log.info("Config loaded: %s", path)
            self._set_status(f"Config loaded: {path}", "#27ae60")

            if state.dxf_path:
                self._on_load_geometry(state.dxf_path, state.dx_um)
                if state.edges:
                    self._edge_view.set_edges_from_state(state.edges)
                if state.phase_regions:
                    self._phase_view.set_regions_from_state(state.phase_regions)

        except Exception as e:
            log.exception("Failed to load config")
            QMessageBox.critical(self, "Error", f"Failed to load config:\n{e}")

    def closeEvent(self, event):
        self._keep_alive.stop()
        if self._worker is not None:
            log.info("Shutting down — requesting worker stop")
            try:
                self._worker.frame_ready.disconnect(self._on_frame)
                self._worker.state_saved.disconnect(self._on_state_saved)
                self._worker.finished.disconnect(self._on_sim_finished)
            except RuntimeError:
                pass
            self._worker.request_stop()
            self._worker.wait()
            self._worker = None
        event.accept()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    configure_pyqtgraph(pg)

    app = QApplication(sys.argv)
    app.setApplicationName("droplesim")
    app.setStyleSheet(GLOBAL_QSS)

    # Install SIGINT handler AFTER QApplication (Qt resets signal handlers)
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
