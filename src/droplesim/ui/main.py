"""QApplication + MainWindow with 5-tab workflow."""

from __future__ import annotations

import logging
import signal
import sys

import dropletui as ui
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from droplesim.solver.geometry2d import (
    BCSpec,
    EdgeSpec,
    Geometry2D,
    assign_bcs,
    assign_edge_bcs,
    build_sparse_maps,
    extract_edges,
    load_contours,
    rasterize_contours,
)
from droplesim.solver.sim import PhysParams, TwoPhaseSim, contact_angle_to_phi_wall
from droplesim.ui.frame_buffer import FrameBuffer, FrameRecord
from droplesim.ui.panels.params_panel import ParamsPanel
from droplesim.ui.state import SessionState
from droplesim.ui.views.edge_view import EdgeView
from droplesim.ui.views.experiment_view import ExperimentView
from droplesim.ui.views.geometry_view import GeometryView
from droplesim.ui.views.phase_view import PhaseView
from droplesim.ui.views.sim_view import SimView
from droplesim.ui.workers.sim_worker import SimWorker

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
        self._saved_state: dict | None = None  # {f, phi, psi, C, A_xx, A_xy, A_yy}
        self._frame_buffer = FrameBuffer(maxlen=1000)
        self._replay_timer = QTimer()
        self._replay_timer.timeout.connect(self._on_replay_tick)
        self._replay_idx = 0
        self._is_replaying = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._params = ParamsPanel()
        self._params.setMinimumWidth(320)
        self._params.load_geometry_requested.connect(self._on_load_geometry)
        self._params.channel_depth_changed.connect(self._on_channel_depth_changed)
        self._params.save_config_requested.connect(self._on_save_config)
        self._params.load_config_requested.connect(self._on_load_config)
        self._params.auto_drho_requested.connect(self._on_auto_drho)

        # Right panel: stage buttons + stacked views
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(ui.Theme.SPACE_2)

        from PySide6.QtWidgets import QSizePolicy

        stage_bar = QWidget()
        stage_row = QHBoxLayout(stage_bar)
        stage_row.setContentsMargins(0, 0, 0, 0)
        stage_row.setSpacing(1)

        self._stage_btns = []
        stage_labels = ["1. Geometry", "2. Phase", "3. BCs", "4. Simulate", "5. Experiment"]
        for i, label in enumerate(stage_labels):
            btn = ui.stage_button(label)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.clicked.connect(lambda checked, idx=i: self._on_stage_clicked(idx))
            stage_row.addWidget(btn, 1)
            self._stage_btns.append(btn)

        right_layout.addWidget(stage_bar)

        self._stack = QStackedWidget()
        right_layout.addWidget(self._stack, stretch=1)

        # -- Body: shared horizontal splitter (params | right panel) --
        body = ui.horizontal_splitter(
            self._params,
            right,
            sizes=(320, 1080),
            stretch=(0, 1),
            collapsible=(False, False),
        )

        # Views — order: Geometry, Phase, BCs, Simulate, Experiment
        self._geometry_view = GeometryView()
        self._geometry_view.solid_mask_changed.connect(self._on_solid_mask_changed)
        self._stack.addWidget(self._geometry_view)

        self._phase_view = PhaseView()
        self._phase_view.phase_changed.connect(self._on_phase_changed)
        self._stack.addWidget(self._phase_view)

        self._edge_view = EdgeView()
        self._edge_view.edges_changed.connect(self._on_edges_changed)
        self._stack.addWidget(self._edge_view)

        self._sim_view = SimView()
        self._sim_view.start_requested.connect(self._on_start)
        self._sim_view.stop_requested.connect(self._on_stop)
        self._sim_view.reset_requested.connect(self._on_reset)
        self._sim_view.timeline_scrubbed.connect(self._on_timeline_scrubbed)
        self._sim_view.play_toggled.connect(self._on_play_toggled)
        self._sim_view.export_requested.connect(self._on_export)
        self._stack.addWidget(self._sim_view)

        self._experiment_view = ExperimentView()
        self._experiment_view.status_changed.connect(self._set_status)
        self._stack.addWidget(self._experiment_view)
        self._experiment_view.set_settings_provider(
            lambda: (self._build_phys_params(), self._params.simulation_dict())
        )

        # Disable stages 2-4 until geometry is loaded; experiment is self-contained
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
        _ = color
        self._status_bar.showMessage(msg)

    def _on_stage_clicked(self, idx: int):
        if not self._stage_btns[idx].isEnabled():
            return
        if idx == 4:
            self._sync_experiment_settings()
        self._stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._stage_btns):
            btn.setChecked(i == idx)
        self._update_stage_styles()

    def _update_stage_styles(self):
        for btn in self._stage_btns:
            ui.apply_button_style(
                btn,
                variant="primary" if btn.isChecked() else "neutral",
                size="stage",
            )

    def _on_load_geometry(self, path: str, dx_um: float):
        log.info("Loading geometry: %s (dx=%.1f µm)", path, dx_um)
        try:
            self._dx_um = dx_um
            polygons_mm, contours_mm, _ = load_contours(path)
            self._polygons_mm = polygons_mm
            solid_mask, origin_um = rasterize_contours(polygons_mm, contours_mm, dx_um)
            self._solid_mask = solid_mask
            self._origin_um = origin_um

            edge_polylines_mm = extract_edges(contours_mm)
            self._edge_polylines_mm = edge_polylines_mm

            self._geometry_view.set_geometry(solid_mask, dx_um, origin_um)
            self._edge_view.set_geometry(solid_mask, dx_um, origin_um, edge_polylines_mm)
            self._phase_view.set_geometry(solid_mask, dx_um, origin_um, edge_polylines_mm)

            # Compute fluid cell indices for sparse view
            n_fluid = int((~solid_mask).sum())
            fluid_yx = np.argwhere(~solid_mask).astype(np.int32)
            self._fluid_yx = fluid_yx
            index_map = np.full(solid_mask.shape, -1, dtype=np.int32)
            index_map[~solid_mask] = np.arange(n_fluid, dtype=np.int32)
            self._sim_view.set_geometry_info(
                dx_um, origin_um, solid_mask, fluid_yx, index_map=index_map,
            )

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

    def _on_solid_mask_changed(self, solid_mask: np.ndarray):
        self._solid_mask = solid_mask.copy()
        self._geometry_view.set_geometry(self._solid_mask, self._dx_um, self._origin_um)
        self._edge_view.set_solid_mask(self._solid_mask)
        if self._edge_polylines_mm is not None:
            self._phase_view.set_geometry(
                self._solid_mask, self._dx_um, self._origin_um, self._edge_polylines_mm
            )
        n_fluid = int((~self._solid_mask).sum())
        fluid_yx = np.argwhere(~self._solid_mask).astype(np.int32)
        self._fluid_yx = fluid_yx
        index_map = np.full(self._solid_mask.shape, -1, dtype=np.int32)
        index_map[~self._solid_mask] = np.arange(n_fluid, dtype=np.int32)
        self._sim_view.set_geometry_info(
            self._dx_um, self._origin_um, self._solid_mask, fluid_yx, index_map=index_map,
        )
        ny, nx = self._solid_mask.shape
        pct = 100.0 * n_fluid / (ny * nx)
        self._set_status(f"{nx}x{ny} grid  |  {n_fluid:,} fluid cells ({pct:.1f}%)")

    def _on_auto_drho(self):
        if self._solid_mask is None:
            return
        fluid = ~self._solid_mask
        ny, nx = fluid.shape
        # Minimum contiguous fluid run across all rows and columns
        d_min = max(ny, nx)
        for row in fluid:
            run = 0
            for v in row:
                if v:
                    run += 1
                    d_min = min(d_min, run)
                else:
                    run = 0
        for col in fluid.T:
            run = 0
            for v in col:
                if v:
                    run += 1
                    d_min = min(d_min, run)
                else:
                    run = 0
        d_min = max(d_min, 1)
        L = max(ny, nx)
        tau_c = self._params.simulation_dict().get("tau_c", 0.55)
        nu = (tau_c - 0.5) / 3.0
        delta_rho = 0.05 * 24.0 * nu * L / (d_min ** 2)
        delta_rho = max(0.001, min(delta_rho, 0.05))
        self._params.set_delta_rho_max(delta_rho)
        log.info("Auto delta_rho_max: D_min=%d, L=%d, nu=%.4f → %.4f",
                 d_min, L, nu, delta_rho)

    def _on_edges_changed(self):
        n_bc = sum(1 for e in self._edge_view.get_edges() if e["kind"] != "wall")
        n_areas = len(self._edge_view.get_areas())
        log.info("BCs updated: %d edge BCs, %d area BCs", n_bc, n_areas)

    def _on_phase_changed(self):
        n = len(self._phase_view.get_regions())
        log.info("Phase regions updated: %d regions", n)

    def _apply_wall_contact_overrides(self, sparse, edges_data: list[dict]):
        overrides = [
            e for e in edges_data
            if e.get("kind") == "wall" and e.get("contact_angle_deg") is not None
        ]
        if not overrides:
            return

        fy = sparse.fluid_yx[:, 0]
        fx = sparse.fluid_yx[:, 1]
        ox, oy = self._origin_um
        cell_pts = np.column_stack((
            ox + (fx + 0.5) * self._dx_um,
            oy + (fy + 0.5) * self._dx_um,
        ))
        phi_wall = np.full(sparse.nbr8_solid.shape, np.nan, dtype=np.float64)
        threshold = 1.5 * self._dx_um

        for edge in overrides:
            pts = np.asarray(edge.get("points_um", []), dtype=np.float64)
            if len(pts) < 2:
                continue
            near_edge = np.zeros(len(cell_pts), dtype=bool)
            for k in range(len(pts) - 1):
                p0 = pts[k]
                p1 = pts[k + 1]
                seg = p1 - p0
                seg_len2 = float(np.dot(seg, seg))
                if seg_len2 < 1e-24:
                    continue
                rel = cell_pts - p0
                t = np.clip((rel @ seg) / seg_len2, 0.0, 1.0)
                closest = p0 + t[:, None] * seg
                dist = np.linalg.norm(cell_pts - closest, axis=1)
                near_edge |= dist <= threshold
            if near_edge.any():
                value = contact_angle_to_phi_wall(edge["contact_angle_deg"])
                phi_wall[:, near_edge] = np.where(
                    sparse.nbr8_solid[:, near_edge],
                    value,
                    phi_wall[:, near_edge],
                )

        sparse.phi_wall_nbr8 = phi_wall

    def _build_geometry(self):
        # 1) Edge-based BCs (click-on-edge)
        edges_data = self._edge_view.get_edges()
        edge_specs = []
        for e in edges_data:
            if e["kind"] != "wall":
                spec = EdgeSpec(
                    name=e["name"],
                    kind=e["kind"],
                    points_um=e["points_um"],
                    phi=e.get("phi", 1.0),
                    pressure_mbar=e.get("pressure_mbar", 0.0),
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
                pressure_mbar=es.pressure_mbar,
                outlet_bc=es.outlet_bc,
                rho_target=es.rho_target,
            )
            bs.type_id = es.type_id
            bc_specs.append(bs)

        # 2) Area-based BCs (drag-rectangle) — applied on top, wins on overlap
        areas_data = self._edge_view.get_areas()
        if areas_data:
            area_bc_specs = []
            for a in areas_data:
                area_bc_specs.append(BCSpec(
                    name=a["name"],
                    kind=a["kind"],
                    x1_um=a["x1_um"],
                    y1_um=a["y1_um"],
                    x2_um=a["x2_um"],
                    y2_um=a["y2_um"],
                    phi=a.get("phi", 1.0),
                    pressure_mbar=a.get("pressure_mbar", 0.0),
                    outlet_bc=a.get("outlet_bc", "pressure"),
                    rho_target=a.get("rho_target", 1.0),
                ))
            # assign_bcs overwrites bc_map in-place for area regions
            bc_map, area_bc_specs = assign_bcs(
                self._solid_mask, area_bc_specs, self._dx_um, self._origin_um,
                bc_map=bc_map,
                inlet_counter=max((s.type_id for s in bc_specs if s.kind == "inlet"),
                                  default=0) + 1,
            )
            bc_specs.extend(area_bc_specs)

        sparse = build_sparse_maps(self._solid_mask, bc_map)
        self._apply_wall_contact_overrides(sparse, edges_data)

        return Geometry2D(
            solid_mask=self._solid_mask,
            bc_map=bc_map,
            specs=bc_specs,
            dx_um=self._dx_um,
            origin_um=self._origin_um,
            sparse=sparse,
        )

    def _build_phys_params(self) -> PhysParams:
        p = self._params.physics_dict()
        surf = p.get("surfactant")
        ve = p.get("viscoelastic")
        return PhysParams(
            mu_c=p["continuous"]["mu_mPas"] * 1e-3,
            mu_d=p["disperse"]["mu_mPas"] * 1e-3,
            rho_c=p["continuous"]["rho_kg_m3"],
            rho_d=p["disperse"]["rho_kg_m3"],
            sigma=p["interface"]["sigma_mNm"] * 1e-3,
            contact_angle_deg=p["interface"]["contact_angle_deg"],
            D_s=surf["D_s"] if surf else None,
            D_bulk=surf["D_bulk"] if surf else None,
            psi_inf=surf.get("Gamma_max", surf["psi_inf"]) if surf else None,
            E0=surf["E0"] if surf else None,
            k_a=surf["k_a"] if surf else None,
            k_d=surf["k_d"] if surf else None,
            C_inlet=surf.get("C_inlet", 0.1) if surf else None,
            sigma_floor=surf.get("sigma_floor") if surf else None,
            surfactant_initial_coverage=surf.get("initial_coverage", 0.0)
            if surf else 0.0,
            lambda_p=ve["lambda_p"] if ve else None,
            mu_p=ve["mu_p"] if ve else None,
            kappa_ve=ve.get("kappa_ve") if ve else None,
        )

    def _sync_experiment_settings(self):
        self._experiment_view.set_global_settings(
            self._build_phys_params(),
            self._params.simulation_dict(),
        )

    def _on_start(self):
        if self._solid_mask is None:
            QMessageBox.warning(self, "Warning", "Load geometry first.")
            return

        try:
            s = self._params.simulation_dict()

            maxlen = s.get("history_frames", 1000)
            if self._frame_buffer.maxlen != maxlen:
                self._frame_buffer = FrameBuffer(maxlen=maxlen)

            if self._saved_state is not None and self._current_sim is not None:
                # Resume from saved state
                log.info("Resuming simulation from step %d", self._saved_step)
                st = self._saved_state
                self._worker = SimWorker(
                    self._current_sim,
                    f_resume=st["f"],
                    phi_resume=st["phi"],
                    psi_resume=st.get("psi"),
                    C_resume=st.get("C"),
                    A_xx_resume=st.get("A_xx"),
                    A_xy_resume=st.get("A_xy"),
                    A_yy_resume=st.get("A_yy"),
                    start_step=self._saved_step,
                    emit_interval=s["emit_interval"],
                )
            else:
                # Fresh start
                geom = self._build_geometry()
                phys = self._build_phys_params()
                sim = TwoPhaseSim(
                    geom, phys,
                    tau_c=s["tau_c"],
                    interface_width=s["interface_width"],
                    mobility=s["mobility"],
                    delta_rho_max=s.get("delta_rho_max", 0.005),
                )
                self._current_sim = sim
                phi_init = self._phase_view.build_phi_init()

                log.info(
                    "Starting simulation: tau_c=%.3f, tau_d=%.3f, W=%d, M=%.3f, emit=%d",
                    sim.units.tau_c, sim.units.tau_d,
                    s["interface_width"], s["mobility"], s["emit_interval"],
                )
                log.info("  dt=%.3e s, dx=%.3e m, sigma_lbm=%.3e, kappa=%.3e, beta=%.3e",
                         sim.units.dt, sim.units.dx, sim.units.sigma_lbm,
                         sim.units.kappa, sim.units.beta)
                if sim.units.pressure_scale < 1.0:
                    log.warning(
                        "  pressure drive capped for LBM stability: scale=%.4f "
                        "(effective pressure = requested × scale)",
                        sim.units.pressure_scale,
                    )
                u_scale = sim.units.dt / sim.units.dx
                log.info("  u_scale=%.4e, phi_wall=%.4f (contact_angle=%.1f°)",
                         u_scale, sim.phi_wall, phys.contact_angle_deg)
                sp = geom.sparse
                for tid, phi_val, rho_in in sim.inlet_data:
                    n_cells = int((sp.bc_map_fluid == tid).sum())
                    log.info("  inlet type=%d: phi=%.1f rho_in=%.6f  (%d cells)",
                             tid, phi_val, rho_in, n_cells)
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
                if sim.surfactant_enabled:
                    log.info("  surfactant: D_s_lu=%.3e D_bulk_lu=%.3e psi_inf_lu=%.3e E0=%.3f",
                             sim.units.D_s_lu, sim.units.D_bulk_lu,
                             sim.units.psi_inf_lu, sim.units.E0)
                    log.info("  surfactant: k_a_lu=%.3e k_d_lu=%.3e C_inlet_lu=%.3e",
                             sim.units.k_a_lu, sim.units.k_d_lu, sim.units.C_inlet_lu)
                    log.info("  surfactant: sigma_floor_lbm=%.3e initial_coverage=%.3f",
                             sim.units.sigma_floor_lbm,
                             sim.units.surfactant_initial_coverage)
                if sim.viscoelastic_enabled:
                    log.info("  viscoelastic: lambda_p_lu=%.2f mu_p_lu=%.3e beta_visc=%.4f",
                             sim.units.lambda_p_lu, sim.units.mu_p_lu, sim.units.beta_visc)
                    log.info("  viscoelastic: tau_d_solvent=%.4f kappa_ve_lu=%.3e",
                             sim.units.tau_d_solvent, sim.units.kappa_ve_lu)
                self._worker = SimWorker(
                    sim, phi_init=phi_init, emit_interval=s["emit_interval"],
                )

            self._worker.frame_ready.connect(self._on_frame)
            self._worker.state_saved.connect(self._on_state_saved)
            self._worker.diverged.connect(self._on_diverged)
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

    def _on_state_saved(self, step, state_dict):
        self._saved_step = step
        self._saved_state = state_dict
        log.info("Simulation state saved at step %d", step)

    def _on_sim_finished(self):
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        self._sim_view.set_running(False)
        self._sim_view.set_has_saved_state(self._saved_state is not None)
        self._sim_view.set_timeline_state(
            len(self._frame_buffer), max(0, len(self._frame_buffer) - 1), is_live=False,
        )
        log.info("Simulation stopped")
        self._set_status("Simulation paused — press Resume to continue", "#888888")

    def _on_reset(self):
        self._saved_state = None
        self._saved_step = 0
        self._current_sim = None
        self._sim_view.set_has_saved_state(False)
        self._replay_timer.stop()
        self._is_replaying = False
        self._replay_idx = 0
        self._frame_buffer.clear()
        self._sim_view.set_timeline_state(0, 0, is_live=True)
        log.info("Simulation reset")
        self._set_status("Simulation reset", "#888888")

    def _on_diverged(self, step: int, field: str):
        log.error("Simulation diverged at step %d (field: %s)", step, field)
        self._saved_state = None
        self._saved_step = 0
        self._current_sim = None
        self._sim_view.set_has_saved_state(False)
        self._set_status(
            f"Simulation diverged at step {step} (NaN/Inf in {field}) — try lower flow rates or higher sigma",
        )
        QMessageBox.warning(
            self, "Simulation Diverged",
            f"Numerical instability detected at step {step}.\n"
            f"Field '{field}' contains NaN or Inf values.\n\n"
            "Try:\n"
            "  - Lowering inlet flow rates\n"
            "  - Increasing surface tension (sigma)\n"
            "  - Reducing tau_c closer to 0.55\n"
            "  - Increasing emit interval to check earlier",
        )

    def _on_frame(self, step, phi, rho, ux, uy, elapsed, mlups, extra=None):
        rec = FrameRecord(
            step,
            np.array(phi, dtype=np.float32, copy=True),
            np.array(rho, dtype=np.float32, copy=True),
            np.array(ux, dtype=np.float32, copy=True),
            np.array(uy, dtype=np.float32, copy=True),
            elapsed,
            mlups,
            {k: np.array(v, dtype=np.float32, copy=True) for k, v in extra.items()}
            if extra
            else None,
        )
        self._frame_buffer.append(rec)
        self._sim_view.update_frame(step, phi, rho, ux, uy, elapsed, mlups, extra=extra)
        self._sim_view.set_timeline_state(
            len(self._frame_buffer), len(self._frame_buffer) - 1, is_live=True,
        )
        self._set_status(
            f"Step: {step}  |  MLUPS: {mlups:.1f}  |  Elapsed: {elapsed:.1f}s",
            "#27ae60",
        )

    def _on_timeline_scrubbed(self, idx: int):
        if idx < 0 or idx >= len(self._frame_buffer):
            return
        rec = self._frame_buffer[idx]
        self._sim_view.update_frame(
            rec.step, rec.phi, rec.rho, rec.ux, rec.uy,
            rec.elapsed, rec.mlups, extra=rec.extra,
        )
        self._sim_view.set_timeline_state(
            len(self._frame_buffer), idx, is_live=False,
        )
        self._replay_idx = idx

    def _on_play_toggled(self, playing: bool):
        if playing:
            speed = self._sim_view.playback_speed
            interval_ms = max(16, int(1000 / (speed * 30)))
            self._replay_timer.start(interval_ms)
            self._is_replaying = True
        else:
            self._replay_timer.stop()
            self._is_replaying = False

    def _on_replay_tick(self):
        if self._replay_idx < len(self._frame_buffer) - 1:
            self._replay_idx += 1
            self._on_timeline_scrubbed(self._replay_idx)
        else:
            self._replay_timer.stop()
            self._is_replaying = False
            self._sim_view.set_play_state(False)

    def _on_export(self):
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Export HDF5", "", "HDF5 (*.h5)")
        if path:
            self._frame_buffer.export_hdf5(path)
            log.info("Exported %d frames to %s", len(self._frame_buffer), path)
            self._set_status(f"Exported {len(self._frame_buffer)} frames to {path}")

    def _build_session_state(self) -> SessionState:
        edges = self._edge_view.get_edges()
        serialized_edges = []
        for e in edges:
            se = {
                "name": e["name"],
                "kind": e["kind"],
                "phi": e.get("phi", 1.0),
                "pressure_mbar": e.get("pressure_mbar", 0.0),
            }
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
            bc_areas=self._edge_view.get_areas(),
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
                if state.bc_areas:
                    self._edge_view.set_areas_from_state(state.bc_areas)
                if state.phase_regions:
                    self._phase_view.set_regions_from_state(state.phase_regions)

        except Exception as e:
            log.exception("Failed to load config")
            QMessageBox.critical(self, "Error", f"Failed to load config:\n{e}")

    def closeEvent(self, event):
        self._keep_alive.stop()
        self._replay_timer.stop()
        self._experiment_view.shutdown()
        if self._worker is not None:
            log.info("Shutting down — requesting worker stop")
            try:
                self._worker.frame_ready.disconnect(self._on_frame)
                self._worker.state_saved.disconnect(self._on_state_saved)
                self._worker.diverged.disconnect(self._on_diverged)
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
    ui.configure_pyqtgraph(pg)

    app = ui.create_app("droplesim", sys.argv)
    ui.apply_app_theme(app)

    # Install SIGINT handler AFTER QApplication (Qt resets signal handlers)
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
