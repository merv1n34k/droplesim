"""Tab 5: click-to-inject droplet corridor experiment."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import dropletui as ui
import numpy as np
import pyqtgraph as pg
from dropletui.theme import text_qss
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi

from droplesim.solver.geometry2d import (
    BC_OUTLET,
    BCSpec,
    Geometry2D,
    build_sparse_maps,
)
from droplesim.solver.sim import PhysParams, TwoPhaseSim

log = logging.getLogger(__name__)

# ── Colormaps (same as SimView) ─────────────────────────────────────────────

_PHI_CMAP = pg.ColorMap(
    pos=[0.0, 1.0],
    color=[(52, 152, 219), (231, 76, 60)],
)
_VEL_CMAP = pg.ColorMap(
    pos=[0.0, 0.25, 0.5, 0.75, 1.0],
    color=[
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
)
_PRS_CMAP = pg.ColorMap(
    pos=[0.0, 0.5, 1.0],
    color=[(59, 76, 192), (221, 221, 221), (180, 4, 38)],
)

_VF_COLORS = [
    (52, 152, 219),   # blue
    (46, 204, 113),   # green
    (231, 76, 60),    # red
    (155, 89, 182),   # purple
]
_VF_FRACTIONS = [0.25, 0.45, 0.65, 0.85]
_VF_HISTORY_LEN = 500
_DROPLET_METRIC_INTERVAL = 50


@dataclass(frozen=True)
class ExperimentSettings:
    kind: str
    droplet_diameter_um: float
    corridor_width_um: float
    dx_um: float
    pressure_mbar: float
    steps_per_tick: int


class ExperimentView(QWidget):
    status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Controls row ────────────────────────────────────────────────
        controls = QGridLayout()
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)

        controls.addWidget(QLabel("Corridor:"), 0, 0)
        self._corridor = ui.combo_box(["Straight", "L-turn", "Zig-zag (generator.dxf)"])
        self._corridor.currentIndexChanged.connect(self._reset_simulation)
        controls.addWidget(self._corridor, 0, 1)

        controls.addWidget(QLabel("Droplet diameter [µm]:"), 0, 2)
        self._diameter = ui.double_box(
            minimum=2.0, maximum=300.0, value=30.0, step=2.5, decimals=1,
        )
        controls.addWidget(self._diameter, 0, 3)

        controls.addWidget(QLabel("Width [µm]:"), 0, 4)
        self._width = ui.double_box(
            minimum=10.0, maximum=300.0, value=45.0, step=5.0, decimals=1,
        )
        self._width.valueChanged.connect(self._reset_simulation)
        controls.addWidget(self._width, 0, 5)

        controls.addWidget(QLabel("Resolution [µm]:"), 1, 0)
        self._dx_um = ui.double_box(
            minimum=1.0, maximum=20.0, value=2.5, step=0.5, decimals=1,
        )
        self._dx_um.valueChanged.connect(self._reset_simulation)
        controls.addWidget(self._dx_um, 1, 1)

        controls.addWidget(QLabel("Pressure [mbar]:"), 1, 2)
        self._pressure = ui.double_box(
            minimum=0.0, maximum=5000.0, value=1000.0, step=50.0, decimals=0,
        )
        self._pressure.valueChanged.connect(self._reset_simulation)
        controls.addWidget(self._pressure, 1, 3)

        controls.addWidget(QLabel("Steps/tick:"), 1, 4)
        self._steps_per_tick = ui.int_box(minimum=1, maximum=200, value=10)
        controls.addWidget(self._steps_per_tick, 1, 5)

        self._start_btn = ui.button("Start", variant="success")
        self._start_btn.clicked.connect(self._toggle_running)
        controls.addWidget(self._start_btn, 0, 6)

        self._reset_btn = ui.button("Reset", variant="primary")
        self._reset_btn.clicked.connect(self._reset_simulation)
        controls.addWidget(self._reset_btn, 1, 6)
        controls.setColumnStretch(7, 1)
        layout.addLayout(controls)

        # ── Field toggles + mode button ──────────────────────────────────
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(8)

        self._chk_phase = QCheckBox("Phase")
        self._chk_phase.setChecked(True)
        self._chk_phase.toggled.connect(self._on_field_toggle)
        toggle_row.addWidget(self._chk_phase)

        self._chk_velocity = QCheckBox("Velocity")
        self._chk_velocity.setChecked(False)
        self._chk_velocity.toggled.connect(self._on_field_toggle)
        toggle_row.addWidget(self._chk_velocity)

        self._chk_pressure = QCheckBox("Pressure")
        self._chk_pressure.setChecked(False)
        self._chk_pressure.toggled.connect(self._on_field_toggle)
        toggle_row.addWidget(self._chk_pressure)

        toggle_row.addStretch()

        self._mode_btn = QPushButton("Mode: Inject")
        self._mode_btn.setCheckable(True)
        self._mode_btn.clicked.connect(self._on_mode_toggled)
        toggle_row.addWidget(self._mode_btn)

        self._clear_probes_btn = ui.button("Clear Probes", size="inline")
        self._clear_probes_btn.clicked.connect(self._on_clear_probes)
        toggle_row.addWidget(self._clear_probes_btn)

        layout.addLayout(toggle_row)

        # ── Field plots ──────────────────────────────────────────────────
        from droplesim.ui.views.field_plot import FieldPlot

        self._phi_field = FieldPlot("phi", _PHI_CMAP)
        self._vel_field = FieldPlot("|u|", _VEL_CMAP)
        self._prs_field = FieldPlot("\u0394\u03c1", _PRS_CMAP)

        self._fields = [self._phi_field, self._vel_field, self._prs_field]
        self._chk_for_field = [self._chk_phase, self._chk_velocity, self._chk_pressure]

        self._plots_container = QWidget()
        self._grid = QGridLayout(self._plots_container)
        self._grid.setSpacing(4)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._relayout_grid()
        layout.addWidget(self._plots_container, stretch=1)

        # ── VF cross-section rolling plot ────────────────────────────────
        self._vf_plot = pg.PlotWidget()
        self._vf_plot.setBackground(ui.Theme.BG_DARK)
        self._vf_plot.setFixedHeight(100)
        self._vf_plot.setLabel("left", "VF")
        self._vf_plot.setLabel("bottom", "tick")
        self._vf_plot.setYRange(0.0, 1.0)
        self._vf_plot.addLegend(offset=(60, 10))
        self._vf_curves: list[pg.PlotDataItem] = []
        self._vf_deques: list[deque] = []
        for i, frac in enumerate(_VF_FRACTIONS):
            color = _VF_COLORS[i]
            curve = self._vf_plot.plot(
                pen=pg.mkPen(color=color, width=2),
                name=f"{int(frac * 100)}%",
            )
            self._vf_curves.append(curve)
            self._vf_deques.append(deque(maxlen=_VF_HISTORY_LEN))
        layout.addWidget(self._vf_plot)

        # ── Droplet metrics label ────────────────────────────────────────
        self._metrics_label = QLabel("Droplets: --")
        mono = QFont()
        ui.configure_monospace_font(mono, 11)
        self._metrics_label.setFont(mono)
        self._metrics_label.setStyleSheet(text_qss("muted", padding="2px"))
        layout.addWidget(self._metrics_label)

        # ── Status line ──────────────────────────────────────────────────
        self._status = ui.status_label("Experiment ready", kind="muted", small=True)
        mono2 = QFont()
        ui.configure_monospace_font(mono2, 11)
        self._status.setFont(mono2)
        self._status.setStyleSheet(text_qss("muted", padding="4px"))
        layout.addWidget(self._status)

        # ── State ────────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._settings_provider: Callable[[], tuple[PhysParams, dict]] | None = None
        self._phys: PhysParams | None = None
        self._sim_settings: dict | None = None
        self._sim: TwoPhaseSim | None = None
        self._state: tuple | None = None
        self._step = 0
        self._tick = 0
        self._t0 = time.perf_counter()
        self._inject_mode = True
        self._first_frame = True
        self._vf_masks: list[np.ndarray] | None = None
        self._prev_centroids: dict[int, tuple[float, float]] | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def set_global_settings(self, phys: PhysParams, sim_settings: dict):
        self._phys = phys
        self._sim_settings = dict(sim_settings)
        if self._sim is not None:
            self._reset_simulation()

    def set_settings_provider(self, provider: Callable[[], tuple[PhysParams, dict]]):
        self._settings_provider = provider
        phys, sim_settings = provider()
        self.set_global_settings(phys, sim_settings)

    def shutdown(self):
        self._timer.stop()

    # ── Settings ─────────────────────────────────────────────────────────

    def _settings(self) -> ExperimentSettings:
        return ExperimentSettings(
            kind=self._corridor.currentText(),
            droplet_diameter_um=self._diameter.value(),
            corridor_width_um=self._width.value(),
            dx_um=self._dx_um.value(),
            pressure_mbar=self._pressure.value(),
            steps_per_tick=self._steps_per_tick.value(),
        )

    # ── Controls ─────────────────────────────────────────────────────────

    def _toggle_running(self):
        if self._timer.isActive():
            self._timer.stop()
            self._start_btn.setText("Start")
            self._set_status("Experiment paused")
            return
        if self._ensure_simulation():
            self._timer.start(16)
            self._start_btn.setText("Stop")
            self._set_status("Experiment running")

    def _on_mode_toggled(self, checked: bool):
        self._inject_mode = not checked
        self._mode_btn.setText("Mode: Probe" if checked else "Mode: Inject")

    def _on_field_toggle(self, _checked: bool):
        self._relayout_grid()

    def _on_clear_probes(self):
        for field in self._fields:
            field.clear_probes()

    def _relayout_grid(self):
        for f in self._fields:
            self._grid.removeWidget(f)
            f.setParent(None)
            f.setVisible(False)

        for i in range(self._grid.rowCount()):
            self._grid.setRowStretch(i, 0)
        for i in range(self._grid.columnCount()):
            self._grid.setColumnStretch(i, 0)

        visible = [
            f for f, chk in zip(self._fields, self._chk_for_field)
            if chk.isChecked()
        ]
        n = len(visible)
        if n == 0:
            return

        cols = n if n <= 2 else 2
        rows = (n + cols - 1) // cols

        for i, f in enumerate(visible):
            f.setVisible(True)
            self._grid.addWidget(f, i // cols, i % cols)

        for r in range(rows):
            self._grid.setRowStretch(r, 1)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)

    # ── Simulation lifecycle ─────────────────────────────────────────────

    def _reset_simulation(self, *_args):
        self._timer.stop()
        self._start_btn.setText("Start")
        self._sim = None
        self._state = None
        self._step = 0
        self._tick = 0
        self._first_frame = True
        self._vf_masks = None
        self._prev_centroids = None
        for d in self._vf_deques:
            d.clear()
        for c in self._vf_curves:
            c.setData([], [])
        for f in self._fields:
            f.clear_geometry()
        self._metrics_label.setText("Droplets: --")
        self._ensure_simulation()
        self._set_status("Experiment reset")

    def _ensure_simulation(self) -> bool:
        if self._sim is not None:
            return True
        if self._settings_provider is not None:
            self._phys, self._sim_settings = self._settings_provider()
        if self._phys is None or self._sim_settings is None:
            self._set_status("Experiment needs settings")
            return False

        settings = self._settings()
        geom = self._build_geometry(settings)
        s = self._sim_settings
        self._sim = TwoPhaseSim(
            geom,
            self._phys,
            tau_c=s["tau_c"],
            interface_width=s["interface_width"],
            mobility=s["mobility"],
            delta_rho_max=s.get("delta_rho_max", 0.005),
        )
        phi_init = np.ones(geom.shape, dtype=np.float64)
        self._state = self._sim.init_state(phi_init=phi_init)
        self._step = 0
        self._tick = 0
        self._t0 = time.perf_counter()
        self._first_frame = True

        ny, nx = geom.shape
        fy = geom.sparse.fluid_yx[:, 0]
        fx = geom.sparse.fluid_yx[:, 1]
        for field in self._fields:
            field.set_geometry(
                ny, nx, fy, fx,
                geom.dx_um, geom.origin_um,
                index_map=geom.sparse.index_map,
            )

        self._setup_vf_masks(geom, settings)
        self._update_display()
        return True

    # ── Advance loop ─────────────────────────────────────────────────────

    def _advance(self):
        if self._sim is None or self._state is None:
            return

        surf = self._sim.surfactant_enabled
        ve = self._sim.viscoelastic_enabled
        for _ in range(self._settings().steps_per_tick):
            if surf and ve:
                self._state = self._sim.step(*self._state)
            elif surf:
                f, phi, psi, C = self._state
                self._state = self._sim.step(f, phi, psi, C)
            elif ve:
                f, phi, A_xx, A_xy, A_yy = self._state
                self._state = self._sim.step(f, phi, A_xx=A_xx, A_xy=A_xy, A_yy=A_yy)
            else:
                f, phi = self._state
                self._state = self._sim.step(f, phi)
            self._step += 1
        self._tick += 1

        self._update_display()

        # VF cross-section probes
        self._update_vf_probes()

        # Droplet metrics (every N steps)
        if self._step % _DROPLET_METRIC_INTERVAL < self._settings().steps_per_tick:
            self._update_droplet_metrics()

        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        mlups = self._step * self._sim.n_fluid / elapsed / 1e6
        phi_np = np.asarray(self._state[1])
        aq_vf = float(1.0 - phi_np.mean())
        u_max = 0.0
        f = self._state[0]
        rho, ux, uy = self._sim.macroscopic(f)
        u_max = float(np.sqrt(np.asarray(ux) ** 2 + np.asarray(uy) ** 2).max())
        self._set_status(
            f"Step: {self._step:>8d}   MLUPS: {mlups:>6.1f}"
            f"   VF_aq: {aq_vf:.3f}   u_max: {u_max:.4f}"
        )

    def _update_display(self):
        if self._sim is None or self._state is None:
            return

        f = self._state[0]
        phi_np = np.asarray(self._state[1])
        rho, ux, uy = self._sim.macroscopic(f)
        rho_np = np.asarray(rho)
        ux_np = np.asarray(ux)
        uy_np = np.asarray(uy)

        if self._chk_phase.isChecked():
            self._phi_field.update(phi_np, vmin=0.0, vmax=1.0, fmt=".2f")

        if self._chk_velocity.isChecked():
            vel = np.sqrt(ux_np ** 2 + uy_np ** 2)
            vmax = float(vel.max()) or 1.0
            self._vel_field.update(vel, vmin=0.0, vmax=vmax)

        if self._chk_pressure.isChecked():
            dev = rho_np - 1.0
            amp = max(abs(float(dev.min())), abs(float(dev.max())), 1e-8)
            self._prs_field.update(dev, vmin=-amp, vmax=amp)

        if self._first_frame:
            for field in self._fields:
                if field.isVisible():
                    field.auto_range()
            self._first_frame = False

    # ── VF cross-section probes ──────────────────────────────────────────

    def _setup_vf_masks(self, geom: Geometry2D, settings: ExperimentSettings):
        points = self._corridor_points(settings)
        if points is None or len(points) < 2:
            self._vf_masks = None
            return

        path_lengths = [0.0]
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            path_lengths.append(path_lengths[-1] + np.hypot(dx, dy))
        total = path_lengths[-1]
        if total < 1e-6:
            self._vf_masks = None
            return

        fy = geom.sparse.fluid_yx[:, 0]
        fx = geom.sparse.fluid_yx[:, 1]
        ox, oy = geom.origin_um
        px = ox + (fx + 0.5) * geom.dx_um
        py = oy + (fy + 0.5) * geom.dx_um

        masks = []
        for frac in _VF_FRACTIONS:
            target = frac * total
            seg = 0
            for j in range(1, len(path_lengths)):
                if path_lengths[j] >= target:
                    seg = j - 1
                    break
            x0, y0 = points[seg]
            x1, y1 = points[seg + 1]
            seg_len = path_lengths[seg + 1] - path_lengths[seg]
            if seg_len < 1e-6:
                t = 0.5
            else:
                t = (target - path_lengths[seg]) / seg_len
            cx = x0 + t * (x1 - x0)
            cy = y0 + t * (y1 - y0)

            if abs(x1 - x0) >= abs(y1 - y0):
                mask = np.abs(px - cx) < geom.dx_um
            else:
                mask = np.abs(py - cy) < geom.dx_um
            masks.append(mask)
        self._vf_masks = masks

    def _corridor_points(self, settings: ExperimentSettings):
        if settings.kind.startswith("Straight"):
            size_um = (620.0, max(160.0, settings.corridor_width_um * 3.0))
            y = size_um[1] * 0.5
            return [(50.0, y), (size_um[0] - 50.0, y)]
        elif settings.kind.startswith("L-turn"):
            return [(55.0, 95.0), (315.0, 95.0), (315.0, 365.0)]
        else:
            return [
                (70.0, 70.0), (70.0, 210.0), (145.0, 210.0),
                (145.0, 140.0), (220.0, 140.0), (220.0, 210.0),
                (295.0, 210.0), (295.0, 140.0), (370.0, 140.0),
                (370.0, 210.0), (445.0, 210.0), (445.0, 140.0),
                (520.0, 140.0), (520.0, 210.0), (595.0, 210.0),
                (595.0, 140.0), (670.0, 140.0), (670.0, 210.0),
                (745.0, 210.0), (745.0, 140.0), (850.0, 140.0),
            ]

    def _update_vf_probes(self):
        if self._vf_masks is None or self._state is None:
            return
        phi_np = np.asarray(self._state[1])
        for i, mask in enumerate(self._vf_masks):
            if mask.sum() == 0:
                continue
            vf = float(1.0 - phi_np[mask].mean())
            self._vf_deques[i].append(vf)
            xs = list(range(len(self._vf_deques[i])))
            self._vf_curves[i].setData(xs, list(self._vf_deques[i]))

    # ── Droplet metrics ──────────────────────────────────────────────────

    def _update_droplet_metrics(self):
        if self._sim is None or self._state is None:
            return

        geom = self._sim.geom
        ny, nx = geom.shape
        phi_np = np.asarray(self._state[1])

        phi_dense = np.ones((ny, nx), dtype=np.float64)
        fy = geom.sparse.fluid_yx[:, 0]
        fx = geom.sparse.fluid_yx[:, 1]
        phi_dense[fy, fx] = phi_np

        aq_mask = np.zeros((ny, nx), dtype=bool)
        aq_mask[fy, fx] = phi_np < 0.5

        labels, n_drops = ndi.label(aq_mask)
        if n_drops == 0:
            self._metrics_label.setText("Droplets: 0")
            self._prev_centroids = None
            return

        sizes = ndi.sum(aq_mask, labels, index=np.arange(1, n_drops + 1))
        centroids = ndi.center_of_mass(aq_mask, labels, index=np.arange(1, n_drops + 1))

        min_cells = 5
        dx = geom.dx_um
        cell_area = dx * dx

        parts = []
        current_centroids = {}
        for idx in range(n_drops):
            if sizes[idx] < min_cells:
                continue
            area_um2 = sizes[idx] * cell_area
            cy, cx = centroids[idx]
            current_centroids[idx] = (cy, cx)
            parts.append(f"{area_um2:.0f}µm²")

        n_significant = len(parts)
        areas_str = ", ".join(parts[:5])
        if len(parts) > 5:
            areas_str += f" (+{len(parts) - 5})"

        vel_str = ""
        if self._prev_centroids is not None and len(current_centroids) > 0:
            speeds = []
            for idx, (cy, cx) in current_centroids.items():
                if idx in self._prev_centroids:
                    py, px = self._prev_centroids[idx]
                    dist = np.hypot(cx - px, cy - py) * dx
                    dt_steps = _DROPLET_METRIC_INTERVAL
                    speed_um_per_step = dist / max(dt_steps, 1)
                    speeds.append(speed_um_per_step)
            if speeds:
                vel_str = f"  v_avg: {np.mean(speeds):.3f} µm/step"

        self._prev_centroids = current_centroids
        self._metrics_label.setText(
            f"Droplets: {n_significant}  [{areas_str}]{vel_str}"
        )

    # ── Droplet injection ────────────────────────────────────────────────

    def _add_droplet(self, x_um: float, y_um: float):
        if not self._ensure_simulation() or self._sim is None or self._state is None:
            return

        geom = self._sim.geom
        settings = self._settings()
        radius = 0.5 * settings.droplet_diameter_um
        x_um, y_um = _clamp_to_droplet_center(
            geom.solid_mask,
            geom.bc_map,
            geom.dx_um,
            geom.origin_um,
            x_um,
            y_um,
            radius,
        )
        ox, oy = geom.origin_um
        col = int((x_um - ox) / geom.dx_um)
        row = int((y_um - oy) / geom.dx_um)
        if not (0 <= row < geom.shape[0] and 0 <= col < geom.shape[1]):
            return
        if geom.solid_mask[row, col] or geom.bc_map[row, col] != 0:
            return

        f, phi, *rest = self._state
        phi_np = np.asarray(phi, dtype=np.float64).copy()
        fy = geom.sparse.fluid_yx[:, 0]
        fx = geom.sparse.fluid_yx[:, 1]
        px = ox + (fx + 0.5) * geom.dx_um
        py = oy + (fy + 0.5) * geom.dx_um
        interface = max(geom.dx_um, 0.5 * self._sim.units.interface_width * geom.dx_um)
        dist = np.hypot(px - x_um, py - y_um)
        aqueous = 0.5 * (1.0 - np.tanh((dist - radius) / interface))
        phi_np = np.minimum(phi_np, 1.0 - aqueous)
        phi_np = np.where(np.asarray(geom.sparse.bc_map_fluid) == 1, 1.0, phi_np)
        self._state = (f, phi_np, *rest)
        self._update_display()

    # ── Click dispatch ───────────────────────────────────────────────────

    def _on_field_click(self, event):
        if event.button() not in (Qt.LeftButton, Qt.MouseButton.LeftButton):
            return
        if not self._inject_mode:
            return
        for field in self._fields:
            vb = field.plot_widget.plotItem.vb
            if vb.sceneBoundingRect().contains(event.scenePos()):
                pos = vb.mapSceneToView(event.scenePos())
                self._add_droplet(float(pos.x()), float(pos.y()))
                return

    def _connect_click_handlers(self):
        for field in self._fields:
            field.plot_widget.scene().sigMouseClicked.connect(self._on_field_click)

    # ── Status ───────────────────────────────────────────────────────────

    def _set_status(self, text: str):
        self._status.setText(text)
        self.status_changed.emit(text)

    # ── Geometry builders ────────────────────────────────────────────────

    def _build_geometry(self, settings: ExperimentSettings) -> Geometry2D:
        if settings.kind.startswith("Straight"):
            size_um = (620.0, max(160.0, settings.corridor_width_um * 3.0))
            y = size_um[1] * 0.5
            points = ((50.0, y), (size_um[0] - 50.0, y))
        elif settings.kind.startswith("L-turn"):
            size_um = (420.0, 420.0)
            points = ((55.0, 95.0), (315.0, 95.0), (315.0, 365.0))
        else:
            size_um = (920.0, 320.0)
            points = (
                (70.0, 70.0), (70.0, 210.0), (145.0, 210.0),
                (145.0, 140.0), (220.0, 140.0), (220.0, 210.0),
                (295.0, 210.0), (295.0, 140.0), (370.0, 140.0),
                (370.0, 210.0), (445.0, 210.0), (445.0, 140.0),
                (520.0, 140.0), (520.0, 210.0), (595.0, 210.0),
                (595.0, 140.0), (670.0, 140.0), (670.0, 210.0),
                (745.0, 210.0), (745.0, 140.0), (850.0, 140.0),
            )

        solid_mask, bc_map, specs = _sharp_corridor(
            size_um,
            points,
            settings.corridor_width_um,
            settings.dx_um,
            settings.pressure_mbar,
        )
        geom = Geometry2D(
            solid_mask=solid_mask,
            bc_map=bc_map,
            specs=specs,
            dx_um=settings.dx_um,
            origin_um=(0.0, 0.0),
            sparse=build_sparse_maps(solid_mask, bc_map),
        )
        self._connect_click_handlers()
        return geom


def _sharp_corridor(
    size_um: tuple[float, float],
    points_um: tuple[tuple[float, float], ...],
    width_um: float,
    dx_um: float,
    pressure_mbar: float,
) -> tuple[np.ndarray, np.ndarray, list[BCSpec]]:
    nx = int(np.ceil(size_um[0] / dx_um))
    ny = int(np.ceil(size_um[1] / dx_um))
    xs = (np.arange(nx) + 0.5) * dx_um
    ys = (np.arange(ny) + 0.5) * dx_um
    xx, yy = np.meshgrid(xs, ys)

    half = width_um * 0.5
    fluid = np.zeros((ny, nx), dtype=bool)
    for p0, p1 in zip(points_um[:-1], points_um[1:]):
        x0, y0 = p0
        x1, y1 = p1
        if abs(y1 - y0) <= abs(x1 - x0):
            fluid |= (
                (xx >= min(x0, x1) - half) & (xx <= max(x0, x1) + half)
                & (yy >= y0 - half) & (yy <= y0 + half)
            )
        else:
            fluid |= (
                (yy >= min(y0, y1) - half) & (yy <= max(y0, y1) + half)
                & (xx >= x0 - half) & (xx <= x0 + half)
            )

    solid_mask = ~fluid
    bc_map = np.zeros((ny, nx), dtype=np.uint8)
    inlet = _end_cap_mask(xx, yy, points_um[0], points_um[1], width_um)
    outlet = _end_cap_mask(xx, yy, points_um[-1], points_um[-2], width_um)
    bc_map[inlet & fluid] = 1
    bc_map[outlet & fluid] = BC_OUTLET
    specs = _bc_specs(pressure_mbar)
    return solid_mask, bc_map, specs


def _end_cap_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    end: tuple[float, float],
    neighbor: tuple[float, float],
    width_um: float,
) -> np.ndarray:
    ex, ey = end
    half = width_um * 0.5
    return (
        (xx >= ex - half) & (xx <= ex + half)
        & (yy >= ey - half) & (yy <= ey + half)
    )


def _clamp_to_droplet_center(
    solid_mask: np.ndarray,
    bc_map: np.ndarray,
    dx_um: float,
    origin_um: tuple[float, float],
    x_um: float,
    y_um: float,
    radius_um: float,
) -> tuple[float, float]:
    interior = (~solid_mask) & (bc_map == 0)
    if not interior.any():
        return x_um, y_um

    distance_cells = max(1, int(np.ceil(radius_um / dx_um)))
    allowed = interior.copy()
    for dy in range(-distance_cells, distance_cells + 1):
        for dx in range(-distance_cells, distance_cells + 1):
            if dx * dx + dy * dy > distance_cells * distance_cells:
                continue
            allowed &= _shift_bool(interior, dy, dx)

    candidates = np.argwhere(allowed if allowed.any() else interior)
    ox, oy = origin_um
    cx = ox + (candidates[:, 1] + 0.5) * dx_um
    cy = oy + (candidates[:, 0] + 0.5) * dx_um
    idx = int(np.argmin((cx - x_um) ** 2 + (cy - y_um) ** 2))
    return float(cx[idx]), float(cy[idx])


def _shift_bool(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    src_y0 = max(0, -dy)
    src_y1 = mask.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = mask.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _bc_specs(pressure_mbar: float) -> list[BCSpec]:
    inlet = BCSpec("oil_inlet", "inlet", 0.0, 0.0, 1.0, 1.0, phi=1.0,
                   pressure_mbar=pressure_mbar)
    inlet.type_id = 1
    outlet = BCSpec("pressure_outlet", "outlet", 0.0, 0.0, 1.0, 1.0,
                    outlet_bc="pressure")
    outlet.type_id = BC_OUTLET
    return [inlet, outlet]
