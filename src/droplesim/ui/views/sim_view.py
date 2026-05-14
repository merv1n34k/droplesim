"""Tab 4: Live simulation display with start/stop/reset and toggleable plots."""

from __future__ import annotations

import dropletui as ui
import numpy as np
import pyqtgraph as pg
from dropletui.theme import text_qss
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from droplesim.ui.views.field_plot import FieldPlot

# ── Colormaps ────────────────────────────────────────────────────────────────

# Phase: blue (aqueous=0) -> red (oil=1)
_PHI_CMAP = pg.ColorMap(
    pos=[0.0, 1.0],
    color=[(52, 152, 219), (231, 76, 60)],
)
# Velocity: viridis-like
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
# Pressure: cool-warm diverging (blue=low, white=1.0, red=high)
_PRS_CMAP = pg.ColorMap(
    pos=[0.0, 0.5, 1.0],
    color=[(59, 76, 192), (221, 221, 221), (180, 4, 38)],
)
# Surfactant: dark -> green -> yellow
_PSI_CMAP = pg.ColorMap(
    pos=[0.0, 0.5, 1.0],
    color=[(15, 15, 15), (39, 174, 96), (241, 196, 15)],
)
# Polymer stress: dark -> purple -> hot pink
_STRESS_CMAP = pg.ColorMap(
    pos=[0.0, 0.5, 1.0],
    color=[(15, 15, 15), (128, 0, 128), (255, 105, 180)],
)


class SimView(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()
    timeline_scrubbed = Signal(int)
    play_toggled = Signal(bool)
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Control buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._start_btn = ui.button("Start", variant="success")
        self._start_btn.clicked.connect(self.start_requested.emit)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = ui.button("Stop", variant="danger")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self._stop_btn)

        self._reset_btn = ui.button("Reset", variant="primary")
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self.reset_requested.emit)
        btn_row.addWidget(self._reset_btn)

        btn_row.addStretch()

        # Plot toggle checkboxes
        self._chk_phase = QCheckBox("Phase")
        self._chk_phase.setChecked(True)
        self._chk_phase.toggled.connect(self._on_toggle)
        btn_row.addWidget(self._chk_phase)

        self._chk_velocity = QCheckBox("Velocity")
        self._chk_velocity.setChecked(True)
        self._chk_velocity.toggled.connect(self._on_toggle)
        btn_row.addWidget(self._chk_velocity)

        self._chk_pressure = QCheckBox("Pressure")
        self._chk_pressure.setChecked(False)
        self._chk_pressure.toggled.connect(self._on_toggle)
        btn_row.addWidget(self._chk_pressure)

        self._chk_surfactant = QCheckBox("Surfactant")
        self._chk_surfactant.setChecked(False)
        self._chk_surfactant.toggled.connect(self._on_toggle)
        btn_row.addWidget(self._chk_surfactant)

        self._chk_stress = QCheckBox("Stress")
        self._chk_stress.setChecked(False)
        self._chk_stress.toggled.connect(self._on_toggle)
        btn_row.addWidget(self._chk_stress)

        layout.addLayout(btn_row)

        # Image display row — 5 FieldPlot instances
        self._img_row = QHBoxLayout()
        self._img_row.setSpacing(4)

        self._phi_field = FieldPlot("phi", _PHI_CMAP)
        self._img_row.addWidget(self._phi_field)

        self._vel_field = FieldPlot("|u|", _VEL_CMAP)
        self._img_row.addWidget(self._vel_field)

        self._prs_field = FieldPlot("rho", _PRS_CMAP)
        self._prs_field.setVisible(False)
        self._img_row.addWidget(self._prs_field)

        self._psi_field = FieldPlot("psi", _PSI_CMAP)
        self._psi_field.setVisible(False)
        self._img_row.addWidget(self._psi_field)

        self._stress_field = FieldPlot("tr(A)", _STRESS_CMAP)
        self._stress_field.setVisible(False)
        self._img_row.addWidget(self._stress_field)

        self._fields = [
            self._phi_field, self._vel_field, self._prs_field,
            self._psi_field, self._stress_field,
        ]

        layout.addLayout(self._img_row, stretch=1)

        # Timeline bar (hidden until frames exist)
        self._timeline_row = QWidget()
        tl_lay = QHBoxLayout(self._timeline_row)
        tl_lay.setContentsMargins(4, 0, 4, 0)
        tl_lay.setSpacing(6)

        self._play_btn = ui.button("Play", variant="primary", size="inline")
        self._play_btn.clicked.connect(self._on_play_clicked)
        tl_lay.addWidget(self._play_btn)

        self._timeline = ui.slider(minimum=0, maximum=0, value=0)
        self._timeline.valueChanged.connect(self._on_slider_moved)
        tl_lay.addWidget(self._timeline, stretch=1)

        self._frame_label = ui.status_label("", kind="muted", small=True)
        self._frame_label.setMinimumWidth(120)
        tl_lay.addWidget(self._frame_label)

        self._speed_spin = ui.double_box(
            minimum=0.1, maximum=10.0, value=1.0, step=0.5, decimals=1, suffix="x",
        )
        tl_lay.addWidget(self._speed_spin)

        self._export_btn = ui.button("Export", size="inline")
        self._export_btn.clicked.connect(self.export_requested.emit)
        tl_lay.addWidget(self._export_btn)

        self._timeline_row.setVisible(False)
        self._is_live = True
        self._playing = False
        layout.addWidget(self._timeline_row)

        # Status line
        self._status = ui.status_label("Ready", kind="muted", small=True)
        mono = QFont()
        ui.configure_monospace_font(mono, 11)
        self._status.setFont(mono)
        self._status.setStyleSheet(text_qss("muted", padding="4px"))
        layout.addWidget(self._status)

        self._first_frame = True

    def _on_toggle(self, _checked: bool):
        self._phi_field.setVisible(self._chk_phase.isChecked())
        self._vel_field.setVisible(self._chk_velocity.isChecked())
        self._prs_field.setVisible(self._chk_pressure.isChecked())
        self._psi_field.setVisible(self._chk_surfactant.isChecked())
        self._stress_field.setVisible(self._chk_stress.isChecked())

    def set_geometry_info(
        self,
        dx_um: float,
        origin_um: tuple[float, float],
        solid_mask: np.ndarray | None = None,
        fluid_yx: np.ndarray | None = None,
    ):
        self._first_frame = True

        if solid_mask is not None and fluid_yx is not None:
            ny, nx = solid_mask.shape
            fy = fluid_yx[:, 0]
            fx = fluid_yx[:, 1]
            for field in self._fields:
                field.set_geometry(ny, nx, fy, fx, dx_um, origin_um)
        else:
            for field in self._fields:
                field.clear_geometry()

    def set_running(self, running: bool):
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        if not running:
            self._status.setStyleSheet(text_qss("muted", padding="4px"))

    def set_has_saved_state(self, has_state: bool):
        self._reset_btn.setEnabled(has_state)
        if has_state:
            self._start_btn.setText("Resume")
        else:
            self._start_btn.setText("Start")

    def update_frame(
        self,
        step: int,
        phi: np.ndarray,
        rho: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        elapsed: float,
        mlups: float,
        extra: dict | None = None,
    ):
        # Phase
        if self._chk_phase.isChecked():
            self._phi_field.update(phi, vmin=0.0, vmax=1.0, fmt=".2f")

        # Velocity
        if self._chk_velocity.isChecked():
            vel = np.sqrt(ux**2 + uy**2)
            vmax = float(vel.max()) or 1.0
            self._vel_field.update(vel, vmin=0.0, vmax=vmax)

        # Pressure (rho deviation from 1.0)
        if self._chk_pressure.isChecked():
            dev = np.clip((rho - 1.0) / 0.01, -1.0, 1.0)
            rmin, rmax = float(rho.min()), float(rho.max())
            self._prs_field.update(dev, vmin=-1.0, vmax=1.0, fmt=".4f")
            # Override legend with actual rho range
            self._prs_field._legend.setText(f"rho: {rmin:.4f} – {rmax:.4f}")

        # Surfactant
        psi = extra.get("psi") if extra else None
        if self._chk_surfactant.isChecked() and psi is not None:
            psi_max = float(psi.max()) or 1.0
            self._psi_field.update(psi, vmin=0.0, vmax=psi_max)

        # Polymer stress: tr(A) - 2
        A_xx = extra.get("A_xx") if extra else None
        if self._chk_stress.isChecked() and A_xx is not None:
            A_yy = extra["A_yy"]
            trace_dev = A_xx + A_yy - 2.0
            s_max = float(np.abs(trace_dev).max()) or 1.0
            self._stress_field.update(trace_dev, vmin=0.0, vmax=s_max)

        if self._first_frame:
            for field in self._fields:
                if field.isVisible():
                    field.auto_range()
            self._first_frame = False

        self._status.setStyleSheet(text_qss("success", padding="4px"))
        self._status.setText(
            f"Step: {step:>8d}   MLUPS: {mlups:>6.1f}   Elapsed: {elapsed:>7.1f}s"
        )

    def _on_slider_moved(self, value: int):
        if not self._is_live:
            self.timeline_scrubbed.emit(value)

    def _on_play_clicked(self):
        self._playing = not self._playing
        self.set_play_state(self._playing)
        self.play_toggled.emit(self._playing)

    def set_timeline_state(self, n_frames: int, current_idx: int, is_live: bool):
        self._is_live = is_live
        self._timeline.blockSignals(True)
        self._timeline.setMaximum(max(0, n_frames - 1))
        self._timeline.setValue(current_idx)
        self._timeline.blockSignals(False)
        self._timeline.setEnabled(not is_live and n_frames > 0)
        self._play_btn.setVisible(not is_live and n_frames > 1)
        self._speed_spin.setVisible(not is_live and n_frames > 1)
        self._export_btn.setVisible(not is_live and n_frames > 0)
        self._timeline_row.setVisible(n_frames > 0)
        if n_frames > 0:
            self._frame_label.setText(f"Frame {current_idx + 1} / {n_frames}")
        else:
            self._frame_label.setText("")

    def set_play_state(self, playing: bool):
        self._playing = playing
        self._play_btn.setText("Pause" if playing else "Play")

    @property
    def playback_speed(self) -> float:
        return self._speed_spin.value()
