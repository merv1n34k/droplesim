"""Dialog for area-based BC configuration (inlet/outlet)."""

from __future__ import annotations

import math
from typing import Callable

import dropletui as ui
from dropletui.theme import text_qss
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QWidget,
)


class AreaBCDialog(QDialog):
    def __init__(
        self,
        parent=None,
        name="",
        kind="inlet",
        phi=0.0,
        flow_rate=0.0,
        flow_angle_deg=0.0,
        outlet_bc="pressure",
        rho_target=1.0,
        dx_um=0.0,
        dy_um=0.0,
        channel_depth_um=100.0,
        cs_width_fn: Callable[[float], float] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Area BC Zone")
        self.setMinimumWidth(340)
        self._dx_um = dx_um
        self._dy_um = dy_um
        self._channel_depth_um = channel_depth_um
        # Callback: angle_deg → channel width in µm (measured from geometry)
        self._cs_width_fn = cs_width_fn

        layout = QFormLayout(self)
        layout.setSpacing(8)

        self._name = ui.line_edit(name)
        layout.addRow("Name:", self._name)

        self._kind = ui.combo_box(["inlet", "outlet"])
        self._kind.setCurrentText(kind)
        self._kind.currentTextChanged.connect(self._on_kind_changed)
        layout.addRow("Kind:", self._kind)

        info = f"Zone: {dx_um:.1f} x {dy_um:.1f} µm  |  Depth: {channel_depth_um:.0f} µm"
        info_label = QLabel(info)
        info_label.setStyleSheet(text_qss("subtle", font_size=11))
        layout.addRow(info_label)

        # -- Inlet fields --
        self._inlet_widget = QWidget()
        inlet_lay = QFormLayout(self._inlet_widget)
        inlet_lay.setContentsMargins(0, 0, 0, 0)
        inlet_lay.setSpacing(6)

        self._phi = ui.combo_box(["Aqueous (phi=0)", "Oil (phi=1)"])
        self._phi.setCurrentIndex(0 if phi < 0.5 else 1)
        inlet_lay.addRow("Phase:", self._phi)

        self._flow_rate = ui.double_box(
            minimum=0.0,
            maximum=10000.0,
            value=flow_rate,
            step=0.1,
            decimals=3,
        )
        self._flow_rate.setSuffix(" µL/min")
        self._flow_rate.valueChanged.connect(self._update_velocity)
        inlet_lay.addRow("Flow rate:", self._flow_rate)

        self._flow_angle = ui.double_box(
            minimum=0.0,
            maximum=359.0,
            value=flow_angle_deg,
            step=1.0,
            decimals=1,
        )
        self._flow_angle.setSuffix(" °")
        self._flow_angle.setToolTip("0° = right, 90° = up, 180° = left, 270° = down")
        self._flow_angle.valueChanged.connect(self._update_velocity)
        inlet_lay.addRow("Flow direction:", self._flow_angle)

        self._cs_label = QLabel("—")
        self._cs_label.setStyleSheet(text_qss("subtle", font_size=11))
        inlet_lay.addRow("Channel width:", self._cs_label)

        self._vel_label = QLabel("—")
        self._vel_label.setStyleSheet(text_qss("muted"))
        inlet_lay.addRow("Velocity:", self._vel_label)

        layout.addRow(self._inlet_widget)

        # -- Outlet fields --
        self._outlet_widget = QWidget()
        outlet_lay = QFormLayout(self._outlet_widget)
        outlet_lay.setContentsMargins(0, 0, 0, 0)
        outlet_lay.setSpacing(6)

        self._outlet_bc = ui.combo_box(
            ["Zero gradient (Neumann)", "Atmospheric (fixed pressure)"]
        )
        self._outlet_bc.setCurrentIndex(1 if outlet_bc == "pressure" else 0)
        self._outlet_bc.currentIndexChanged.connect(self._on_outlet_bc_changed)
        outlet_lay.addRow("Outlet type:", self._outlet_bc)

        self._rho_target = ui.double_box(
            minimum=0.9,
            maximum=1.1,
            value=rho_target,
            step=0.001,
            decimals=4,
        )
        self._rho_target_label = QLabel("Backpressure (rho):")
        rho_hint = QLabel("1.0 = atmospheric, >1.0 = backpressure")
        rho_hint.setStyleSheet(text_qss("subtle", font_size=11))
        outlet_lay.addRow(self._rho_target_label, self._rho_target)
        self._rho_hint = rho_hint
        outlet_lay.addRow(self._rho_hint)

        layout.addRow(self._outlet_widget)

        # -- Buttons --
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._on_kind_changed(self._kind.currentText())
        self._update_velocity()

    def _on_kind_changed(self, kind: str):
        self._inlet_widget.setVisible(kind == "inlet")
        self._outlet_widget.setVisible(kind == "outlet")
        if kind == "outlet":
            self._on_outlet_bc_changed(self._outlet_bc.currentIndex())

    def _on_outlet_bc_changed(self, idx: int):
        is_pressure = idx == 1
        self._rho_target.setVisible(is_pressure)
        self._rho_target_label.setVisible(is_pressure)

    def _get_cs_width(self) -> float:
        """Get channel width perpendicular to flow direction.

        Uses geometry-based measurement (wall-to-wall scan through box center)
        when available, falls back to box-dimension projection.
        """
        angle = self._flow_angle.value()
        if self._cs_width_fn is not None:
            return self._cs_width_fn(angle)
        angle_rad = math.radians(angle)
        return abs(self._dx_um * math.sin(angle_rad)) + abs(self._dy_um * math.cos(angle_rad))

    def _update_velocity(self):
        val = self._flow_rate.value()
        cs_width = self._get_cs_width()
        self._cs_label.setText(f"{cs_width:.1f} µm x {self._channel_depth_um:.0f} µm")
        if val > 0 and cs_width > 0 and self._channel_depth_um > 0:
            Q_m3s = val * 1e-9 / 60.0
            width_m = cs_width * 1e-6
            depth_m = self._channel_depth_um * 1e-6
            u = Q_m3s / (width_m * depth_m)
            angle = self._flow_angle.value()
            self._vel_label.setText(f"{u:.4e} m/s @ {angle:.0f}°")
            self._vel_label.setStyleSheet(text_qss("primary"))
        else:
            self._vel_label.setText("—")
            self._vel_label.setStyleSheet(text_qss("muted"))

    def result_data(self) -> dict:
        kind = self._kind.currentText()
        data = {
            "name": self._name.text(),
            "kind": kind,
        }
        if kind == "inlet":
            phi = 0.0 if self._phi.currentIndex() == 0 else 1.0
            flow_rate = self._flow_rate.value()
            flow_angle = self._flow_angle.value()
            data["phi"] = phi
            data["flow_rate"] = flow_rate
            data["flow_angle_deg"] = flow_angle

            cs_width = self._get_cs_width()
            if flow_rate > 0 and cs_width > 0 and self._channel_depth_um > 0:
                Q_m3s = flow_rate * 1e-9 / 60.0
                width_m = cs_width * 1e-6
                depth_m = self._channel_depth_um * 1e-6
                u_ms = Q_m3s / (width_m * depth_m)
                angle_rad = math.radians(flow_angle)
                data["ux"] = u_ms * math.cos(angle_rad)
                data["uy"] = u_ms * math.sin(angle_rad)
            else:
                data["ux"] = 0.0
                data["uy"] = 0.0
        elif kind == "outlet":
            data["phi"] = 1.0
            data["flow_rate"] = 0.0
            data["flow_angle_deg"] = 0.0
            data["ux"] = 0.0
            data["uy"] = 0.0
            data["outlet_bc"] = (
                "neumann" if self._outlet_bc.currentIndex() == 0 else "pressure"
            )
            data["rho_target"] = self._rho_target.value()
        return data
