"""Edge labeling dialog: per-type fields for wall/inlet/outlet."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from droplesim.ui.theme import text_qss


class EdgeDialog(QDialog):
    def __init__(
        self,
        parent=None,
        name="",
        kind="wall",
        phi=1.0,
        flow_rate=0.0,
        edge_width_um=0.0,
        channel_depth_um=100.0,
        contact_angle_deg=None,
        outlet_bc="pressure",
        rho_target=1.0,
        normal_flipped=False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Label Edge")
        self.setMinimumWidth(340)
        self._edge_width_um = edge_width_um
        self._channel_depth_um = channel_depth_um

        layout = QFormLayout(self)
        layout.setSpacing(8)

        # -- Common fields --
        self._name = QLineEdit(name)
        layout.addRow("Name:", self._name)

        self._kind = QComboBox()
        self._kind.addItems(["wall", "inlet", "outlet"])
        self._kind.setCurrentText(kind)
        self._kind.currentTextChanged.connect(self._on_kind_changed)
        layout.addRow("Kind:", self._kind)

        # Edge info (always visible)
        info = f"Edge width: {edge_width_um:.1f} µm  |  Depth: {channel_depth_um:.0f} µm"
        info_label = QLabel(info)
        info_label.setStyleSheet(text_qss("subtle", font_size=11))
        layout.addRow(info_label)

        # -- Wall fields --
        self._wall_widget = QWidget()
        wall_lay = QFormLayout(self._wall_widget)
        wall_lay.setContentsMargins(0, 0, 0, 0)
        wall_lay.setSpacing(6)

        self._contact_angle_override = QDoubleSpinBox()
        self._contact_angle_override.setRange(0.0, 180.0)
        self._contact_angle_override.setSingleStep(1.0)
        self._contact_angle_override.setDecimals(1)
        self._contact_angle_override.setSuffix(" °")
        self._contact_angle_override.setSpecialValueText("Use global")
        self._contact_angle_override.setValue(contact_angle_deg if contact_angle_deg is not None else 0.0)
        wall_lay.addRow("Contact angle:", self._contact_angle_override)

        layout.addRow(self._wall_widget)

        # -- Inlet fields --
        self._inlet_widget = QWidget()
        inlet_lay = QFormLayout(self._inlet_widget)
        inlet_lay.setContentsMargins(0, 0, 0, 0)
        inlet_lay.setSpacing(6)

        self._phi = QComboBox()
        self._phi.addItems(["Aqueous (phi=0)", "Oil (phi=1)"])
        self._phi.setCurrentIndex(0 if phi < 0.5 else 1)
        inlet_lay.addRow("Phase:", self._phi)

        self._flow_rate = QDoubleSpinBox()
        self._flow_rate.setRange(0.0, 10000.0)
        self._flow_rate.setSingleStep(0.1)
        self._flow_rate.setDecimals(3)
        self._flow_rate.setSuffix(" µL/min")
        self._flow_rate.setValue(flow_rate)
        self._flow_rate.valueChanged.connect(self._on_flow_rate_changed)
        inlet_lay.addRow("Flow rate:", self._flow_rate)

        self._vel_label = QLabel("—")
        self._vel_label.setStyleSheet("color: #888888;")
        inlet_lay.addRow("Velocity:", self._vel_label)

        self._flip = QCheckBox("Flip flow direction")
        self._flip.setChecked(normal_flipped)
        self._flip.setToolTip("Reverse the auto-detected inward normal")
        inlet_lay.addRow(self._flip)

        layout.addRow(self._inlet_widget)

        # -- Outlet fields --
        self._outlet_widget = QWidget()
        outlet_lay = QFormLayout(self._outlet_widget)
        outlet_lay.setContentsMargins(0, 0, 0, 0)
        outlet_lay.setSpacing(6)

        self._outlet_bc = QComboBox()
        self._outlet_bc.addItems(["Zero gradient (Neumann)", "Atmospheric (fixed pressure)"])
        self._outlet_bc.setCurrentIndex(1 if outlet_bc == "pressure" else 0)
        self._outlet_bc.currentIndexChanged.connect(self._on_outlet_bc_changed)
        outlet_lay.addRow("Outlet type:", self._outlet_bc)

        self._rho_target = QDoubleSpinBox()
        self._rho_target.setRange(0.9, 1.1)
        self._rho_target.setSingleStep(0.001)
        self._rho_target.setDecimals(4)
        self._rho_target.setValue(rho_target)
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
        self._on_flow_rate_changed(flow_rate)

    def _on_kind_changed(self, kind: str):
        self._wall_widget.setVisible(kind == "wall")
        self._inlet_widget.setVisible(kind == "inlet")
        self._outlet_widget.setVisible(kind == "outlet")
        if kind == "outlet":
            self._on_outlet_bc_changed(self._outlet_bc.currentIndex())

    def _on_outlet_bc_changed(self, idx: int):
        is_pressure = idx == 1
        self._rho_target.setVisible(is_pressure)
        self._rho_target_label.setVisible(is_pressure)

    def _on_flow_rate_changed(self, val: float):
        if val > 0 and self._edge_width_um > 0 and self._channel_depth_um > 0:
            Q_m3s = val * 1e-9 / 60.0
            width_m = self._edge_width_um * 1e-6
            depth_m = self._channel_depth_um * 1e-6
            u = Q_m3s / (width_m * depth_m)
            self._vel_label.setText(f"{u:.4e} m/s")
            self._vel_label.setStyleSheet("color: #3498db;")
        else:
            self._vel_label.setText("—")
            self._vel_label.setStyleSheet("color: #888888;")

    def result_data(self) -> dict:
        kind = self._kind.currentText()
        data = {
            "name": self._name.text(),
            "kind": kind,
        }
        if kind == "wall":
            ca = self._contact_angle_override.value()
            data["contact_angle_deg"] = ca if ca > 0 else None
            data["phi"] = 1.0
            data["flow_rate"] = 0.0
        elif kind == "inlet":
            data["phi"] = 0.0 if self._phi.currentIndex() == 0 else 1.0
            data["flow_rate"] = self._flow_rate.value()
            data["normal_flipped"] = self._flip.isChecked()
        elif kind == "outlet":
            data["phi"] = 1.0
            data["flow_rate"] = 0.0
            data["outlet_bc"] = "neumann" if self._outlet_bc.currentIndex() == 0 else "pressure"
            data["rho_target"] = self._rho_target.value()
        return data
