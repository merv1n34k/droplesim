"""Dialog for area-based BC configuration (inlet/outlet)."""

from __future__ import annotations

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
        pressure_mbar=0.0,
        outlet_bc="pressure",
        rho_target=1.0,
        dx_um=0.0,
        dy_um=0.0,
    ):
        super().__init__(parent)
        self.setWindowTitle("Area BC Zone")
        self.setMinimumWidth(340)

        layout = QFormLayout(self)
        layout.setSpacing(8)

        self._name = ui.line_edit(name)
        layout.addRow("Name:", self._name)

        self._kind = ui.combo_box(["inlet", "outlet"])
        self._kind.setCurrentText(kind)
        self._kind.currentTextChanged.connect(self._on_kind_changed)
        layout.addRow("Kind:", self._kind)

        info = f"Zone: {dx_um:.1f} x {dy_um:.1f} µm"
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

        self._pressure = ui.double_box(
            minimum=0.0,
            maximum=10000.0,
            value=pressure_mbar,
            step=10.0,
            decimals=1,
        )
        self._pressure.setSuffix(" mbar")
        self._pressure.setToolTip("Gauge pressure at inlet (0 = atmospheric)")
        inlet_lay.addRow("Pressure:", self._pressure)

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

    def _on_kind_changed(self, kind: str):
        self._inlet_widget.setVisible(kind == "inlet")
        self._outlet_widget.setVisible(kind == "outlet")
        if kind == "outlet":
            self._on_outlet_bc_changed(self._outlet_bc.currentIndex())

    def _on_outlet_bc_changed(self, idx: int):
        is_pressure = idx == 1
        self._rho_target.setVisible(is_pressure)
        self._rho_target_label.setVisible(is_pressure)

    def result_data(self) -> dict:
        kind = self._kind.currentText()
        data = {
            "name": self._name.text(),
            "kind": kind,
        }
        if kind == "inlet":
            data["phi"] = 0.0 if self._phi.currentIndex() == 0 else 1.0
            data["pressure_mbar"] = self._pressure.value()
        elif kind == "outlet":
            data["phi"] = 1.0
            data["pressure_mbar"] = 0.0
            data["outlet_bc"] = (
                "neumann" if self._outlet_bc.currentIndex() == 0 else "pressure"
            )
            data["rho_target"] = self._rho_target.value()
        return data
