"""Left sidebar: geometry + per-phase physics + simulation parameters."""

from __future__ import annotations

import dropletui as ui
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ParamsPanel(QWidget):
    load_geometry_requested = Signal(str, float)
    channel_depth_changed = Signal(float)
    save_config_requested = Signal()
    load_config_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
        )
        layout.setSpacing(ui.Theme.SPACE_2)

        # ── Geometry ──
        geo_group = QGroupBox("Geometry")
        geo_lay = QVBoxLayout(geo_group)
        geo_lay.setSpacing(ui.Theme.SPACE_2)

        path_row = QHBoxLayout()
        self._dxf_path = ui.line_edit(placeholder="DXF file path...")
        path_row.addWidget(self._dxf_path)
        browse_btn = ui.button("...", size="inline")
        browse_btn.clicked.connect(self._browse_dxf)
        path_row.addWidget(browse_btn)
        geo_lay.addLayout(path_row)

        self._dx_um = self._add_spin(geo_lay, "dx [µm]:", 0.5, 20.0, 2.5, 0.5)

        self._channel_depth = self._add_spin(
            geo_lay, "Channel depth [µm]:", 1.0, 1000.0, 100.0, 10.0, decimals=0
        )
        self._channel_depth.valueChanged.connect(self.channel_depth_changed.emit)

        self._load_btn = ui.button("Load", variant="primary")
        self._load_btn.clicked.connect(self._on_load)
        geo_lay.addWidget(self._load_btn)

        layout.addWidget(geo_group)

        # ── Continuous Phase (oil, phi=1) ──
        cont_group = QGroupBox("Continuous Phase (oil)")
        cont_lay = QVBoxLayout(cont_group)
        cont_lay.setSpacing(ui.Theme.SPACE_2)

        self._mu_cont = self._add_spin(cont_lay, "µ [mPa·s]:", 0.1, 500.0, 1.24, 0.1)
        self._rho_cont = self._add_spin(cont_lay, "ρ [kg/m³]:", 500.0, 2000.0, 1050.0, 10.0)

        layout.addWidget(cont_group)

        # ── Disperse Phase (aqueous, phi=0) ──
        disp_group = QGroupBox("Disperse Phase (aqueous)")
        disp_lay = QVBoxLayout(disp_group)
        disp_lay.setSpacing(ui.Theme.SPACE_2)

        self._mu_disp = self._add_spin(disp_lay, "µ [mPa·s]:", 0.1, 500.0, 1.75, 0.1)
        self._rho_disp = self._add_spin(disp_lay, "ρ [kg/m³]:", 500.0, 2000.0, 1000.0, 10.0)

        layout.addWidget(disp_group)

        # ── Interface ──
        intf_group = QGroupBox("Interface")
        intf_lay = QVBoxLayout(intf_group)
        intf_lay.setSpacing(ui.Theme.SPACE_2)

        self._sigma = self._add_spin(intf_lay, "σ [mN/m]:", 0.1, 100.0, 3.5, 0.5)
        self._contact_angle = self._add_spin(
            intf_lay, "Contact angle [°]:", 90.0, 180.0, 150.0, 1.0
        )

        layout.addWidget(intf_group)

        # ── Simulation ──
        sim_group = QGroupBox("Simulation")
        sim_lay = QVBoxLayout(sim_group)
        sim_lay.setSpacing(ui.Theme.SPACE_2)

        self._emit_interval = ui.int_box(minimum=1, maximum=1000, value=50)
        ei_row = QHBoxLayout()
        ei_row.addWidget(QLabel("Emit interval:"))
        ei_row.addWidget(self._emit_interval)
        sim_lay.addLayout(ei_row)

        # Advanced (collapsible)
        self._advanced = QGroupBox("Advanced")
        self._advanced.setCheckable(True)
        self._advanced.setChecked(False)
        adv_lay = QVBoxLayout(self._advanced)
        adv_lay.setSpacing(ui.Theme.SPACE_2)

        self._tau_c = self._add_spin(adv_lay, "tau_c:", 0.51, 2.0, 0.55, 0.01, decimals=3)

        self._iw = ui.int_box(minimum=2, maximum=8, value=4)
        iw_row = QHBoxLayout()
        iw_row.addWidget(QLabel("Interface W:"))
        iw_row.addWidget(self._iw)
        adv_lay.addLayout(iw_row)

        self._mobility = self._add_spin(
            adv_lay, "Mobility:", 0.01, 1.0, 0.1, 0.01, decimals=3
        )

        self._advanced.toggled.connect(self._on_advanced_toggled)
        self._on_advanced_toggled(False)
        sim_lay.addWidget(self._advanced)

        layout.addWidget(sim_group)

        # ── Config ──
        cfg_group = QGroupBox("Config")
        cfg_lay = QVBoxLayout(cfg_group)
        cfg_lay.setSpacing(ui.Theme.SPACE_2)

        save_btn = ui.button("Save Config")
        save_btn.clicked.connect(self.save_config_requested.emit)
        cfg_lay.addWidget(save_btn)

        load_btn = ui.button("Load Config")
        load_btn.clicked.connect(self._on_load_config)
        cfg_lay.addWidget(load_btn)

        layout.addWidget(cfg_group)
        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_advanced_toggled(self, checked: bool):
        for child in self._advanced.findChildren(QWidget):
            child.setVisible(checked)

    def _add_spin(
        self, parent_layout, label, lo, hi, default, step, decimals=2
    ):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(110)
        row.addWidget(lbl)
        spin = ui.double_box(
            minimum=lo,
            maximum=hi,
            value=default,
            step=step,
            decimals=decimals,
        )
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return spin

    def _browse_dxf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open DXF", "", "DXF Files (*.dxf)")
        if path:
            self._dxf_path.setText(path)

    def _on_load(self):
        path = self._dxf_path.text()
        if not path:
            return
        self.load_geometry_requested.emit(path, self._dx_um.value())

    @property
    def dxf_path(self) -> str:
        return self._dxf_path.text()

    @property
    def dx_um(self) -> float:
        return self._dx_um.value()

    @property
    def channel_depth_um(self) -> float:
        return self._channel_depth.value()

    def physics_dict(self) -> dict:
        return {
            "continuous": {
                "mu_mPas": self._mu_cont.value(),
                "rho_kg_m3": self._rho_cont.value(),
            },
            "disperse": {
                "mu_mPas": self._mu_disp.value(),
                "rho_kg_m3": self._rho_disp.value(),
            },
            "interface": {
                "sigma_mNm": self._sigma.value(),
                "contact_angle_deg": self._contact_angle.value(),
            },
        }

    def simulation_dict(self) -> dict:
        return {
            "tau_c": self._tau_c.value(),
            "interface_width": self._iw.value(),
            "mobility": self._mobility.value(),
            "emit_interval": self._emit_interval.value(),
        }

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "configs", "JSON Files (*.json)"
        )
        if path:
            self.load_config_requested.emit(path)

    def set_from_state(self, state):
        if state.dxf_path:
            self._dxf_path.setText(state.dxf_path)
        self._dx_um.setValue(state.dx_um)
        p = state.physics
        cont = p.get("continuous", {})
        disp = p.get("disperse", {})
        intf = p.get("interface", {})
        self._mu_cont.setValue(cont.get("mu_mPas", 1.24))
        self._rho_cont.setValue(cont.get("rho_kg_m3", 1050.0))
        self._mu_disp.setValue(disp.get("mu_mPas", 1.75))
        self._rho_disp.setValue(disp.get("rho_kg_m3", 1000.0))
        self._sigma.setValue(intf.get("sigma_mNm", 3.5))
        self._contact_angle.setValue(intf.get("contact_angle_deg", 150.0))
        s = state.simulation
        self._tau_c.setValue(s.get("tau_c", 0.55))
        self._iw.setValue(s.get("interface_width", 4))
        self._mobility.setValue(s.get("mobility", 0.1))
        self._emit_interval.setValue(s.get("emit_interval", 50))
