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

        self._history_frames = ui.int_box(minimum=100, maximum=10000, value=1000, step=100)
        hf_row = QHBoxLayout()
        hf_row.addWidget(QLabel("History frames:"))
        hf_row.addWidget(self._history_frames)
        sim_lay.addLayout(hf_row)

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

        # ── Surfactant ──
        surf_group = QGroupBox("Surfactant")
        surf_group.setCheckable(True)
        surf_group.setChecked(False)
        surf_lay = QVBoxLayout(surf_group)
        surf_lay.setSpacing(ui.Theme.SPACE_2)

        # UI uses scaled units; physics_dict() converts to SI
        # D_s, D_bulk: ×10⁻¹⁰ m²/s (user enters 1.0 → 1e-10 SI)
        self._surf_D_s = self._add_spin(
            surf_lay, "D_s [e-10 m²/s]:", 0.01, 100.0, 1.0, 0.1, decimals=2
        )
        self._surf_D_bulk = self._add_spin(
            surf_lay, "D_bulk [e-10 m²/s]:", 0.01, 100.0, 5.0, 0.5, decimals=2
        )
        # ψ_inf: µmol/m² (user enters 3.0 → 3e-6 SI)
        self._surf_psi_inf = self._add_spin(
            surf_lay, "ψ_inf [µmol/m²]:", 0.01, 100.0, 3.0, 0.5, decimals=2
        )
        self._surf_E0 = self._add_spin(
            surf_lay, "E₀:", 0.0, 2.0, 0.2, 0.05, decimals=3
        )
        self._surf_k_a = self._add_spin(
            surf_lay, "k_a [m³/mol·s]:", 0.0, 1000.0, 10.0, 1.0, decimals=2
        )
        self._surf_k_d = self._add_spin(
            surf_lay, "k_d [1/s]:", 0.0, 100.0, 0.1, 0.01, decimals=3
        )
        self._surf_C_inlet = self._add_spin(
            surf_lay, "C_inlet [mol/m³]:", 0.0, 100.0, 0.1, 0.01, decimals=3
        )

        self._surf_group = surf_group
        surf_group.toggled.connect(self._on_surf_toggled)
        self._on_surf_toggled(False)
        layout.addWidget(surf_group)

        # ── Viscoelastic ──
        ve_group = QGroupBox("Viscoelastic")
        ve_group.setCheckable(True)
        ve_group.setChecked(False)
        ve_lay = QVBoxLayout(ve_group)
        ve_lay.setSpacing(ui.Theme.SPACE_2)

        # λ_p: ms (user enters 1.0 → 1e-3 SI)
        self._ve_lambda_p = self._add_spin(
            ve_lay, "λ_p [ms]:", 0.01, 100.0, 1.0, 0.1, decimals=3
        )
        # µ_p: mPa·s (user enters 0.5 → 0.5e-3 SI)
        self._ve_mu_p = self._add_spin(
            ve_lay, "µ_p [mPa·s]:", 0.01, 100.0, 0.5, 0.1, decimals=3
        )
        # κ_ve: artificial diffusion (e-12 m²/s → user enters 1.0 for 1e-12)
        self._ve_kappa = self._add_spin(
            ve_lay, "κ_ve [e-12 m²/s]:", 0.0, 100.0, 1.0, 0.1, decimals=2
        )

        self._ve_group = ve_group
        ve_group.toggled.connect(self._on_ve_toggled)
        self._on_ve_toggled(False)
        layout.addWidget(ve_group)

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

    def _on_surf_toggled(self, checked: bool):
        for child in self._surf_group.findChildren(QWidget):
            child.setVisible(checked)

    def _on_ve_toggled(self, checked: bool):
        for child in self._ve_group.findChildren(QWidget):
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
        d = {
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
        if self._surf_group.isChecked():
            d["surfactant"] = {
                "D_s": self._surf_D_s.value() * 1e-10,       # e-10 m²/s → m²/s
                "D_bulk": self._surf_D_bulk.value() * 1e-10,  # e-10 m²/s → m²/s
                "psi_inf": self._surf_psi_inf.value() * 1e-6, # µmol/m² → mol/m²
                "E0": self._surf_E0.value(),
                "k_a": self._surf_k_a.value(),
                "k_d": self._surf_k_d.value(),
                "C_inlet": self._surf_C_inlet.value(),
            }
        if self._ve_group.isChecked():
            d["viscoelastic"] = {
                "lambda_p": self._ve_lambda_p.value() * 1e-3,   # ms → s
                "mu_p": self._ve_mu_p.value() * 1e-3,           # mPa·s → Pa·s
                "kappa_ve": self._ve_kappa.value() * 1e-12,     # e-12 m²/s → m²/s
            }
        return d

    def simulation_dict(self) -> dict:
        return {
            "tau_c": self._tau_c.value(),
            "interface_width": self._iw.value(),
            "mobility": self._mobility.value(),
            "emit_interval": self._emit_interval.value(),
            "history_frames": self._history_frames.value(),
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
        self._history_frames.setValue(s.get("history_frames", 1000))
        surf = p.get("surfactant")
        if surf:
            self._surf_group.setChecked(True)
            self._surf_D_s.setValue(surf.get("D_s", 1e-10) / 1e-10)      # m²/s → e-10
            self._surf_D_bulk.setValue(surf.get("D_bulk", 5e-10) / 1e-10) # m²/s → e-10
            self._surf_psi_inf.setValue(surf.get("psi_inf", 3e-6) / 1e-6) # mol/m² → µmol/m²
            self._surf_E0.setValue(surf.get("E0", 0.2))
            self._surf_k_a.setValue(surf.get("k_a", 10.0))
            self._surf_k_d.setValue(surf.get("k_d", 0.1))
            self._surf_C_inlet.setValue(surf.get("C_inlet", 0.1))
        else:
            self._surf_group.setChecked(False)
        ve = p.get("viscoelastic")
        if ve:
            self._ve_group.setChecked(True)
            self._ve_lambda_p.setValue(ve.get("lambda_p", 1e-3) / 1e-3)   # s → ms
            self._ve_mu_p.setValue(ve.get("mu_p", 0.5e-3) / 1e-3)         # Pa·s → mPa·s
            self._ve_kappa.setValue(ve.get("kappa_ve", 1e-12) / 1e-12)     # m²/s → e-12
        else:
            self._ve_group.setChecked(False)
