"""Session state with JSON config save/load."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def _default_physics() -> dict:
    return {
        "continuous": {"mu_mPas": 1.24, "rho_kg_m3": 1050.0},
        "disperse": {"mu_mPas": 1.75, "rho_kg_m3": 1000.0},
        "interface": {"sigma_mNm": 3.5, "contact_angle_deg": 150.0},
    }


def _migrate_physics(physics: dict) -> dict:
    """Migrate old flat physics format to nested per-phase format."""
    if "mu_oil_mPas" in physics:
        return {
            "continuous": {
                "mu_mPas": physics["mu_oil_mPas"],
                "rho_kg_m3": physics.get("rho_kg_m3", 1050.0),
            },
            "disperse": {
                "mu_mPas": physics.get("mu_aq_mPas", 1.75),
                "rho_kg_m3": 1000.0,
            },
            "interface": {
                "sigma_mNm": physics.get("sigma_mNm", 3.5),
                "contact_angle_deg": physics.get("contact_angle_deg", 150.0),
            },
        }
    return physics


@dataclass
class SessionState:
    dxf_path: str = ""
    dx_um: float = 2.5
    edges: list[dict] = field(default_factory=list)
    phase_regions: list[dict] = field(default_factory=list)
    physics: dict = field(default_factory=_default_physics)
    simulation: dict = field(default_factory=lambda: {
        "tau_oil": 0.55,
        "interface_width": 4,
        "mobility": 0.1,
        "emit_interval": 50,
    })
    timestamp: str = ""

    def save(self, directory: str = "configs") -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().isoformat(timespec="seconds")
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = d / f"{ts}.json"
        data = {
            "timestamp": self.timestamp,
            "geometry": {
                "dxf_path": self.dxf_path,
                "dx_um": self.dx_um,
            },
            "edges": self.edges,
            "phase_regions": self.phase_regions,
            "physics": self.physics,
            "simulation": self.simulation,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> SessionState:
        data = json.loads(Path(path).read_text())
        geom = data.get("geometry", {})
        physics = _migrate_physics(data.get("physics", {}))
        return cls(
            dxf_path=geom.get("dxf_path", ""),
            dx_um=geom.get("dx_um", 2.5),
            edges=data.get("edges", []),
            phase_regions=data.get("phase_regions", []),
            physics=physics,
            simulation=data.get("simulation", {}),
            timestamp=data.get("timestamp", ""),
        )

    @classmethod
    def list_configs(cls, directory: str = "configs") -> list[Path]:
        d = Path(directory)
        if not d.exists():
            return []
        return sorted(d.glob("*.json"), reverse=True)
