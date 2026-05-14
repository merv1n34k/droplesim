"""Tab 3: Edge clicking + area dragging for BC assignment."""

from __future__ import annotations

import logging
import math

import dropletui as ui
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from droplesim.ui.dialogs.area_bc_dialog import AreaBCDialog
from droplesim.ui.dialogs.edge_dialog import EdgeDialog
from droplesim.ui.panels.edge_panel import EdgePanel

log = logging.getLogger(__name__)

_EDGE_COLORS = {
    "wall": (150, 150, 150, 220),
    "inlet": (52, 152, 219, 255),   # #3498db
    "outlet": (231, 76, 60, 255),   # #e74c3c
}

_AREA_INLET_RGBA = np.array([52, 152, 219, 80], dtype=np.uint8)
_AREA_OUTLET_RGBA = np.array([231, 76, 60, 80], dtype=np.uint8)

# Minimum drag distance (in view coords) to distinguish click from drag
_DRAG_THRESHOLD = 3.0


class BCViewBox(pg.ViewBox):
    """ViewBox that emits *point_clicked* on short clicks and
    *rect_drawn* on drag-rectangles.  This lets the same plot support
    edge-click labeling AND area-drag BC creation."""

    rect_drawn = Signal(float, float, float, float)
    point_clicked = Signal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._press_pos = None      # scene coords at press
        self._drag_start = None     # view coords (clamped)
        self._dragging = False
        self._drag_rect = None
        self._bounds = None

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self._bounds = (xmin, ymin, xmax, ymax)

    def _clamp(self, pt):
        x, y = pt.x(), pt.y()
        if self._bounds:
            x = max(self._bounds[0], min(x, self._bounds[2]))
            y = max(self._bounds[1], min(y, self._bounds[3]))
        return x, y

    def mousePressEvent(self, ev):
        if ev.button() == ev.button().LeftButton:
            self._press_pos = ev.pos()
            self._drag_start = self._clamp(self.mapToView(ev.pos()))
            self._dragging = False
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._press_pos is not None:
            delta = ev.pos() - self._press_pos
            dist = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
            if dist > _DRAG_THRESHOLD:
                self._dragging = True

            if self._dragging:
                cx, cy = self._clamp(self.mapToView(ev.pos()))
                if self._drag_rect is not None:
                    self.removeItem(self._drag_rect)
                x1, y1 = self._drag_start
                rect = QRectF(min(x1, cx), min(y1, cy), abs(cx - x1), abs(cy - y1))
                self._drag_rect = pg.QtWidgets.QGraphicsRectItem(rect)
                self._drag_rect.setPen(
                    pg.mkPen("#f39c12", width=2, style=pg.QtCore.Qt.DashLine)
                )
                self.addItem(self._drag_rect)
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._press_pos is not None and ev.button() == ev.button().LeftButton:
            if self._drag_rect is not None:
                self.removeItem(self._drag_rect)
                self._drag_rect = None

            if self._dragging:
                # Finished a drag → emit rectangle
                x2, y2 = self._clamp(self.mapToView(ev.pos()))
                x1, y1 = self._drag_start
                if abs(x2 - x1) > 1 and abs(y2 - y1) > 1:
                    self.rect_drawn.emit(
                        min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                    )
            else:
                # Short click → emit point
                vpt = self.mapToView(ev.pos())
                self.point_clicked.emit(vpt.x(), vpt.y())

            self._press_pos = None
            self._drag_start = None
            self._dragging = False
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)


class EdgeView(QWidget):
    edges_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._vb = BCViewBox()
        self._vb.rect_drawn.connect(self._on_rect_drawn)
        self._vb.point_clicked.connect(self._on_point_clicked)
        self._plot = pg.PlotWidget(
            viewBox=self._vb,
            title="BCs  (click edge to label, drag rectangle for area BC)",
        )
        self._plot.setBackground(ui.Theme.BG_DARK)
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x [µm]")
        self._plot.setLabel("left", "y [µm]")

        self._panel = EdgePanel()
        self._panel.edit_edge_requested.connect(self._on_edit_edge)
        self._panel.delete_edge_requested.connect(self._on_delete_edge)
        self._panel.edit_area_requested.connect(self._on_edit_area)
        self._panel.delete_area_requested.connect(self._on_delete_area)

        layout.addWidget(
            ui.split_view(
                self._plot,
                self._panel,
                side_position="right",
                sizes=(1000, 320),
            )
        )

        self._edge_curves: list[pg.PlotDataItem] = []
        self._arrow_items: list[pg.ArrowItem | None] = []
        self._area_items: list[pg.ImageItem] = []
        self._area_arrow_items: list[pg.ArrowItem | None] = []
        self._edges: list[dict] = []
        self._areas: list[dict] = []
        self._area_counter = 0
        self._edge_polylines_um: list[list[tuple[float, float]]] = []
        self._dx_um = 2.5
        self._origin_um = (0.0, 0.0)
        self._solid_mask = None
        self._channel_depth_um = 100.0

    def set_channel_depth(self, depth_um: float):
        self._channel_depth_um = depth_um

    def set_geometry(
        self,
        solid_mask: np.ndarray,
        dx_um: float,
        origin_um: tuple[float, float],
        edge_polylines_mm: list[list[tuple[float, float]]],
    ):
        self._solid_mask = solid_mask
        self._dx_um = dx_um
        self._origin_um = origin_um

        ny, nx = solid_mask.shape
        ox, oy = origin_um
        self._vb.set_bounds(ox, oy, ox + nx * dx_um, oy + ny * dx_um)

        self._edge_polylines_um = []
        self._edges = []
        for i, poly_mm in enumerate(edge_polylines_mm):
            pts_um = [(x * 1000.0, y * 1000.0) for x, y in poly_mm]
            self._edge_polylines_um.append(pts_um)
            self._edges.append({
                "name": f"edge_{i}",
                "kind": "wall",
                "points_um": pts_um,
                "phi": 1.0,
                "ux": 0.0,
                "uy": 0.0,
                "flow_rate": 0.0,
            })

        self._areas = []
        self._area_counter = 0
        self._redraw_all()
        self._panel.set_edges(self._edges)
        self._panel.set_areas(self._areas)
        self._plot.autoRange()
        log.info("Edge view: %d edges loaded", len(self._edges))

    # ── Drawing ─────────────────────────────────────────────────────────

    def _redraw_all(self):
        self._redraw_edges()
        self._redraw_areas()

    def _redraw_edges(self):
        for c in self._edge_curves:
            self._plot.removeItem(c)
        self._edge_curves.clear()
        for a in self._arrow_items:
            if a is not None:
                self._plot.removeItem(a)
        self._arrow_items.clear()

        for i, edge in enumerate(self._edges):
            pts = edge["points_um"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color = _EDGE_COLORS.get(edge["kind"], (150, 150, 150, 220))
            width = 3 if edge["kind"] != "wall" else 1.5
            curve = self._plot.plot(xs, ys, pen=pg.mkPen(color=color, width=width))
            self._edge_curves.append(curve)

            if edge["kind"] == "inlet" and edge.get("flow_rate", 0) > 0:
                ndir = self._edge_normal_dir(i)
                p0 = np.array(pts[0])
                p1 = np.array(pts[-1])
                mid = (p0 + p1) / 2
                arrow_len = 10 * self._dx_um
                tip = mid + np.array(ndir) * arrow_len
                angle = np.degrees(np.arctan2(-ndir[1], -ndir[0]))
                arrow = pg.ArrowItem(
                    pos=(tip[0], tip[1]),
                    angle=angle,
                    headLen=8,
                    headWidth=6,
                    tailLen=arrow_len * 0.6,
                    tailWidth=2,
                    pen=pg.mkPen(color=(52, 152, 219, 200), width=1),
                    brush=(52, 152, 219, 180),
                )
                self._plot.addItem(arrow)
                self._arrow_items.append(arrow)
            else:
                self._arrow_items.append(None)

    def _redraw_areas(self):
        for item in self._area_items:
            self._plot.removeItem(item)
        self._area_items.clear()
        for a in self._area_arrow_items:
            if a is not None:
                self._plot.removeItem(a)
        self._area_arrow_items.clear()

        for area in self._areas:
            self._add_area_overlay(area)

    def _add_area_overlay(self, area: dict):
        if self._solid_mask is None:
            return
        ny, nx = self._solid_mask.shape
        ox, oy = self._origin_um
        dx = self._dx_um

        xs = ox + (np.arange(nx) + 0.5) * dx
        ys = oy + (np.arange(ny) + 0.5) * dx
        XX, YY = np.meshgrid(xs, ys)

        fluid_in_rect = (
            (XX >= area["x1_um"]) & (XX <= area["x2_um"])
            & (YY >= area["y1_um"]) & (YY <= area["y2_um"])
            & (~self._solid_mask)
        )

        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        color = _AREA_INLET_RGBA if area["kind"] == "inlet" else _AREA_OUTLET_RGBA
        rgba[fluid_in_rect] = color

        img = pg.ImageItem()
        img.setImage(rgba.transpose(1, 0, 2))
        img.setRect(QRectF(ox, oy, nx * dx, ny * dx))
        self._plot.addItem(img)
        self._area_items.append(img)

        # Draw flow arrow for inlet areas with flow_rate > 0
        if area["kind"] == "inlet" and area.get("flow_rate", 0) > 0:
            mid_x = (area["x1_um"] + area["x2_um"]) / 2
            mid_y = (area["y1_um"] + area["y2_um"]) / 2
            flow_angle = math.radians(area.get("flow_angle_deg", 0.0))
            arrow_len = 10 * self._dx_um
            tip_x = mid_x + math.cos(flow_angle) * arrow_len
            tip_y = mid_y + math.sin(flow_angle) * arrow_len
            # pyqtgraph ArrowItem angle: 0° = pointing left, measured CCW
            pg_angle = np.degrees(np.arctan2(-math.sin(flow_angle), -math.cos(flow_angle)))
            arrow = pg.ArrowItem(
                pos=(tip_x, tip_y),
                angle=pg_angle,
                headLen=8,
                headWidth=6,
                tailLen=arrow_len * 0.6,
                tailWidth=2,
                pen=pg.mkPen(color=(52, 152, 219, 200), width=1),
                brush=(52, 152, 219, 180),
            )
            self._plot.addItem(arrow)
            self._area_arrow_items.append(arrow)
        else:
            self._area_arrow_items.append(None)

    # ── Channel width measurement ─────────────────────────────────────

    def _measure_channel_width(self, cx_um: float, cy_um: float, angle_deg: float) -> float:
        """Scan perpendicular to flow through (cx, cy), return contiguous fluid width."""
        solid = self._solid_mask
        if solid is None:
            return 0.0
        dx = self._dx_um
        ox, oy = self._origin_um
        ny, nx = solid.shape

        angle_rad = math.radians(angle_deg)
        # Perpendicular direction (90° CCW from flow)
        px_dir = -math.sin(angle_rad)
        py_dir = math.cos(angle_rad)

        # Walk from center outward in +perp then -perp, stop at solid/OOB
        total = 0
        for sign in (+1, -1):
            for step in range(0 if sign == +1 else 1, max(nx, ny)):
                x = cx_um + sign * px_dir * step * dx
                y = cy_um + sign * py_dir * step * dx
                ix = int((x - ox) / dx)
                iy = int((y - oy) / dx)
                if not (0 <= ix < nx and 0 <= iy < ny) or solid[iy, ix]:
                    break
                total += 1

        return total * dx

    def _make_cs_width_fn(self, x1, y1, x2, y2):
        """Return a callable angle_deg → channel width (µm) for the area center."""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return lambda angle_deg: self._measure_channel_width(cx, cy, angle_deg)

    # ── Click → edge dialog ─────────────────────────────────────────────

    def _on_point_clicked(self, mx: float, my: float):
        if not self._edges:
            return

        best_idx = -1
        best_dist = float("inf")
        threshold = 5 * self._dx_um

        for i, edge in enumerate(self._edges):
            pts = edge["points_um"]
            for k in range(len(pts) - 1):
                d = self._point_to_segment_dist(
                    mx, my, pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1]
                )
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        if best_idx >= 0 and best_dist < threshold:
            self._show_edge_dialog(best_idx)

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        cx, cy = x1 + t * dx, y1 + t * dy
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    def _edge_width_um(self, idx: int) -> float:
        pts = self._edges[idx]["points_um"]
        if len(pts) < 2:
            return 0.0
        p0 = np.array(pts[0])
        p1 = np.array(pts[-1])
        return float(np.linalg.norm(p1 - p0))

    def _edge_normal_dir(self, idx: int) -> tuple[float, float]:
        pts = self._edges[idx]["points_um"]
        if len(pts) < 2:
            return (1.0, 0.0)
        p0 = np.array(pts[0])
        p1 = np.array(pts[-1])
        tangent = p1 - p0
        length = np.linalg.norm(tangent)
        if length < 1e-12:
            return (1.0, 0.0)
        tangent /= length
        nx, ny = -tangent[1], tangent[0]
        if self._solid_mask is not None:
            mid = (p0 + p1) / 2
            ox, oy = self._origin_um
            test_pt = mid + np.array([nx, ny]) * self._dx_um * 3
            ix = int((test_pt[0] - ox) / self._dx_um)
            iy = int((test_pt[1] - oy) / self._dx_um)
            h, w = self._solid_mask.shape
            is_fluid = (0 <= iy < h and 0 <= ix < w and not self._solid_mask[iy, ix])
            if not is_fluid:
                nx, ny = -nx, -ny
        if self._edges[idx].get("normal_flipped", False):
            nx, ny = -nx, -ny
        return (nx, ny)

    def _show_edge_dialog(self, idx: int):
        edge = self._edges[idx]
        width_um = self._edge_width_um(idx)
        dlg = EdgeDialog(
            self,
            name=edge.get("name", ""),
            kind=edge.get("kind", "wall"),
            phi=edge.get("phi", 1.0),
            flow_rate=edge.get("flow_rate", 0.0),
            edge_width_um=width_um,
            channel_depth_um=self._channel_depth_um,
            contact_angle_deg=edge.get("contact_angle_deg"),
            outlet_bc=edge.get("outlet_bc", "pressure"),
            rho_target=edge.get("rho_target", 1.0),
            normal_flipped=edge.get("normal_flipped", False),
        )
        if dlg.exec():
            data = dlg.result_data()
            edge["name"] = data["name"]
            edge["kind"] = data["kind"]
            edge["phi"] = data.get("phi", 1.0)
            edge["flow_rate"] = data.get("flow_rate", 0.0)
            edge["contact_angle_deg"] = data.get("contact_angle_deg")
            edge["outlet_bc"] = data.get("outlet_bc", "pressure")
            edge["rho_target"] = data.get("rho_target", 1.0)
            edge["normal_flipped"] = data.get("normal_flipped", False)

            if data.get("flow_rate", 0) > 0 and data["kind"] == "inlet":
                Q_m3s = data["flow_rate"] * 1e-9 / 60.0
                width_m = width_um * 1e-6
                depth_m = self._channel_depth_um * 1e-6
                area = width_m * depth_m
                if area > 0:
                    u_ms = Q_m3s / area
                    ndir = self._edge_normal_dir(idx)
                    edge["ux"] = u_ms * ndir[0]
                    edge["uy"] = u_ms * ndir[1]
                    flipped = edge.get("normal_flipped", False)
                    log.info(
                        "Edge '%s': Q=%.2f µL/min -> u=%.4e m/s, normal=(%.3f,%.3f), "
                        "ux=%.4e uy=%.4e, flipped=%s",
                        data["name"], data["flow_rate"], u_ms,
                        ndir[0], ndir[1], edge["ux"], edge["uy"], flipped,
                    )
            else:
                edge["ux"] = 0.0
                edge["uy"] = 0.0

            self._redraw_edges()
            self._panel.set_edges(self._edges)
            self.edges_changed.emit()

    def _on_edit_edge(self, idx: int):
        if 0 <= idx < len(self._edges):
            self._show_edge_dialog(idx)

    def _on_delete_edge(self, idx: int):
        if 0 <= idx < len(self._edges):
            self._edges[idx]["kind"] = "wall"
            self._edges[idx]["name"] = f"edge_{idx}"
            self._edges[idx]["flow_rate"] = 0.0
            self._edges[idx]["ux"] = 0.0
            self._edges[idx]["uy"] = 0.0
            self._redraw_edges()
            self._panel.set_edges(self._edges)
            self.edges_changed.emit()

    # ── Drag → area BC dialog ───────────────────────────────────────────

    def _on_rect_drawn(self, x1, y1, x2, y2):
        if self._solid_mask is None:
            return
        dx_um = abs(x2 - x1)
        dy_um = abs(y2 - y1)
        self._area_counter += 1
        dlg = AreaBCDialog(
            self,
            name=f"area_{self._area_counter}",
            dx_um=dx_um,
            dy_um=dy_um,
            channel_depth_um=self._channel_depth_um,
            cs_width_fn=self._make_cs_width_fn(x1, y1, x2, y2),
        )
        if dlg.exec():
            data = dlg.result_data()
            area = {
                "name": data["name"],
                "kind": data["kind"],
                "x1_um": x1,
                "y1_um": y1,
                "x2_um": x2,
                "y2_um": y2,
                "phi": data.get("phi", 1.0),
                "flow_rate": data.get("flow_rate", 0.0),
                "flow_angle_deg": data.get("flow_angle_deg", 0.0),
                "ux": data.get("ux", 0.0),
                "uy": data.get("uy", 0.0),
                "outlet_bc": data.get("outlet_bc", "pressure"),
                "rho_target": data.get("rho_target", 1.0),
            }
            self._areas.append(area)
            self._add_area_overlay(area)
            self._panel.set_areas(self._areas)
            self.edges_changed.emit()
            log.info("Area BC added: %s (%s)", area["name"], area["kind"])

    def _on_edit_area(self, idx: int):
        if not (0 <= idx < len(self._areas)):
            return
        area = self._areas[idx]
        dx_um = abs(area["x2_um"] - area["x1_um"])
        dy_um = abs(area["y2_um"] - area["y1_um"])
        dlg = AreaBCDialog(
            self,
            name=area["name"],
            kind=area["kind"],
            phi=area.get("phi", 1.0),
            flow_rate=area.get("flow_rate", 0.0),
            flow_angle_deg=area.get("flow_angle_deg", 0.0),
            outlet_bc=area.get("outlet_bc", "pressure"),
            rho_target=area.get("rho_target", 1.0),
            dx_um=dx_um,
            dy_um=dy_um,
            channel_depth_um=self._channel_depth_um,
            cs_width_fn=self._make_cs_width_fn(
                area["x1_um"], area["y1_um"], area["x2_um"], area["y2_um"],
            ),
        )
        if dlg.exec():
            data = dlg.result_data()
            area["name"] = data["name"]
            area["kind"] = data["kind"]
            area["phi"] = data.get("phi", 1.0)
            area["flow_rate"] = data.get("flow_rate", 0.0)
            area["flow_angle_deg"] = data.get("flow_angle_deg", 0.0)
            area["ux"] = data.get("ux", 0.0)
            area["uy"] = data.get("uy", 0.0)
            area["outlet_bc"] = data.get("outlet_bc", "pressure")
            area["rho_target"] = data.get("rho_target", 1.0)
            self._redraw_areas()
            self._panel.set_areas(self._areas)
            self.edges_changed.emit()

    def _on_delete_area(self, idx: int):
        if 0 <= idx < len(self._areas):
            self._areas.pop(idx)
            item = self._area_items.pop(idx)
            self._plot.removeItem(item)
            arrow = self._area_arrow_items.pop(idx)
            if arrow is not None:
                self._plot.removeItem(arrow)
            self._panel.set_areas(self._areas)
            self.edges_changed.emit()

    # ── Public API ──────────────────────────────────────────────────────

    def get_edges(self) -> list[dict]:
        return self._edges

    def get_areas(self) -> list[dict]:
        return self._areas

    def set_edges_from_state(self, edges: list[dict]):
        if len(edges) == len(self._edges):
            for i, e in enumerate(edges):
                self._edges[i].update(e)
                self._edges[i]["points_um"] = self._edge_polylines_um[i]
                if e.get("kind") == "inlet" and e.get("flow_rate", 0) > 0:
                    width_um = self._edge_width_um(i)
                    Q_m3s = e["flow_rate"] * 1e-9 / 60.0
                    width_m = width_um * 1e-6
                    depth_m = self._channel_depth_um * 1e-6
                    area = width_m * depth_m
                    if area > 0:
                        u_ms = Q_m3s / area
                        ndir = self._edge_normal_dir(i)
                        self._edges[i]["ux"] = u_ms * ndir[0]
                        self._edges[i]["uy"] = u_ms * ndir[1]
                        flipped = self._edges[i].get("normal_flipped", False)
                        log.info(
                            "Edge '%s' (load): normal=(%.3f,%.3f), ux=%.4e uy=%.4e, flipped=%s",
                            e.get("name", f"edge_{i}"),
                            ndir[0], ndir[1],
                            self._edges[i]["ux"], self._edges[i]["uy"], flipped,
                        )
        self._redraw_edges()
        self._panel.set_edges(self._edges)

    def set_areas_from_state(self, areas: list[dict]):
        for item in self._area_items:
            self._plot.removeItem(item)
        self._area_items.clear()
        self._areas = []
        for a in areas:
            area = dict(a)
            if area.get("kind") == "inlet" and area.get("flow_rate", 0) > 0:
                flow_angle = area.get("flow_angle_deg", 0.0)
                cx = (area["x1_um"] + area["x2_um"]) / 2
                cy = (area["y1_um"] + area["y2_um"]) / 2
                cs_width = self._measure_channel_width(cx, cy, flow_angle)
                if cs_width > 0 and self._channel_depth_um > 0:
                    angle_rad = math.radians(flow_angle)
                    Q_m3s = area["flow_rate"] * 1e-9 / 60.0
                    w_m = cs_width * 1e-6
                    d_m = self._channel_depth_um * 1e-6
                    u_ms = Q_m3s / (w_m * d_m)
                    area["ux"] = u_ms * math.cos(angle_rad)
                    area["uy"] = u_ms * math.sin(angle_rad)
            self._areas.append(area)
            self._add_area_overlay(area)
        self._area_counter = len(self._areas)
        self._panel.set_areas(self._areas)
