"""Splines de tráfico (Fase 6 del plan; §S4 y §H del informe forense).

Pipeline (réplica de ``i3d.py._add_splines`` de Maps4FS 1.8.242):

1. leer ``roads_polylines`` de ``info_layers/textures.json`` (contrato de la
   Fase 3: ``[{"points": [[x, y], ...], "tags": "str(dict de tags)"}, ...]``);
2. por carretera (``road_id`` desde 1): ``fit_linestring_into_bounds``
   (LineString ∩ box del mapa, con rotación si procede) — si no encaja
   (se parte en trozos / queda vacía) la carretera se salta, pero el
   ``road_id`` avanza igual (``enumerate``, como 1.8);
3. ``interpolate_points(num_points=spline_density)``: puntos extra ENTRE cada
   par, truncados a int (réplica EXACTA de ``component.py``); los puntos
   originales conservan su tipo (floats de shapely → ``3143.0``, los
   interpolados salen como ints → ``3143`` — el mismo patrón mixto del golden);
4. ``reversed`` = lista invertida (``[::-1]``) si ``add_reversed_splines``;
5. por spline: coords de centro (``x − map_size//2``), z muestreada del DEM
   ``not_resized`` (:func:`mapforge.terrain.dem.spline_z`: sin interpolación,
   clamp a bordes, ``z = dem[y, x] × 1/multiplier × height_scale/65535``);
6. serialización sobre el esqueleto ``map/splines.i3d`` del template:
   ``NurbsCurve degree=3 form=open`` con ``<cv c="{x}, {z}, {y}"/>`` en
   ``Shapes``, ``<Shape name=… translation="0 0 0" nodeId shapeId/>`` en
   ``Scene`` y ``UserAttribute`` (maxSpeedScale integer 1, speedLimit
   integer 100) en ``UserAttributes``; nombre
   ``spline_{id}_{original|reversed}_{tags}``; nodeId/shapeId consecutivos
   desde 5000.

Formato de salida: indentación de 2 espacios (``ET.indent``), declaración XML
``<?xml version="1.0" encoding="iso-8859-1"?>`` con comillas dobles y prefijo
``xsi`` registrado — byte a byte como el golden salvo los finales de línea
(el golden se escribió en Windows con CRLF; aquí LF).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Sequence
from xml.etree import ElementTree as ET

import cv2
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, box

from mapforge.terrain.dem import spline_z

if TYPE_CHECKING:
    import numpy as np

    from mapforge.project import Project

logger = logging.getLogger("mapforge.splines")

#: nodeId/shapeId inicial de las splines (SPLINES_NODE_ID_STARTING_VALUE, 1.8).
SPLINES_NODE_ID_STARTING_VALUE = 5000

#: UserAttributes de cada spline (i3d.py 1.8, FACT-source).
SPLINE_USER_ATTRIBUTES: tuple[tuple[str, str, str], ...] = (
    ("maxSpeedScale", "integer", "1"),
    ("speedLimit", "integer", "100"),
)

XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"

#: Declaración XML del golden (comillas dobles; ET emitiría comillas simples).
XML_DECLARATION = '<?xml version="1.0" encoding="iso-8859-1"?>\n'


def interpolate_points(
    polyline: list[tuple[int, int]], num_points: int = 4
) -> list[tuple[int, int]]:
    """Añade ``num_points`` puntos ENTRE cada par de puntos de la polilínea.

    Réplica EXACTA de ``Component.interpolate_points`` (component.py 1.8.242):
    los puntos nuevos se calculan a fracciones ``j/(num_points+1)`` del
    segmento y se truncan a int (``int()``); los puntos originales se
    conservan tal cual (sin convertir), por eso en el i3d conviven ``3143.0``
    (originales, floats de shapely) y ``3143`` (interpolados).

    Arguments:
        polyline: lista de puntos (tuplas; los floats pasan sin tocar).
        num_points: puntos extra por par (``spline_density``); < 1 → sin cambio.

    Returns:
        La polilínea con los puntos adicionales.
    """
    if not polyline or num_points < 1:
        return polyline

    interpolated_polyline = []
    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        interpolated_polyline.append(p1)
        for j in range(1, num_points + 1):
            new_point = (
                p1[0] + (p2[0] - p1[0]) * j / (num_points + 1),
                p1[1] + (p2[1] - p1[1]) * j / (num_points + 1),
            )
            interpolated_polyline.append((int(new_point[0]), int(new_point[1])))
    interpolated_polyline.append(polyline[-1])

    return interpolated_polyline


def fit_linestring_into_bounds(
    points: Sequence[Sequence[float]],
    map_size: int,
    map_rotated_size: int | None = None,
    angle: float = 0,
    border: int = 0,
) -> list[tuple[float, float]]:
    """Encaja una polilínea en los límites del mapa.

    Réplica de ``Component.fit_object_into_bounds`` (component.py 1.8.242)
    para LineString: rotación opcional (−angle en torno al centro del lienzo
    rotado + traslación al lienzo final) e intersección con
    ``box(border, border, map_size − border, map_size − border)``.

    Arguments:
        points: puntos de la polilínea en píxeles (sistema top-left).
        map_size: tamaño del mapa en píxeles.
        map_rotated_size: tamaño del lienzo rotado (para ``angle`` ≠ 0).
        angle: rotación del mapa en grados.
        border: margen interior de los límites.

    Returns:
        Lista de puntos (tuplas float, coords de shapely) de la polilínea
        encajada.

    Raises:
        ValueError: si la polilínea no encaja (se parte en varios trozos,
            queda vacía o la intersección falla) — el llamador la salta,
            como 1.8.
    """
    osm_object = LineString(points)

    if angle:
        if map_rotated_size is None:
            raise ValueError("map_rotated_size es obligatorio si angle != 0")
        center_x = center_y = map_rotated_size // 2
        osm_object = rotate(osm_object, -angle, origin=(center_x, center_y))
        offset = (map_size / 2) - (map_rotated_size / 2)
        osm_object = translate(osm_object, xoff=offset, yoff=offset)

    min_x = min_y = 0 + border
    max_x = max_y = map_size - border
    bounds = box(min_x, min_y, max_x, max_y)

    try:
        fitted_osm_object = osm_object.intersection(bounds)
    except Exception as e:  # noqa: BLE001 — paridad con 1.8
        raise ValueError(f"Could not fit the osm_object into the bounds: {e}")

    if not isinstance(fitted_osm_object, LineString):
        raise ValueError("The fitted osm_object is not valid (probably splitted into parts).")

    as_list = list(fitted_osm_object.coords)
    if not as_list:
        raise ValueError("The fitted osm_object has no points.")
    return as_list


def top_left_to_center(point: tuple[float, float], map_size: int) -> tuple[float, float]:
    """Convierte coords top-left → coords de centro (``x − map_size//2``).

    Réplica de ``Component.top_left_coordinates_to_center``: los ints se
    mantienen ints y los floats, floats (afecta al formato del i3d).
    """
    x, y = point
    return x - map_size // 2, y - map_size // 2


class TrafficSplinesWriter:
    """Escribe las NurbsCurve de tráfico sobre el ``map/splines.i3d`` del template.

    Arguments:
        project: proyecto MapForge (rutas, settings, height_scale de la Fase 1).
        dem_not_resized: DEM uint16 ``map_size²`` (crop central pre-resta). Si
            no se pasa, se lee de ``{output}/background/not_resized.png``.
        height_scale: height_scale del pipeline DEM. Si no se pasa, se usa
            ``project.height_scale`` (lo deja la Fase 1).
    """

    def __init__(
        self,
        project: "Project",
        dem_not_resized: "np.ndarray | None" = None,
        height_scale: int | None = None,
    ) -> None:
        self.project = project
        self.logger = logging.getLogger("mapforge.splines")
        self.dem_not_resized = dem_not_resized
        self.height_scale = height_scale

    # ------------------------------------------------------------- entradas

    def _load_roads_polylines(self) -> list[dict[str, Any]]:
        """Lee ``roads_polylines`` de ``info_layers/textures.json``."""
        path = self.project.paths.textures_json
        if not path.is_file():
            self.logger.warning("textures.json no encontrado: %s", path)
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        roads_polylines = data.get("roads_polylines")
        if not roads_polylines:
            self.logger.warning("roads_polylines no encontrado en %s", path)
            return []
        return roads_polylines

    def _resolve_dem(self) -> "np.ndarray":
        """DEM ``not_resized``: el pasado al constructor o el del disco.

        En el fallback de disco se prefiere el DEM **aplanado** cuando existe:
        los CVs deben seguir la superficie de la calzada, no el terreno crudo
        que quedó bajo ella (es lo que hace el golden de Maps4FS 3.x).
        """
        if self.dem_not_resized is not None:
            return self.dem_not_resized
        background_dir = self.project.paths.background_dir
        candidates = (
            background_dir / "not_resized_with_flattened_roads.png",
            background_dir / "not_resized.png",
        )
        for path in candidates:
            dem = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if dem is not None:
                return dem
        raise FileNotFoundError(
            f"DEM not_resized no encontrado: {candidates[-1]} (ejecuta la fase "
            "DEM o pasa dem_not_resized al constructor)"
        )

    def _resolve_height_scale(self) -> int:
        """height_scale: el del constructor o el calculado por la Fase 1."""
        height_scale = (
            self.height_scale
            if self.height_scale is not None
            else self.project.height_scale
        )
        if height_scale is None:
            raise RuntimeError(
                "height_scale no disponible: ejecuta la fase DEM antes de las "
                "splines o pásalo al constructor"
            )
        return int(height_scale)

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        """Genera las splines en ``map/splines.i3d``.

        Returns:
            Número de NurbsCurve escritas (incluye las reversed).
        """
        splines_i3d_path = self.project.paths.splines_i3d
        if not splines_i3d_path.is_file():
            self.logger.warning("splines.i3d no encontrado: %s", splines_i3d_path)
            return 0

        roads_polylines = self._load_roads_polylines()
        if not roads_polylines:
            return 0

        dem_not_resized = self._resolve_dem()
        height_scale = self._resolve_height_scale()
        multiplier = self.project.settings.dem.multiplier
        spline_density = self.project.settings.i3d.spline_density
        add_reversed = self.project.settings.i3d.add_reversed_splines
        map_size = self.project.map.size
        map_rotated_size = self.project.map.rotated_size
        rotation = self.project.map.rotation

        ET.register_namespace("xsi", XSI_NAMESPACE)
        tree = ET.parse(splines_i3d_path)
        root = tree.getroot()
        shapes_node = root.find(".//Shapes")
        scene_node = root.find(".//Scene")
        user_attributes_node = root.find(".//UserAttributes")
        if shapes_node is None or scene_node is None or user_attributes_node is None:
            self.logger.warning(
                "splines.i3d sin nodos Shapes/Scene/UserAttributes: %s",
                splines_i3d_path,
            )
            return 0

        written = 0
        node_id = SPLINES_NODE_ID_STARTING_VALUE
        for road_id, road_info in enumerate(roads_polylines, start=1):
            points = road_info.get("points")
            tags = road_info.get("tags")

            try:
                fitted_road = fit_linestring_into_bounds(
                    points, map_size, map_rotated_size, angle=rotation
                )
            except ValueError as e:
                self.logger.debug(
                    "la carretera %s no encaja en los límites del mapa: %s",
                    road_id,
                    e,
                )
                continue

            fitted_road = interpolate_points(fitted_road, num_points=spline_density)
            fitted_roads = [(fitted_road, "original")]

            if add_reversed:
                reversed_fitted_road = fitted_road[::-1]
                fitted_roads.append((reversed_fitted_road, "reversed"))

            for fitted_points, direction in fitted_roads:
                spline_name = f"spline_{road_id}_{direction}_{tags}"

                shape_element = ET.Element("Shape")
                shape_element.set("name", spline_name)
                shape_element.set("translation", "0 0 0")
                shape_element.set("nodeId", str(node_id))
                shape_element.set("shapeId", str(node_id))
                scene_node.append(shape_element)

                road_ccs = [
                    top_left_to_center(point, map_size) for point in fitted_points
                ]

                nurbs_curve_node = ET.Element("NurbsCurve")
                nurbs_curve_node.set("name", spline_name)
                nurbs_curve_node.set("shapeId", str(node_id))
                nurbs_curve_node.set("degree", "3")
                nurbs_curve_node.set("form", "open")

                for point_ccs, point in zip(road_ccs, fitted_points):
                    cx, cy = point_ccs
                    x, y = point
                    z = spline_z(dem_not_resized, x, y, multiplier, height_scale)
                    cv_node = ET.Element("cv")
                    cv_node.set("c", f"{cx}, {z}, {cy}")
                    nurbs_curve_node.append(cv_node)

                shapes_node.append(nurbs_curve_node)

                user_attribute_node = ET.Element("UserAttribute")
                user_attribute_node.set("nodeId", str(node_id))
                for attr_name, attr_type, attr_value in SPLINE_USER_ATTRIBUTES:
                    attribute_node = ET.Element("Attribute")
                    attribute_node.set("name", attr_name)
                    attribute_node.set("type", attr_type)
                    attribute_node.set("value", attr_value)
                    user_attribute_node.append(attribute_node)
                user_attributes_node.append(user_attribute_node)

                node_id += 1
                written += 1

        self._write_tree(root, splines_i3d_path)
        self.logger.info(
            "splines.i3d: %s NurbsCurve escritas (%s carreteras) → %s",
            written,
            len(roads_polylines),
            splines_i3d_path,
        )
        return written

    # ------------------------------------------------------------ escritura

    @staticmethod
    def _write_tree(root: ET.Element, path) -> None:
        """Serializa con el formato del golden: indent de 2 espacios,
        declaración con comillas dobles, iso-8859-1 y LF."""
        ET.indent(root, space="  ")
        body = ET.tostring(root, encoding="unicode")
        with open(path, "w", encoding="iso-8859-1", newline="\n") as f:
            f.write(XML_DECLARATION + body)
