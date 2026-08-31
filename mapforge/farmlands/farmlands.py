"""Farmlands (Fase 5 del plan; §S5 del informe forense).

Réplica de ``GRLE._add_farmlands()`` de Maps4FS 1.8.242 (component/grle.py)::

    farmlands = farmyards (si add_farmyards) + fields   ← de textures.json
    por polígono:
        fit_object_into_bounds(margin=farmland_margin, angle=rotation)
            margin = buffer(margin, join_style='mitre') en px (1 px = 1 m)
            intersección con box(0, 0, map_size, map_size)
        coords ÷2  (polygon_points_to_np(divide=2))
        fillPoly(image, id)     ← IDs secuenciales desde 1, tope 254
        <farmland id priceScale="1" npcName="FORESTER"/>  → farmlands.xml
    fill_empty_farmlands: image[image == 0] = 255

Salidas: ``map/data/infoLayer_farmlands.png`` (uint8, (S/2)², píxel = ID) y
``map/config/farmlands.xml`` regenerado (``pricePerHa`` = ``base_price``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

import cv2

if TYPE_CHECKING:
    from mapforge.project import Project

logger = logging.getLogger("mapforge.farmlands")

#: Máximo de farmlands que soporta el Giants Editor (FARMLAND_ID_LIMIT de 1.8).
FARMLAND_ID_LIMIT = 254

#: Valor de relleno de fill_empty_farmlands (zona no comprable, §G del informe).
EMPTY_FARMLAND_VALUE = 255

#: Nombre del PNG de farmlands dentro de ``map/data``.
FARMLANDS_FILENAME = "infoLayer_farmlands.png"

#: Namespace xsi del farmlands.xml del template (se preserva al reescribir).
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

#: Esqueleto usado solo si el template no aportó map/config/farmlands.xml.
_FARMLANDS_XML_SKELETON = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    f'<map xmlns:xsi="{_XSI_NS}" '
    'xsi:noNamespaceSchemaLocation="../../../../shared/xml/schema/farmlands.xsd">\n'
    '  <farmlands infoLayer="farmlands" pricePerHa="60000">\n'
    "  </farmlands>\n"
    "</map>\n"
)


class FarmlandsWriter:
    """Genera ``infoLayer_farmlands.png`` y regenera ``map/config/farmlands.xml``."""

    def __init__(self, project: "Project") -> None:
        self.project = project
        self.logger = project.logger.getChild("farmlands")
        self.settings = project.settings.grle
        self.map_size = project.map.size
        self.map_rotated_size = project.map.rotated_size
        self.rotation = project.map.rotation

    # ------------------------------------------------------------------ rutas

    @property
    def farmlands_png_path(self) -> Path:
        return self.project.paths.map_data_dir / FARMLANDS_FILENAME

    @property
    def farmlands_xml_path(self) -> Path:
        return self.project.paths.map_config_dir / "farmlands.xml"

    # --------------------------------------------------------------- pipeline

    def run(self) -> dict[str, int]:
        """Dibuja los farmlands y sincroniza el XML.

        Returns:
            Estadísticas: ``added`` (farmlands dibujados/declarados) y
            ``skipped`` (polígonos que no cupieron en bounds o fallaron).
        """
        farmlands = self._collect_farmland_polygons()
        image = self._load_farmlands_image()
        tree, farmlands_node = self._load_farmlands_xml()

        farmlands_node.set("pricePerHa", str(self.settings.base_price))
        # Regeneración limpia: sin entradas heredadas de una ejecución previa.
        for existing in list(farmlands_node.findall("farmland")):
            farmlands_node.remove(existing)

        farmland_id = 1
        skipped = 0

        for farmland in farmlands:
            try:
                fitted_farmland = self._fit_polygon_into_bounds(
                    farmland,
                    margin=self.settings.farmland_margin,
                    angle=self.rotation,
                )
            except ValueError as error:
                self.logger.debug(
                    "farmland %s no cupo en bounds: %s", farmland_id, error
                )
                skipped += 1
                continue

            farmland_np = self._polygon_points_to_np(fitted_farmland, divide=2)

            if farmland_id > FARMLAND_ID_LIMIT:
                self.logger.warning(
                    "límite de %d farmlands alcanzado; el resto se omite "
                    "(máximo del Giants Editor)",
                    FARMLAND_ID_LIMIT,
                )
                break

            try:
                cv2.fillPoly(image, [farmland_np], (float(farmland_id),))
            except Exception as error:  # noqa: BLE001 — paridad con 1.8
                self.logger.debug(
                    "farmland %s no se pudo dibujar: %s", farmland_id, error
                )
                skipped += 1
                continue

            entry = ET.SubElement(farmlands_node, "farmland")
            entry.set("id", str(farmland_id))
            entry.set("priceScale", "1")
            entry.set("npcName", "FORESTER")

            farmland_id += 1

        self._save_farmlands_xml(tree)

        if self.settings.fill_empty_farmlands:
            image[image == 0] = EMPTY_FARMLAND_VALUE

        cv2.imwrite(str(self.farmlands_png_path), image)

        added = farmland_id - 1
        self.logger.info(
            "%d farmlands dibujados (%d omitidos) → %s + %s",
            added,
            skipped,
            self.farmlands_png_path.name,
            self.farmlands_xml_path,
        )
        return {"added": added, "skipped": skipped}

    # ------------------------------------------------------------------ datos

    def _collect_farmland_polygons(self) -> list[list[list[int]]]:
        """Farmyards (si ``add_farmyards``) + fields de ``textures.json``,
        en ese orden (IDs de farmyards primero, como 1.8)."""
        textures_json = self.project.paths.textures_json
        if not textures_json.is_file():
            raise FileNotFoundError(
                f"no existe {textures_json}; ejecuta antes el motor de texturas"
            )
        with open(textures_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        farmlands: list[list[list[int]]] = []
        farmyards = data.get("farmyards")
        if farmyards and self.settings.add_farmyards:
            farmlands.extend(farmyards)
            self.logger.debug("%d farmyards de textures.json", len(farmyards))

        fields = data.get("fields")
        if fields:
            farmlands.extend(fields)
            self.logger.debug("%d fields de textures.json", len(fields))

        if not farmlands:
            self.logger.warning("textures.json sin farmyards ni fields")
        return farmlands

    def _load_farmlands_image(self) -> np.ndarray:
        """Carga el PNG en cero creado por ``create_empty_grle_layers``; si no
        existe (uso standalone) se crea en memoria con (S/2)² uint8."""
        path = self.farmlands_png_path
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                return image
        self.logger.warning(
            "%s no existe; se crea en cero (¿faltó create_empty_grle_layers?)",
            path,
        )
        half = self.map_size // 2
        return np.zeros((half, half), dtype=np.uint8)

    # -------------------------------------------------------------------- xml

    def _load_farmlands_xml(self) -> tuple[ET.ElementTree, ET.Element]:
        path = self.farmlands_xml_path
        if not path.is_file():
            self.logger.warning(
                "%s no existe (¿template sin desplegar?); se crea esqueleto", path
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_FARMLANDS_XML_SKELETON, encoding="utf-8")

        tree = ET.parse(path)
        root = tree.getroot()
        farmlands_node = root.find("farmlands")
        if farmlands_node is None:
            raise ValueError(f"elemento <farmlands> no encontrado en {path}")
        return tree, farmlands_node

    def _save_farmlands_xml(self, tree: ET.ElementTree) -> None:
        ET.register_namespace("xsi", _XSI_NS)
        ET.indent(tree, space="  ")
        tree.write(str(self.farmlands_xml_path), encoding="utf-8", xml_declaration=True)

    # -------------------------------------------------------------- geometría

    def _fit_polygon_into_bounds(
        self,
        polygon_points: list[list[int]],
        margin: int = 0,
        angle: float = 0,
    ) -> list[tuple[float, float]]:
        """Réplica de ``Component.fit_object_into_bounds`` de 1.8 (polígonos).

        Con rotación: rota −angle alrededor del centro del lienzo rotado y
        traslada al sistema del mapa. ``margin`` = ``buffer(margin,
        join_style='mitre')`` en píxeles ANTES de la intersección con
        ``box(0, 0, map_size, map_size)``.
        """
        osm_object = Polygon(polygon_points)

        if angle:
            center_x = center_y = self.map_rotated_size // 2
            osm_object = rotate(osm_object, -angle, origin=(center_x, center_y))
            offset = (self.map_size / 2) - (self.map_rotated_size / 2)
            osm_object = translate(osm_object, xoff=offset, yoff=offset)

        if margin:
            osm_object = osm_object.buffer(margin, join_style="mitre")
            if osm_object.is_empty:
                raise ValueError("el polígono quedó vacío tras aplicar el margin")

        bounds = box(0, 0, self.map_size, self.map_size)
        try:
            fitted = osm_object.intersection(bounds)
        except Exception as error:  # noqa: BLE001 — paridad con 1.8
            raise ValueError(f"no se pudo intersecar con bounds: {error}")

        if not isinstance(fitted, Polygon):
            raise ValueError("el polígono ajustado no es válido (se partió)")

        as_list = list(fitted.exterior.coords)
        if not as_list:
            raise ValueError("el polígono ajustado no tiene puntos")
        return as_list

    @staticmethod
    def _polygon_points_to_np(
        polygon_points: list[tuple[float, float]], divide: int | None = None
    ) -> np.ndarray:
        """Réplica de ``ImageComponent.polygon_points_to_np`` (int32 + ``//``)."""
        array = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        if divide:
            return array // divide
        return array
