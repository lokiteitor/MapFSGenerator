"""Fields: polígonos → TransformGroups del map.i3d (Fase 4 del plan; §S4).

Pipeline (réplica de ``I3d._add_fields`` de Maps4FS 1.8.242):

1. polígonos de ``info_layers/textures.json`` (clave ``fields``, píxeles
   top-left, 1 px = 1 m);
2. :func:`fit_polygon_into_bounds` — intersección shapely con el ``box`` del
   mapa (con ``border`` de la capa de fields del texture schema, FACT-source
   ``component.py``);
3. coords al sistema de centro (``x − map_size//2``);
4. centroide ``Polygon.centroid`` truncado a int;
5. puntos relativos enteros (``punto − centroide``);
6. subárbol XML ``TransformGroup field{n}`` (translation = centroide) >
   ``polygonPoints`` (``point{i}`` relativos) + ``nameIndicator`` con ``Note``
   ``field{n}&#xA;{ha:.2f} ha`` (área shapely a hectáreas — 1.8 escribía
   ``0.00 ha``; el ``&#xA;`` es LITERAL en el valor del atributo, quirk de
   Maps4FS verificado byte a byte en el artefacto: ``&amp;#xA;`` en el XML),
   color 4278190080, fixedSize true + ``teleportIndicator``;
7. ``UserAttribute`` por field (angle=0, missionAllowed=true,
   missionOnlyGrass=false, nameIndicatorIndex=1, polygonIndex=0,
   teleportIndicatorIndex=2) apuntando al nodeId del TransformGroup del field.

nodeIds desde 2000 (``NODE_ID_STARTING_VALUE``), replicando la numeración
exacta de 1.8 — incluida su peculiaridad de saltarse un id entre
``polygonPoints`` y ``point1``.

La inserción se hace sobre ``gameplay/fields`` de un ``map.i3d`` ya parseado
(:func:`add_fields_to_i3d`, la usará el escritor i3d de la Fase 7) o sobre un
fichero (:class:`FieldsWriter` / modo standalone ``python -m
mapforge.fields.fields``).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from xml.etree import ElementTree as ET

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

if TYPE_CHECKING:
    from mapforge.project import Project

logger = logging.getLogger("mapforge.fields")

#: Primer nodeId de los fields (FACT-source i3d.py: NODE_ID_STARTING_VALUE).
NODE_ID_STARTING_VALUE = 2000

#: UserAttributes de cada field (FACT-source i3d.py: FIELDS_ATTRIBUTES).
FIELDS_ATTRIBUTES: list[tuple[str, str, str]] = [
    ("angle", "integer", "0"),
    ("missionAllowed", "boolean", "true"),
    ("missionOnlyGrass", "boolean", "false"),
    ("nameIndicatorIndex", "string", "1"),
    ("polygonIndex", "string", "0"),
    ("teleportIndicatorIndex", "string", "2"),
]

#: Color del Note del nameIndicator (FACT artefacto: 4278190080 = negro ARGB).
NOTE_COLOR = "4278190080"

#: m² por hectárea (1 px = 1 m en el lienzo del mapa).
SQUARE_METERS_PER_HA = 10_000.0

#: Indentación del map.i3d (2 espacios, como template y artefacto).
INDENT = "  "


# --------------------------------------------------------------- geometría


def fit_polygon_into_bounds(
    polygon_points: Sequence[Sequence[float]],
    map_size: int,
    map_rotated_size: int | None = None,
    margin: float = 0,
    angle: float = 0,
    border: int = 0,
) -> list[tuple[float, float]]:
    """Encaja un polígono en los límites del mapa.

    Réplica de la rama de polígonos de ``Component.fit_object_into_bounds``
    (component.py 1.8): rotación opcional alrededor del centro del lienzo
    rotado + traslación al lienzo final, ``buffer(margin, 'mitre')`` opcional,
    e intersección con ``box(border, border, S−border, S−border)``.

    Raises:
        ValueError: si el resultado no es un Polygon simple con puntos (mismo
            contrato que 1.8: el llamador salta ese field).
    """
    min_x = min_y = 0 + border
    max_x = max_y = map_size - border

    osm_object = Polygon(polygon_points)

    if angle:
        if map_rotated_size is None:
            raise ValueError("map_rotated_size es obligatorio si angle != 0")
        center_x = center_y = map_rotated_size // 2
        osm_object = rotate(osm_object, -angle, origin=(center_x, center_y))
        offset = (map_size / 2) - (map_rotated_size / 2)
        osm_object = translate(osm_object, xoff=offset, yoff=offset)

    if margin:
        osm_object = osm_object.buffer(margin, join_style="mitre")
        if osm_object.is_empty:
            raise ValueError("The osm_object is empty after adding the margin.")

    bounds = box(min_x, min_y, max_x, max_y)

    try:
        fitted_osm_object = osm_object.intersection(bounds)
    except Exception as e:  # noqa: BLE001 — paridad con 1.8
        raise ValueError(f"Could not fit the osm_object into the bounds: {e}")

    if not isinstance(fitted_osm_object, Polygon):
        raise ValueError("The fitted osm_object is not valid (probably splitted into parts).")

    as_list = list(fitted_osm_object.exterior.coords)
    if not as_list:
        raise ValueError("The fitted osm_object has no points.")
    return as_list


def top_left_to_center(point: Sequence[float], map_size: int) -> tuple[float, float]:
    """Coordenadas top-left → sistema de centro (``x − S//2``), como
    ``Component.top_left_coordinates_to_center`` de 1.8."""
    x, y = point
    return x - map_size // 2, y - map_size // 2


def polygon_center(polygon_points: Sequence[Sequence[float]]) -> tuple[int, int]:
    """Centroide shapely truncado a int (``Component.get_polygon_center``)."""
    center = Polygon(polygon_points).centroid
    return int(center.x), int(center.y)


def polygon_area_ha(polygon_points: Sequence[Sequence[float]]) -> float:
    """Área shapely del polígono en hectáreas (1 px = 1 m → área px² = m²)."""
    return Polygon(polygon_points).area / SQUARE_METERS_PER_HA


# ------------------------------------------------------------- subárbol XML


def _create_element(name: str, data: dict[str, str]) -> ET.Element:
    element = ET.Element(name)
    for key, value in data.items():
        element.set(key, value)
    return element


def get_user_attribute_node(
    node_id: int, attributes: list[tuple[str, str, str]]
) -> ET.Element:
    """``<UserAttribute nodeId>`` con sus ``<Attribute name type value/>``
    (réplica de ``I3d.get_user_attribute_node`` de 1.8)."""
    user_attribute_node = ET.Element("UserAttribute")
    user_attribute_node.set("nodeId", str(node_id))
    for name, attr_type, value in attributes:
        user_attribute_node.append(
            _create_element("Attribute", {"name": name, "type": attr_type, "value": value})
        )
    return user_attribute_node


def build_field_node(
    field_id: int, field_ccs: Sequence[Sequence[float]], node_id: int
) -> tuple[ET.Element | None, int]:
    """Subárbol ``TransformGroup field{n}`` para un polígono en coords centro.

    Réplica de ``I3d._get_field_xml_entry`` de 1.8 — incluida su numeración de
    nodeIds (el bucle de puntos incrementa ANTES de usar, dejando un id sin
    asignar entre ``polygonPoints`` y ``point1``) — con el Note del artefacto
    3.1.2: ``field{n}&#xA;{ha:.2f} ha``. El ``&#xA;`` va como TEXTO LITERAL en
    el atributo (quirk de Maps4FS: el golden contiene ``&amp;#xA;``), no como
    salto de línea real.

    Returns:
        (nodo, node_id actualizado), o (None, node_id) si el centroide no se
        puede calcular (polígono degenerado).
    """
    try:
        cx, cy = polygon_center(field_ccs)
    except Exception as e:  # noqa: BLE001 — paridad con 1.8
        logger.debug("Field %s: centroide no calculable (%s)", field_id, e)
        return None, node_id

    field_node = _create_element(
        "TransformGroup",
        {"name": f"field{field_id}", "translation": f"{cx} 0 {cy}", "nodeId": str(node_id)},
    )
    node_id += 1

    polygon_points_node = _create_element(
        "TransformGroup", {"name": "polygonPoints", "nodeId": str(node_id)}
    )
    node_id += 1

    for point_id, point in enumerate(field_ccs, start=1):
        # Relativos al centroide, enteros (absolute_to_relative + int).
        rx, ry = int(point[0] - cx), int(point[1] - cy)

        node_id += 1  # quirk 1.8: incrementa antes de usar (un id queda sin asignar)
        polygon_points_node.append(
            _create_element(
                "TransformGroup",
                {
                    "name": f"point{point_id}",
                    "translation": f"{rx} 0 {ry}",
                    "nodeId": str(node_id),
                },
            )
        )

    field_node.append(polygon_points_node)

    # nameIndicator + Note (artefacto 3.1.2: superficie real en ha).
    ha = polygon_area_ha(field_ccs)
    node_id += 1
    name_indicator_node = _create_element(
        "TransformGroup", {"name": "nameIndicator", "nodeId": str(node_id)}
    )
    node_id += 1
    name_indicator_node.append(
        _create_element(
            "Note",
            {
                "name": "Note",
                "nodeId": str(node_id),
                "text": f"field{field_id}&#xA;{ha:.2f} ha",
                "color": NOTE_COLOR,
                "fixedSize": "true",
            },
        )
    )
    field_node.append(name_indicator_node)

    node_id += 1
    field_node.append(
        _create_element("TransformGroup", {"name": "teleportIndicator", "nodeId": str(node_id)})
    )

    return field_node, node_id


# --------------------------------------------------- inserción en el map.i3d


@dataclass
class FieldsStats:
    """Resultado de la inserción de fields (para logs/validación)."""

    added: int = 0
    skipped: int = 0
    fields: list[dict[str, Any]] = dataclass_field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"added": self.added, "skipped": self.skipped, "fields": self.fields}


def _element_depth(root: ET.Element, target: ET.Element) -> int:
    """Profundidad de ``target`` bajo ``root`` (root = 0)."""
    parents = {child: parent for parent in root.iter() for child in parent}
    depth = 0
    node = target
    while node in parents:
        node = parents[node]
        depth += 1
    return depth


def _indent_subtree(element: ET.Element, level: int) -> None:
    """Indenta recursivamente un subárbol NUEVO (text/tail con 2 espacios)."""
    children = list(element)
    if not children:
        return
    element.text = "\n" + INDENT * (level + 1)
    for child in children:
        _indent_subtree(child, level + 1)
        child.tail = "\n" + INDENT * (level + 1)
    children[-1].tail = "\n" + INDENT * level


def _append_pretty(parent: ET.Element, new_children: list[ET.Element], level: int) -> None:
    """Añade hijos a ``parent`` manteniendo la indentación del fichero.

    ``level`` es la profundidad de ``parent``. Preserva la estructura existente
    (solo toca el tail del último hijo previo y el text del padre si estaba
    vacío/autocerrado).
    """
    if not new_children:
        return
    existing = list(parent)
    child_indent = "\n" + INDENT * (level + 1)
    close_indent = "\n" + INDENT * level

    if existing:
        existing[-1].tail = child_indent
    else:
        parent.text = child_indent

    for child in new_children:
        _indent_subtree(child, level + 1)
        child.tail = child_indent
        parent.append(child)
    new_children[-1].tail = close_indent


def add_fields_to_i3d(
    root: ET.Element,
    fields: Sequence[Sequence[Sequence[float]]],
    map_size: int,
    map_rotated_size: int | None = None,
    rotation: float = 0,
    border: int = 0,
    node_id_start: int = NODE_ID_STARTING_VALUE,
) -> FieldsStats:
    """Inserta los fields en ``gameplay/fields`` de un map.i3d ya parseado.

    Réplica de ``I3d._add_fields`` de 1.8 sobre un ``ET.Element`` raíz dado
    (lo usará el escritor i3d de la Fase 7). Los ids de field son consecutivos
    solo para los añadidos (los que no encajan en bounds se saltan sin
    consumir id, como 1.8). El UserAttribute de cada field apunta al nodeId
    de su TransformGroup.

    Arguments:
        root: raíz del map.i3d (elemento ``<i3D>``).
        fields: lista de polígonos en píxeles top-left (textures.json).
        map_size: tamaño del mapa en px (= m).
        map_rotated_size: tamaño del lienzo rotado (solo si rotation != 0).
        rotation: ángulo de rotación del mapa.
        border: border de la capa de fields del texture schema (mudDark: 10).
        node_id_start: primer nodeId (default 2000).

    Raises:
        ValueError: si faltan ``gameplay``/``fields``/``UserAttributes`` en el
            i3d (map.i3d no válido — mejor fallar que generar mapa sin fields).
    """
    gameplay_node = root.find(".//TransformGroup[@name='gameplay']")
    if gameplay_node is None:
        raise ValueError("map.i3d sin TransformGroup 'gameplay'")
    fields_node = gameplay_node.find(".//TransformGroup[@name='fields']")
    user_attributes_node = root.find(".//UserAttributes")
    if fields_node is None or user_attributes_node is None:
        raise ValueError("map.i3d sin nodo 'fields' o sin 'UserAttributes'")

    stats = FieldsStats()
    new_field_nodes: list[ET.Element] = []
    new_user_attributes: list[ET.Element] = []

    node_id = node_id_start
    field_id = 1

    for field_polygon in fields:
        try:
            fitted_field = fit_polygon_into_bounds(
                field_polygon,
                map_size=map_size,
                map_rotated_size=map_rotated_size,
                angle=rotation,
                border=border,
            )
        except ValueError as e:
            logger.debug("Field %s no encaja en bounds: %s", field_id, e)
            stats.skipped += 1
            continue

        field_ccs = [top_left_to_center(point, map_size) for point in fitted_field]

        field_node, updated_node_id = build_field_node(field_id, field_ccs, node_id)
        if field_node is None:
            stats.skipped += 1
            continue

        # El UserAttribute referencia el nodeId del TransformGroup del field.
        new_user_attributes.append(get_user_attribute_node(node_id, FIELDS_ATTRIBUTES))
        node_id = updated_node_id

        new_field_nodes.append(field_node)
        stats.fields.append(
            {
                "field_id": field_id,
                "centroid": [int(field_node.get("translation").split()[0]),  # type: ignore[union-attr]
                             int(field_node.get("translation").split()[2])],  # type: ignore[union-attr]
                "n_points": len(field_ccs),
                "ha": round(polygon_area_ha(field_ccs), 2),
            }
        )
        stats.added += 1

        node_id += 1
        field_id += 1

    _append_pretty(fields_node, new_field_nodes, _element_depth(root, fields_node))
    _append_pretty(
        user_attributes_node, new_user_attributes, _element_depth(root, user_attributes_node)
    )

    logger.info("fields: %s añadidos, %s saltados", stats.added, stats.skipped)
    return stats


def write_i3d(tree: ET.ElementTree, path: str | Path) -> None:
    """Escribe el i3d con declaración XML preservando la estructura parseada.

    El ``&#xA;`` literal de los Notes sale serializado como ``&amp;#xA;``,
    byte a byte igual que el artefacto golden (quirk de Maps4FS heredado
    de 1.8: el texto del atributo contiene los caracteres ``&#xA;``).
    """
    payload = ET.tostring(tree.getroot(), encoding="unicode")
    Path(path).write_text('<?xml version="1.0" encoding="utf-8"?>\n' + payload, encoding="utf-8")


def add_fields_to_i3d_file(
    i3d_in: str | Path,
    i3d_out: str | Path,
    fields: Sequence[Sequence[Sequence[float]]],
    map_size: int,
    map_rotated_size: int | None = None,
    rotation: float = 0,
    border: int = 0,
) -> FieldsStats:
    """Parsea ``i3d_in``, inserta los fields y escribe ``i3d_out`` (modo
    fichero, preservando la estructura existente)."""
    tree = ET.parse(i3d_in)
    stats = add_fields_to_i3d(
        tree.getroot(),
        fields,
        map_size=map_size,
        map_rotated_size=map_rotated_size,
        rotation=rotation,
        border=border,
    )
    write_i3d(tree, i3d_out)
    return stats


# ----------------------------------------------------------------- pipeline


def load_fields_from_textures_json(textures_json: str | Path) -> list[list[list[float]]]:
    """Lee la clave ``fields`` de ``info_layers/textures.json`` (contrato de
    la Fase 3). Devuelve [] si no existe la clave."""
    with open(textures_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fields") or []


def field_layer_border(texture_schema_path: str | Path) -> int:
    """Border de la capa con ``usage == "field"`` del texture schema (mudDark:
    10 en el schema FS25), como hace ``I3d._add_fields`` en 1.8."""
    from mapforge.textures.schema import load_texture_schema

    for layer in load_texture_schema(texture_schema_path):
        if layer.usage == "field" and layer.border:
            return layer.border
    return 0


class FieldsWriter:
    """Inserta los fields de ``textures.json`` en el map.i3d del proyecto."""

    def __init__(self, project: "Project") -> None:
        self.project = project
        self.logger = logging.getLogger("mapforge.fields")
        self.stats: FieldsStats | None = None

    def run(self) -> FieldsStats:
        """Genera e inserta los fields en ``map/map.i3d`` del output."""
        paths = self.project.paths
        fields = load_fields_from_textures_json(paths.textures_json)
        if not fields:
            self.logger.warning("Sin fields en %s; no se inserta nada", paths.textures_json)
            self.stats = FieldsStats()
            return self.stats

        border = field_layer_border(paths.texture_schema)
        self.stats = add_fields_to_i3d_file(
            paths.map_i3d,
            paths.map_i3d,
            fields,
            map_size=self.project.map.size,
            map_rotated_size=self.project.map.rotated_size,
            rotation=self.project.map.rotation,
            border=border,
        )
        self.logger.info(
            "fields → %s (%s añadidos)", paths.map_i3d, self.stats.added
        )
        return self.stats


# --------------------------------------------------------------- standalone


def main(argv: Sequence[str] | None = None) -> int:
    """Modo standalone: ``python -m mapforge.fields.fields`` (validación)."""
    parser = argparse.ArgumentParser(
        description="Inserta los fields de un textures.json en un map.i3d (Fase 4)."
    )
    parser.add_argument("--textures-json", required=True, help="info_layers/textures.json")
    parser.add_argument("--map-i3d", required=True, help="map.i3d de entrada (template)")
    parser.add_argument("--output", required=True, help="map.i3d de salida")
    parser.add_argument("--map-size", type=int, required=True)
    parser.add_argument("--rotated-size", type=int, default=None)
    parser.add_argument("--rotation", type=float, default=0.0)
    parser.add_argument(
        "--border",
        type=int,
        default=None,
        help="border de bounds; si se omite se lee del texture schema",
    )
    parser.add_argument(
        "--texture-schema",
        default=None,
        help="texture_schema.json para leer el border de la capa de fields",
    )
    parser.add_argument("--summary", default=None, help="JSON de resumen (opcional)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    border = args.border
    if border is None:
        border = field_layer_border(args.texture_schema) if args.texture_schema else 0

    fields = load_fields_from_textures_json(args.textures_json)
    logger.info("%s fields en %s (border=%s)", len(fields), args.textures_json, border)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats = add_fields_to_i3d_file(
        args.map_i3d,
        args.output,
        fields,
        map_size=args.map_size,
        map_rotated_size=args.rotated_size,
        rotation=args.rotation,
        border=border,
    )

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(stats.to_json(), f, ensure_ascii=False, indent=2)
        logger.info("resumen → %s", args.summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
