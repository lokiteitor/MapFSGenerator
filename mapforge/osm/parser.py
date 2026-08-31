"""Parser OSM XML propio (Fase 2 del plan).

Lee ``custom.osm`` (XML, formato OSM 0.6) con ``xml.etree.ElementTree``
(stdlib) y produce un modelo interno: nodos, ways y relations con sus tags,
más geometrías shapely ensambladas:

- node con tags → ``Point`` (x=lon, y=lat, convención shapely/osmnx).
- way cerrado (primer ref == último ref) → ``Polygon``.
- way abierto → ``LineString``.
- relation ``type=multipolygon`` → ``Polygon``/``MultiPolygon`` con anillos
  outer/inner; los ways miembros abiertos se cosen por extremos hasta cerrar
  anillos.

Sin osmnx ni geopandas (decisión técnica del plan). El matching de tags
(:func:`features_matching`) usa la semántica osmnx de
:mod:`mapforge.osm.tags`.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from mapforge.osm.tags import TagsDefinition, tags_match

logger = logging.getLogger(__name__)

Coord = tuple[float, float]  # (lon, lat)


@dataclass
class OsmWay:
    """Way crudo: refs de nodos + tags."""

    way_id: int
    node_refs: list[int]
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def is_closed(self) -> bool:
        return len(self.node_refs) >= 4 and self.node_refs[0] == self.node_refs[-1]


@dataclass
class OsmRelationMember:
    member_type: str  # "way" | "node" | "relation"
    ref: int
    role: str


@dataclass
class OsmRelation:
    relation_id: int
    members: list[OsmRelationMember]
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def is_multipolygon(self) -> bool:
        return self.tags.get("type") == "multipolygon"


@dataclass
class OsmFeature:
    """Elemento OSM con geometría shapely lista para proyectar.

    ``geometry`` está en grados: x = lon, y = lat (convención shapely/osmnx,
    la misma que espera ``latlon_to_pixel`` recorriendo ``coords`` como
    ``(lon, lat)``).
    """

    element_type: str  # "node" | "way" | "relation"
    element_id: int
    tags: dict[str, str]
    geometry: BaseGeometry


class OsmData:
    """Contenedor del resultado del parseo: nodos, ways, relations y features."""

    def __init__(self) -> None:
        self.nodes: dict[int, Coord] = {}
        self.node_tags: dict[int, dict[str, str]] = {}
        self.ways: dict[int, OsmWay] = {}
        self.relations: dict[int, OsmRelation] = {}
        self.features: list[OsmFeature] = []

    # ------------------------------------------------------------- matching

    def features_matching(self, tags: TagsDefinition | None) -> list[OsmFeature]:
        """Features cuyos tags casan con ``tags`` (semántica osmnx, unión).

        Devuelve las features en orden de documento (nodes, ways, relations),
        que es el orden en que Maps4FS itera el GeoDataFrame de osmnx.
        """
        if not tags:
            return []
        return [f for f in self.features if tags_match(f.tags, tags)]

    # ------------------------------------------------------------- ensamblado

    def build_features(self) -> None:
        """Ensambla las geometrías shapely de nodes/ways/relations parseados."""
        self.features = []

        for node_id, node_tags in self.node_tags.items():
            coord = self.nodes.get(node_id)
            if coord is None:
                continue
            self.features.append(
                OsmFeature("node", node_id, node_tags, Point(coord))
            )

        for way in self.ways.values():
            geometry = self._way_geometry(way)
            if geometry is None:
                continue
            self.features.append(OsmFeature("way", way.way_id, way.tags, geometry))

        for relation in self.relations.values():
            if not relation.is_multipolygon:
                continue
            geometry = self._multipolygon_geometry(relation)
            if geometry is None:
                continue
            self.features.append(
                OsmFeature("relation", relation.relation_id, relation.tags, geometry)
            )

    def _way_coords(self, way: OsmWay) -> list[Coord]:
        coords = []
        for ref in way.node_refs:
            coord = self.nodes.get(ref)
            if coord is None:
                logger.debug("Way %s referencia nodo inexistente %s", way.way_id, ref)
                continue
            coords.append(coord)
        return coords

    def _way_geometry(self, way: OsmWay) -> BaseGeometry | None:
        coords = self._way_coords(way)
        if len(coords) < 2:
            logger.debug("Way %s con <2 nodos resolubles: descartado", way.way_id)
            return None
        if way.is_closed and len(coords) >= 4:
            return Polygon(coords)
        return LineString(coords)

    def _multipolygon_geometry(self, relation: OsmRelation) -> BaseGeometry | None:
        outer_segments: list[list[Coord]] = []
        inner_segments: list[list[Coord]] = []
        for member in relation.members:
            if member.member_type != "way":
                continue
            way = self.ways.get(member.ref)
            if way is None:
                logger.debug(
                    "Relation %s referencia way inexistente %s",
                    relation.relation_id,
                    member.ref,
                )
                continue
            coords = self._way_coords(way)
            if len(coords) < 2:
                continue
            # Rol vacío se trata como outer (convención osmnx).
            if member.role == "inner":
                inner_segments.append(coords)
            else:
                outer_segments.append(coords)

        outer_rings = _assemble_rings(outer_segments)
        inner_rings = _assemble_rings(inner_segments)
        if not outer_rings:
            logger.debug(
                "Relation %s sin anillos outer cerrados: descartada",
                relation.relation_id,
            )
            return None

        outer_polygons = [Polygon(ring) for ring in outer_rings]
        holes_by_outer: list[list[list[Coord]]] = [[] for _ in outer_polygons]
        for ring in inner_rings:
            probe = Point(ring[0])
            assigned = False
            for i, outer in enumerate(outer_polygons):
                if outer.contains(probe) or outer.exterior.distance(probe) == 0.0:
                    holes_by_outer[i].append(ring)
                    assigned = True
                    break
            if not assigned:
                logger.debug(
                    "Relation %s: anillo inner fuera de todo outer, descartado",
                    relation.relation_id,
                )

        polygons = [
            Polygon(ring, holes)
            for ring, holes in zip(outer_rings, holes_by_outer)
        ]
        if len(polygons) == 1:
            return polygons[0]
        return MultiPolygon(polygons)


def _assemble_rings(segments: list[list[Coord]]) -> list[list[Coord]]:
    """Cose ways (listas de coords) por extremos hasta formar anillos cerrados.

    Los ways ya cerrados pasan directos; los abiertos se concatenan
    encadenando extremos coincidentes (con inversión si hace falta). Los
    encadenados que no llegan a cerrar se descartan (mismo comportamiento
    tolerante que osmnx con multipolygons rotos).
    """
    rings: list[list[Coord]] = []
    open_segments: list[list[Coord]] = []
    for segment in segments:
        if len(segment) >= 4 and segment[0] == segment[-1]:
            rings.append(list(segment))
        else:
            open_segments.append(list(segment))

    while open_segments:
        current = open_segments.pop(0)
        progress = True
        while current[0] != current[-1] and progress:
            progress = False
            for i, segment in enumerate(open_segments):
                if segment[0] == current[-1]:
                    current += segment[1:]
                elif segment[-1] == current[-1]:
                    current += segment[-2::-1]
                elif segment[-1] == current[0]:
                    current = segment[:-1] + current
                elif segment[0] == current[0]:
                    current = segment[:0:-1] + current
                else:
                    continue
                open_segments.pop(i)
                progress = True
                break
        if len(current) >= 4 and current[0] == current[-1]:
            rings.append(current)
        else:
            logger.debug("Anillo multipolygon sin cerrar (%d puntos): descartado", len(current))
    return rings


def _collect_tags(elem: ET.Element) -> dict[str, str]:
    tags: dict[str, str] = {}
    for tag_elem in elem.findall("tag"):
        key = tag_elem.get("k")
        value = tag_elem.get("v")
        if key is not None and value is not None:
            tags[key] = value
    return tags


def parse_osm(osm_path: str | Path) -> OsmData:
    """Parsea el fichero OSM XML y devuelve el modelo interno con features.

    Ignora elementos marcados ``action="delete"`` (convención JOSM) y
    referencias a nodos/ways inexistentes (con log en DEBUG).
    """
    osm_path = Path(osm_path)
    if not osm_path.is_file():
        raise FileNotFoundError(f"Fichero OSM no encontrado: {osm_path}")

    data = OsmData()
    for _event, elem in ET.iterparse(str(osm_path), events=("end",)):
        name = elem.tag
        if name not in ("node", "way", "relation"):
            continue
        if elem.get("action") == "delete":
            elem.clear()
            continue
        elem_id = int(elem.get("id"))
        if name == "node":
            data.nodes[elem_id] = (float(elem.get("lon")), float(elem.get("lat")))
            tags = _collect_tags(elem)
            if tags:
                data.node_tags[elem_id] = tags
        elif name == "way":
            refs = [
                int(nd.get("ref")) for nd in elem.findall("nd") if nd.get("ref")
            ]
            data.ways[elem_id] = OsmWay(elem_id, refs, _collect_tags(elem))
        else:  # relation
            members = [
                OsmRelationMember(
                    member.get("type", ""),
                    int(member.get("ref")),
                    member.get("role", ""),
                )
                for member in elem.findall("member")
                if member.get("ref")
            ]
            data.relations[elem_id] = OsmRelation(
                elem_id, members, _collect_tags(elem)
            )
        elem.clear()

    data.build_features()
    logger.info(
        "OSM parseado: %d nodos, %d ways, %d relations, %d features",
        len(data.nodes),
        len(data.ways),
        len(data.relations),
        len(data.features),
    )
    return data


def features_matching(
    osm_data: OsmData, tags: TagsDefinition | None
) -> list[OsmFeature]:
    """Features de ``osm_data`` cuyos tags casan con ``tags`` (semántica osmnx)."""
    return osm_data.features_matching(tags)
