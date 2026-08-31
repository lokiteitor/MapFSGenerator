"""Proyección geográfica de MapForge (Fase 2 del plan).

- ``bbox_from_point``: bbox geodésico desde el centro del mapa y
  ``dist = map_rotated_size / 2`` — misma fórmula que
  ``ox.utils_geo.bbox_from_point`` (desplazamiento norte/sur/este/oeste sobre
  el gran círculo de radio medio terrestre 6 371 009 m).
- ``latlon_to_pixel``: interpolación lineal dentro del bbox con truncado a
  ``int`` — réplica exacta de ``Texture.latlon_to_pixel`` de Maps4FS 1.8.242
  (verificado <1 m contra el artefacto en la Parte I del informe forense).

Convenio de bbox en todo MapForge: ``(north, south, east, west)`` en grados,
el mismo orden que ``Component.get_bbox`` de Maps4FS y que el campo
``Texture.bbox`` de ``generation_info.json``.

Convenio de píxel: x crece hacia el este, y crece hacia el sur (fila 0 = borde
norte del bbox), imagen de ``image_size`` px de lado (``map_rotated_size``).
"""

from __future__ import annotations

import math

from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

# Radio medio terrestre usado por osmnx (ox.utils_geo): metros.
EARTH_RADIUS_M = 6_371_009

Bbox = tuple[float, float, float, float]  # (north, south, east, west)


def bbox_from_point(lat: float, lon: float, dist: float) -> Bbox:
    """Bbox geodésico ``(north, south, east, west)`` a ``dist`` metros del centro.

    Fórmula idéntica a ``ox.utils_geo.bbox_from_point``: el ángulo recorrido
    sobre el gran círculo es ``dist / R`` y el desplazamiento en longitud se
    corrige por ``cos(lat)``. Maps4FS llama a esta función con
    ``dist = map_rotated_size / 2``.
    """
    delta_lat = (dist / EARTH_RADIUS_M) * (180.0 / math.pi)
    delta_lon = (dist / EARTH_RADIUS_M) * (180.0 / math.pi) / math.cos(
        lat * math.pi / 180.0
    )
    north = lat + delta_lat
    south = lat - delta_lat
    east = lon + delta_lon
    west = lon - delta_lon
    return north, south, east, west


def bbox_for_map(lat: float, lon: float, map_rotated_size: int) -> Bbox:
    """Bbox del mapa: centro + ``dist = map_rotated_size / 2`` (regla Maps4FS)."""
    return bbox_from_point(lat, lon, map_rotated_size / 2)


def latlon_to_pixel(
    lat: float,
    lon: float,
    bbox: Bbox,
    image_size: int,
) -> tuple[int, int]:
    """Convierte (lat, lon) a coordenadas de píxel ``(x, y)`` por interpolación
    lineal en el bbox, truncando a ``int`` (réplica de Maps4FS)::

        x = int((lon - west) / (east - west) * image_size)
        y = int((lat - north) / (south - north) * image_size)
    """
    north, south, east, west = bbox
    x = int((lon - west) / (east - west) * image_size)
    y = int((lat - north) / (south - north) * image_size)
    return x, y


def project_geometry(
    geometry: BaseGeometry, bbox: Bbox, image_size: int
) -> BaseGeometry:
    """Proyecta una geometría shapely en grados (x=lon, y=lat) a píxeles.

    Aplica :func:`latlon_to_pixel` vértice a vértice (Point, LineString,
    Polygon con agujeros, y sus variantes Multi*), igual que
    ``polygon_to_pixel_coordinates`` / ``linestring_to_pixel_coordinates``
    de Maps4FS.
    """

    def _px(coords):  # coords shapely: (lon, lat)
        return [latlon_to_pixel(lat, lon, bbox, image_size) for lon, lat in coords]

    if isinstance(geometry, Point):
        return Point(latlon_to_pixel(geometry.y, geometry.x, bbox, image_size))
    if isinstance(geometry, LineString):
        return LineString(_px(geometry.coords))
    if isinstance(geometry, Polygon):
        return Polygon(
            _px(geometry.exterior.coords),
            [_px(ring.coords) for ring in geometry.interiors],
        )
    if isinstance(geometry, MultiLineString):
        return MultiLineString(
            [_px(line.coords) for line in geometry.geoms]
        )
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon(
            [
                (
                    _px(poly.exterior.coords),
                    [_px(ring.coords) for ring in poly.interiors],
                )
                for poly in geometry.geoms
            ]
        )
    raise ValueError(f"Tipo de geometría no soportado: {geometry.geom_type}")


class MapProjection:
    """Proyección de un mapa concreto: centro + tamaño rotado.

    Azúcar sobre las funciones del módulo para que las fases 3-6 no tengan
    que arrastrar el bbox a mano::

        proj = MapProjection(lat, lon, map_rotated_size)
        x, y = proj.latlon_to_pixel(lat, lon)
    """

    def __init__(self, lat: float, lon: float, map_rotated_size: int) -> None:
        self.center = (lat, lon)
        self.image_size = int(map_rotated_size)
        self.bbox: Bbox = bbox_for_map(lat, lon, self.image_size)

    @property
    def north(self) -> float:
        return self.bbox[0]

    @property
    def south(self) -> float:
        return self.bbox[1]

    @property
    def east(self) -> float:
        return self.bbox[2]

    @property
    def west(self) -> float:
        return self.bbox[3]

    def latlon_to_pixel(self, lat: float, lon: float) -> tuple[int, int]:
        return latlon_to_pixel(lat, lon, self.bbox, self.image_size)

    def project_geometry(self, geometry: BaseGeometry) -> BaseGeometry:
        return project_geometry(geometry, self.bbox, self.image_size)
