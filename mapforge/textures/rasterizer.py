"""Rasterización de geometrías (en píxeles) a máscaras uint8 (Fase 3 del plan).

Réplica de los conversores de ``Texture`` de Maps4FS 1.8.242 (§S3 del informe
forense), trabajando sobre geometrías shapely YA proyectadas a píxeles:

- ``Polygon`` → se usa SOLO el anillo exterior (1.8 ignora los agujeros:
  ``polygon_to_pixel_coordinates``/``_to_np`` solo leen ``exterior``).
- ``LineString``/``Point`` → ``geometry.buffer(width)`` — FACT-source: el
  ``width`` del schema es el RADIO del buffer (ancho total = 2×width); sin
  width se aplica ``buffer(0)`` (colapsa a polígono vacío → se descarta).
- Otros tipos (MultiPolygon, etc.) → sin conversor (1.8 los salta).
- Polígonos con menos de 3 puntos se saltan (``if not len(polygon) > 2``).
- Dibujo: ``cv2.fillPoly(color=255)``.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

import cv2

#: Puntos mínimos de un polígono dibujable (1.8: ``if not len(polygon) > 2``).
MIN_POLYGON_POINTS = 3


def geometry_to_polygon(
    geometry: BaseGeometry, width: int | None = None
) -> Polygon | None:
    """Convierte una geometría en píxeles al ``Polygon`` que rasteriza 1.8.

    ``Polygon`` pasa tal cual; ``LineString``/``Point`` se bufferizan con
    ``width`` como RADIO (``buffer(width if width else 0)``); cualquier otro
    tipo devuelve ``None`` (no soportado, se salta — réplica de
    ``Texture._converters``).
    """
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, (LineString, Point)):
        return geometry.buffer(width if width else 0)
    return None


def polygon_to_np(polygon: Polygon) -> np.ndarray:
    """Anillo exterior del polígono como array int32 ``(N, 1, 2)`` para
    ``cv2.fillPoly`` (réplica de ``Texture._to_np``; los agujeros se ignoran).
    """
    coords = list(polygon.exterior.coords)
    pts = np.array(coords, np.int32)
    return pts.reshape((-1, 1, 2))


def np_to_polygon_points(np_array: np.ndarray) -> list[tuple[int, int]]:
    """Array de puntos → lista de tuplas ``(x, y)`` (para textures.json)."""
    return [(int(x), int(y)) for x, y in np_array.reshape(-1, 2)]


def draw_polygon(mask: np.ndarray, polygon_np: np.ndarray) -> bool:
    """Dibuja un polígono (array de :func:`polygon_to_np`) sobre ``mask`` con
    ``fillPoly(255)``. Devuelve False si se saltó (<3 puntos, regla de 1.8).
    """
    if not len(polygon_np) > MIN_POLYGON_POINTS - 1:
        return False
    cv2.fillPoly(mask, [polygon_np], color=255)
    return True


def rasterize(
    geometry: BaseGeometry,
    image_size: tuple[int, int],
    width: int | None = None,
) -> np.ndarray:
    """Rasteriza una geometría (en píxeles) a máscara uint8 {0, 255}.

    Azúcar sobre :func:`geometry_to_polygon` + :func:`draw_polygon`, con las
    mismas reglas que el motor: buffer(width)=RADIO para líneas, solo anillo
    exterior, polígonos de <3 puntos ignorados.
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    polygon = geometry_to_polygon(geometry, width)
    if polygon is None or polygon.is_empty:
        return mask
    draw_polygon(mask, polygon_to_np(polygon))
    return mask
