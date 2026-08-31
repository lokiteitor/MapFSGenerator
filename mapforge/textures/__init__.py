"""Subpaquete textures: schema de capas, motor de dibujo y rasterizador (Fase 3)."""

from mapforge.textures.engine import TextureEngine
from mapforge.textures.rasterizer import (
    MIN_POLYGON_POINTS,
    draw_polygon,
    geometry_to_polygon,
    np_to_polygon_points,
    polygon_to_np,
    rasterize,
)
from mapforge.textures.schema import (
    Layer,
    get_base_layer,
    layers_by_priority,
    load_texture_schema,
)

__all__ = [
    "Layer",
    "MIN_POLYGON_POINTS",
    "TextureEngine",
    "draw_polygon",
    "geometry_to_polygon",
    "get_base_layer",
    "layers_by_priority",
    "load_texture_schema",
    "np_to_polygon_points",
    "polygon_to_np",
    "rasterize",
]
