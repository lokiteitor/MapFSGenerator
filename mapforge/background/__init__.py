"""Subpaquete background: mesh, textura procedural y exportador i3d (Fase 8)."""

from mapforge.background.exporter import BackgroundExporter
from mapforge.background.mesh import (
    background_z_scaling_factor,
    build_background_mesh,
    mesh_stats,
)
from mapforge.background.texture import (
    generate_background_texture,
    write_background_texture,
)

__all__ = [
    "BackgroundExporter",
    "background_z_scaling_factor",
    "build_background_mesh",
    "generate_background_texture",
    "mesh_stats",
    "write_background_texture",
]
