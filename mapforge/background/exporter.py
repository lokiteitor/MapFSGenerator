"""Exportador del background (Fase 8 del plan; §I del informe forense).

Salidas (patrón del artefacto ``FS25_Valle_Bonito``):

- ``background/decimated_background.obj``: OBJ intermedio del mesh completo
  (mismo nombre y convención de ejes que el golden: XY plano, Z altura ≤ 0).
- ``assets/background/background_texture.png``: textura procedural
  (:mod:`mapforge.background.texture`; PNG en lugar del DDS satelital).
- ``assets/background/background_terrain_part_0{1..4}.i3d``: i3d 1.6 XML por
  cuadrante con ``IndexedTriangleSet`` inline (vértices con normal + uv0
  planar), material con la textura y ``Shape`` con ``translation="0 Y 0"``
  donde ``Y = max_DEM_entrada × z_scaling_factor`` (regla del artefacto:
  114.9416… = 29540/257). El editor GIANTS compila el XML a binario al
  guardar; no replicamos ``.i3d.shapes``.

Convención de ejes del i3d (verificada en el golden
``background_textured_mesh.obj``): ``(x, y, z)_mesh → (x, z, −y)_i3d``
(rotación propia: el winding de las caras se conserva y las normales quedan
hacia +Y).

Cuadrantes (sobre el plano XY del mesh, determinista): parte 01 = x<0,y≥0;
02 = x≥0,y≥0; 03 = x<0,y<0; 04 = x≥0,y<0, asignando cada cara por el signo
del centroide. Las normales de vértice se calculan UNA vez sobre el mesh
completo y se reparten a las partes (evita costuras de iluminación en los
bordes del split).

La integración en ``map.i3d`` (``<File>`` por parte + ``<ReferenceNode
name="background_terrain_part_0N">`` en la raíz de Scene) la hace el escritor
i3d de la Fase 7/9 a partir del dict que devuelve :meth:`BackgroundExporter.run`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh

from mapforge.background.mesh import background_z_scaling_factor, mesh_stats
from mapforge.background.texture import write_background_texture

if TYPE_CHECKING:
    from mapforge.project import Project

#: Nombre base de las partes (patrón del artefacto).
PART_BASENAME = "background_terrain_part"

#: Nombre del fichero de textura referenciado por los i3d.
TEXTURE_FILENAME = "background_texture.png"

_I3D_HEADER = (
    '<?xml version="1.0" encoding="iso-8859-1"?>\n'
    '<i3D xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="{name}"'
    ' version="1.6"'
    ' xsi:noNamespaceSchemaLocation="http://i3d.giants.ch/schema/i3d-1.6.xsd">\n'
    "  <Asset>\n"
    '    <Export program="MapForge" version="1.0" />\n'
    "  </Asset>\n"
    "  <Files>\n"
    '    <File fileId="1" filename="{texture}" />\n'
    "  </Files>\n"
    "  <Materials>\n"
    '    <Material name="{name}_material" materialId="1">\n'
    '      <Texture fileId="1" />\n'
    "    </Material>\n"
    "  </Materials>\n"
)

_I3D_FOOTER = (
    "  <Scene>\n"
    '    <TransformGroup name="{name}" nodeId="4">\n'
    '      <Shape name="{name}_shape" shapeId="1" static="true" nodeId="5"'
    ' castsShadows="false" receiveShadows="true" materialIds="1"'
    ' translation="0 {translation_y} 0" />\n'
    "    </TransformGroup>\n"
    "  </Scene>\n"
    "</i3D>\n"
)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normales de vértice ponderadas por área (numpy puro, sin scipy).

    ``trimesh.vertex_normals`` requiere scipy (y su fallback es un bucle
    Python inviable a este tamaño): se acumulan los productos vectoriales de
    cada cara (norma = 2×área, la ponderación estándar) y se normaliza.
    """
    tri = vertices[faces]
    face_cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_cross)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def _split_quadrants(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Máscaras de caras por cuadrante (01=NW, 02=NE, 03=SW, 04=SE)."""
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    cx, cy = centroids[:, 0], centroids[:, 1]
    return [
        (cx < 0) & (cy >= 0),  # 01
        (cx >= 0) & (cy >= 0),  # 02
        (cx < 0) & (cy < 0),  # 03
        (cx >= 0) & (cy < 0),  # 04
    ]


def _write_part_i3d(
    path: Path,
    name: str,
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    translation_y: float,
) -> None:
    """Escribe el i3d 1.6 XML de una parte con IndexedTriangleSet inline.

    ``positions``/``normals`` llegan en la convención del mesh (XY plano,
    Z altura); aquí se aplica el swap a la convención i3d ``(x, z, −y)``.
    """
    p = np.column_stack([positions[:, 0], positions[:, 2], -positions[:, 1]])
    n = np.column_stack([normals[:, 0], normals[:, 2], -normals[:, 1]])

    lines: list[str] = []
    lines.append(_I3D_HEADER.format(name=name, texture=TEXTURE_FILENAME))
    lines.append("  <Shapes>\n")
    lines.append(
        f'    <IndexedTriangleSet name="{name}_shape" shapeId="1">\n'
        f'      <Vertices count="{len(p)}" normal="true" uv0="true">\n'
    )
    lines.extend(
        f'        <v n="{ni[0]:.6f} {ni[1]:.6f} {ni[2]:.6f}"'
        f' p="{pi[0]:.6f} {pi[1]:.6f} {pi[2]:.6f}"'
        f' t0="{ti[0]:.6f} {ti[1]:.6f}" />\n'
        for pi, ni, ti in zip(p, n, uvs)
    )
    lines.append("      </Vertices>\n")
    lines.append(f'      <Triangles count="{len(faces)}">\n')
    lines.extend(f'        <t vi="{f[0]} {f[1]} {f[2]}" />\n' for f in faces)
    lines.append("      </Triangles>\n")
    lines.append(
        '      <Subsets count="1">\n'
        f'        <Subset firstIndex="0" firstVertex="0"'
        f' numIndices="{len(faces) * 3}" numVertices="{len(p)}" />\n'
        "      </Subsets>\n"
        "    </IndexedTriangleSet>\n"
    )
    lines.append("  </Shapes>\n")
    lines.append(_I3D_FOOTER.format(name=name, translation_y=repr(float(translation_y))))

    path.write_text("".join(lines), encoding="iso-8859-1")


class BackgroundExporter:
    """Exporta OBJ + textura + i3d de las 4 partes del background."""

    def __init__(self, project: "Project") -> None:
        self.project = project

    @property
    def assets_dir(self) -> Path:
        """Directorio de salida de los i3d y la textura (patrón del artefacto)."""
        return self.project.paths.output_dir / "assets" / "background"

    def run(self, mesh: trimesh.Trimesh, dem_full_max: float) -> dict[str, Any]:
        """Exporta el background completo.

        Arguments:
            mesh: mesh construido por
                :func:`mapforge.background.mesh.build_background_mesh`
                (XY recentrado ±background_size/2, Z ≤ 0).
            dem_full_max: valor máximo del DEM FULL de entrada (uint16); fija
                ``translation Y = dem_full_max × z_scaling_factor``.

        Returns:
            dict con rutas y estadísticas (obj, texture, translation_y,
            parts[], mesh) para la integración de Fase 7/9 y
            ``generation_info``.
        """
        project = self.project
        paths = project.paths

        translation_y = float(dem_full_max) * background_z_scaling_factor(project)

        # 1. OBJ intermedio (mismo nombre que el golden).
        paths.background_dir.mkdir(parents=True, exist_ok=True)
        obj_path = paths.background_dir / "decimated_background.obj"
        mesh.export(str(obj_path))
        project.logger.info(
            "OBJ intermedio exportado: %s (%s vértices)", obj_path, len(mesh.vertices)
        )

        # 2. Textura procedural.
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        texture_path = write_background_texture(project, self.assets_dir / TEXTURE_FILENAME)

        # 3. Normales y UVs sobre el mesh completo (sin costuras entre partes).
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        normals = _vertex_normals(vertices, np.asarray(mesh.faces))
        lo = mesh.bounds[0][:2]
        span = mesh.extents[:2]
        span = np.where(span > 0, span, 1.0)
        uvs = np.clip((vertices[:, :2] - lo) / span, 0.0, 1.0)

        # 4. Split en 4 cuadrantes + i3d por parte.
        parts: list[dict[str, Any]] = []
        for index, face_mask in enumerate(_split_quadrants(mesh), start=1):
            name = f"{PART_BASENAME}_{index:02d}"
            faces = mesh.faces[face_mask]
            if len(faces) == 0:
                project.logger.warning("cuadrante %s sin caras; se omite", name)
                continue
            used = np.unique(faces)
            remap = np.full(len(vertices), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            part_faces = remap[faces]

            i3d_path = self.assets_dir / f"{name}.i3d"
            _write_part_i3d(
                i3d_path,
                name,
                vertices[used],
                normals[used],
                uvs[used],
                part_faces,
                translation_y,
            )
            project.logger.info(
                "parte %s: %s vértices, %s caras → %s",
                name,
                len(used),
                len(part_faces),
                i3d_path,
            )
            parts.append(
                {
                    "name": name,
                    "i3d": str(i3d_path.relative_to(paths.output_dir)),
                    "vertices": int(len(used)),
                    "faces": int(len(part_faces)),
                }
            )

        return {
            "obj": str(obj_path),
            "texture": str(texture_path),
            "translation_y": translation_y,
            "parts": parts,
            "mesh": mesh_stats(mesh),
        }
