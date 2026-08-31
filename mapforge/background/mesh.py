"""Mesh del terreno de background (Fase 8 del plan; algoritmo S6 del informe).

Pipeline (réplica de ``component_mesh.mesh_from_np`` de Maps4FS 1.8, con las
desviaciones documentadas abajo):

1. invertir: ``image.max() − image`` (por eso las alturas del mesh son
   negativas y el máximo queda en z = 0),
2. subsample ``[::resize_factor]`` (default 8: 12288² → 1536²),
3. rejilla de 2 triángulos por celda (mismo winding que Maps4FS),
4. rotar 180° sobre Y y después 180° sobre Z,
5. escala ``[rf, rf, z_scaling_factor]`` con ``z_scaling_factor =
   (1/multiplier) × height_scale_multiplier × height_scale/65535`` (réplica de
   ``get_z_scaling_factor``; para el golden = 1/257),
6. decimación quadric (``fast_simplification`` vía
   ``trimesh.simplify_quadric_decimation``), ver :func:`_decimate`, y después
   reescala XY exacta al tamaño original usando los extents,
7. recentrado XY: bbox centrada en el origen (±background_size/2),
8. ``remove_center``: FILTRADO DE CARAS del cuadrado central de lado
   ``map_size`` (decisión técnica del plan: determinista, sin depender de
   manifold3d; Maps4FS hace resta booleana con una caja).

Desviaciones respecto a Maps4FS 1.8 (documentadas):

- **Decimación**: ``decimation_percent`` se interpreta como *porcentaje de
  resolución lineal conservada por eje* → caras objetivo = caras originales ×
  (percent/100)². Con el default 25 el objetivo es 6.25 % de las caras, que es
  lo observado en el golden (331 646/4 712 450 = 7.0 %, 167 556 vértices).
  Maps4FS 1.8 pasa ``percent/100`` directo a trimesh (= ``target_reduction``
  de fast_simplification), que con 25/3 apenas elimina un 22 % de caras y
  dejaría un mesh de ~1.8 M de vértices, inservible como i3d XML e
  incompatible con el artefacto. Como fast_simplification no alcanza
  reducciones profundas en una sola pasada (su umbral de error depende de
  ``aggression``), se itera escalando la agresividad hasta llegar al objetivo
  (:func:`_decimate`). Determinista (fast_simplification no usa RNG).
- **Orden escala/decimación**: se decima DESPUÉS de aplicar la escala métrica
  ``[rf, rf, zf]`` (Maps4FS decima antes). Con las unidades crudas (XY en
  píxeles subsampleados 0..1535, Z en uint16 0..27540) el error quadric está
  tan distorsionado por la anisotropía que fast_simplification se planta en
  ~42 % de reducción incluso con aggression alta (verificado empíricamente);
  en proporciones métricas alcanza el objetivo profundo sin despeinarse. La
  reescala XY exacta por extents se mantiene tras la decimación, como en
  Maps4FS.
- **Recentrado XY**: Maps4FS 1.8 deja el mesh en [0, S]×[−S, 0]; el artefacto
  3.x (``decimated_background.obj``) está recentrado a ±S/2. Replicamos el
  artefacto (Parte I manda en formatos).
- **remove_center** por filtrado de caras: se eliminan las caras cuyos TRES
  vértices caen estrictamente dentro del cuadrado abierto central; las caras
  que tocan el borde sobreviven (el agujero queda como mucho una celda más
  pequeño que ``map_size``, solapando ligeramente el terreno jugable — mejor
  que un hueco).
- Sin ``fix_mesh``: el golden no es watertight (F = 331 646 ≠ 2V−4), o sea que
  la reparación de Maps4FS no cerró nada relevante; se omite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh

if TYPE_CHECKING:
    from mapforge.project import Project

#: Tolerancia sobre el objetivo de caras de la decimación (se acepta hasta
#: objetivo × (1 + tolerancia)).
_DECIMATION_TOLERANCE = 0.10

#: Progreso mínimo por pasada de decimación; por debajo se escala aggression.
_MIN_PROGRESS = 0.005

#: Máximo de pasadas de decimación (red de seguridad).
_MAX_PASSES = 24


def background_z_scaling_factor(project: "Project") -> float:
    """Factor de escala Z del mesh (réplica de ``get_z_scaling_factor``).

    ``(1/multiplier) × height_scale_multiplier × height_scale/65535``; para el
    golden (multiplier=1, height_scale=255) = 1/257. Requiere que la fase DEM
    haya dejado ``project.height_scale``.
    """
    if project.height_scale is None:
        raise ValueError(
            "project.height_scale no está definido: ejecuta la fase DEM antes "
            "del background (o asigna project.height_scale a mano)"
        )
    height_scale = float(project.height_scale)
    multiplier = float(project.settings.dem.multiplier)
    height_scale_multiplier = height_scale / 255.0
    return (1.0 / multiplier) * height_scale_multiplier * (height_scale / 65535.0)


def _grid_mesh(image: np.ndarray) -> trimesh.Trimesh:
    """Rejilla de 2 triángulos por celda sobre ``image`` (z = valor).

    Mismo layout y winding que Maps4FS: vértice (fila i, col j) → índice
    ``i*cols + j``; caras ``[tl, bl, br]`` y ``[tl, br, tr]`` (vectorizado en
    lugar del doble bucle Python del original).
    """
    rows, cols = image.shape
    x, y = np.meshgrid(
        np.arange(cols, dtype=np.float64),
        np.arange(rows, dtype=np.float64),
    )
    vertices = np.column_stack([x.ravel(), y.ravel(), image.ravel().astype(np.float64)])

    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    top_left = (i * cols + j).ravel()
    top_right = top_left + 1
    bottom_left = top_left + cols
    bottom_right = bottom_left + 1

    faces = np.empty((top_left.size * 2, 3), dtype=np.int64)
    faces[0::2] = np.column_stack([top_left, bottom_left, bottom_right])
    faces[1::2] = np.column_stack([top_left, bottom_right, top_right])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _rotate_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """180° sobre Y y después 180° sobre Z (equivale a (x,y,z)→(x,−y,−z))."""
    for axis in ([0, 1, 0], [0, 0, 1]):
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, axis))
    return mesh


def _decimate(
    mesh: trimesh.Trimesh,
    percent: int,
    aggression: int,
    logger: Any = None,
) -> trimesh.Trimesh:
    """Decimación quadric iterativa hasta ``caras × (percent/100)²``.

    ``fast_simplification`` no alcanza reducciones profundas en una sola
    pasada (el umbral de error crece con ``aggression`` y se agota): se
    itera y, cuando una pasada deja de progresar, se sube la agresividad de
    dos en dos (hasta 10). Determinista.
    """
    if not 0 < percent <= 100:
        raise ValueError(f"decimation_percent debe estar en (0, 100], recibido {percent}")

    target_faces = max(4, int(round(len(mesh.faces) * (percent / 100.0) ** 2)))
    agg = int(aggression)
    current = mesh
    for _ in range(_MAX_PASSES):
        if len(current.faces) <= target_faces * (1 + _DECIMATION_TOLERANCE):
            break
        before = len(current.faces)
        simplified = current.simplify_quadric_decimation(
            face_count=target_faces, aggression=agg
        )
        if len(simplified.faces) >= before * (1 - _MIN_PROGRESS):
            if agg >= 10:
                current = simplified
                break
            agg = min(agg + 2, 10)
        current = simplified
        if logger is not None:
            logger.debug(
                "decimación: %s caras (objetivo %s, aggression %s)",
                len(current.faces),
                target_faces,
                agg,
            )
    return current


def _remove_center_faces(mesh: trimesh.Trimesh, remove_size: float) -> trimesh.Trimesh:
    """Elimina las caras del cuadrado central abierto de lado ``remove_size``.

    Cuadrado centrado en el origen del plano XY (el mesh ya está recentrado).
    Una cara se elimina solo si sus tres vértices están estrictamente dentro;
    después se purgan los vértices huérfanos.
    """
    half = remove_size / 2.0
    xy = mesh.vertices[:, :2]
    inside_vertex = np.all(np.abs(xy) < half, axis=1)
    face_inside = inside_vertex[mesh.faces].all(axis=1)

    kept_faces = mesh.faces[~face_inside]
    result = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=kept_faces, process=False)
    result.remove_unreferenced_vertices()
    return result


def build_background_mesh(
    project: "Project",
    dem_full: np.ndarray,
    z_scaling_factor: float | None = None,
) -> trimesh.Trimesh:
    """Construye el mesh de background (``trimesh.Trimesh``) desde el DEM FULL.

    Arguments:
        project: proyecto (settings de background + ``height_scale`` de la
            fase DEM).
        dem_full: DEM completo del background (``background_size²``, uint16;
            el ``FULL.png`` de la fase DEM).
        z_scaling_factor: override del factor Z (default: réplica de
            ``get_z_scaling_factor`` — ver
            :func:`background_z_scaling_factor`).

    Returns:
        Mesh en la convención del ``decimated_background.obj`` del golden:
        XY = bbox ±background_size/2 (plano), Z = alturas en metros en
        [−rango, 0], centro jugable recortado si ``remove_center``.
    """
    if dem_full.ndim != 2 or dem_full.shape[0] != dem_full.shape[1]:
        raise ValueError(f"el DEM del background debe ser cuadrado 2D: {dem_full.shape}")

    cfg = project.settings.background
    proc = cfg.procedural
    resize_factor = int(proc.resize_factor)
    if resize_factor < 1:
        raise ValueError(f"resize_factor debe ser >= 1, recibido {resize_factor}")

    output_size = int(dem_full.shape[0])
    expected = project.map.background_size
    if output_size != expected:
        project.logger.warning(
            "el DEM del background mide %s y background_size es %s; "
            "se usa el tamaño real del DEM",
            output_size,
            expected,
        )

    if z_scaling_factor is None:
        z_scaling_factor = background_z_scaling_factor(project)

    # 1. invertir + 2. subsample.
    image = dem_full.max() - dem_full
    image = image[::resize_factor, ::resize_factor]
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(
            f"DEM demasiado pequeño tras subsample [::{resize_factor}]: {image.shape}"
        )

    # 3. rejilla + 4. rotaciones.
    mesh = _grid_mesh(image)
    project.logger.info(
        "mesh de background inicial: %s vértices, %s caras",
        len(mesh.vertices),
        len(mesh.faces),
    )
    mesh = _rotate_mesh(mesh)

    # 5. escala métrica [rf, rf, zf] ANTES de decimar (ver desviación
    # documentada arriba: el error quadric necesita proporciones métricas).
    mesh.apply_scale([resize_factor, resize_factor, z_scaling_factor])

    # 6. decimación + reescala XY exacta por extents.
    if proc.apply_decimation:
        mesh = _decimate(
            mesh,
            int(proc.decimation_percent),
            int(proc.decimation_aggression),
            logger=project.logger,
        )
        project.logger.info(
            "mesh decimado: %s vértices, %s caras",
            len(mesh.vertices),
            len(mesh.faces),
        )

    x_size, y_size, _ = mesh.extents
    mesh.apply_scale([output_size / x_size, output_size / y_size, 1.0])

    # 7. recentrado XY (convención del artefacto: bbox ±background_size/2).
    bounds_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation([-bounds_center[0], -bounds_center[1], 0.0])

    # 8. remove_center por filtrado de caras.
    if cfg.remove_center:
        mesh = _remove_center_faces(mesh, float(project.map.size))
        project.logger.info(
            "remove_center aplicado (%s m): %s vértices, %s caras",
            project.map.size,
            len(mesh.vertices),
            len(mesh.faces),
        )

    return mesh


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Estadísticas del mesh para logs/validación/generation_info."""
    bounds = mesh.bounds
    extents = mesh.extents
    center = bounds.mean(axis=0)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "x_size": round(float(extents[0]), 4),
        "y_size": round(float(extents[1]), 4),
        "z_size": round(float(extents[2]), 4),
        "x_center": round(float(center[0]), 4),
        "y_center": round(float(center[1]), 4),
        "z_center": round(float(center[2]), 4),
        "x_min": round(float(bounds[0][0]), 4),
        "x_max": round(float(bounds[1][0]), 4),
        "y_min": round(float(bounds[0][1]), 4),
        "y_max": round(float(bounds[1][1]), 4),
        "z_min": round(float(bounds[0][2]), 4),
        "z_max": round(float(bounds[1][2]), 4),
    }
