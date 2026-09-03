"""Tests de la Fase 8: background procedural (mesh, textura, exportador)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mapforge.background.exporter import BackgroundExporter  # noqa: E402
from mapforge.background.mesh import (  # noqa: E402
    background_z_scaling_factor,
    build_background_mesh,
    mesh_stats,
)
from mapforge.background.texture import generate_background_texture  # noqa: E402
from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import GenerationSettings  # noqa: E402


def make_project(tmp_path, *, size=32, seed=42, height_scale=255):
    """Proyecto mínimo para tests (rutas dummy; no se leen del disco)."""
    paths = ProjectPaths(
        heightmap=tmp_path / "hm.png",
        osm=tmp_path / "in.osm",
        template=tmp_path / "template",
        texture_schema=tmp_path / "texture_schema.json",
        grle_schema=tmp_path / "grle_schema.json",
        output_dir=tmp_path / "out",
    )
    settings = GenerationSettings()
    settings.seed = seed
    project = Project("test", MapParams(size=size), paths, settings)
    project.height_scale = height_scale
    return project


def synthetic_dem(side=64, low=2000, high=29540):
    """DEM sintético uint16: gradiente lineal a lo largo de las columnas."""
    row = np.linspace(low, high, side)
    return np.tile(row, (side, 1)).astype(np.uint16)


# ---------------------------------------------------------------------- mesh


def test_mesh_sin_remove_center(tmp_path):
    project = make_project(tmp_path, size=32)
    project.settings.background.remove_center = False
    proc = project.settings.background.procedural
    proc.resize_factor = 4
    proc.apply_decimation = False

    dem = synthetic_dem(64)
    mesh = build_background_mesh(project, dem)

    # Rejilla 16² intacta (sin decimación): 256 vértices, 2×15² caras.
    assert len(mesh.vertices) == 16 * 16
    assert len(mesh.faces) == 2 * 15 * 15

    stats = mesh_stats(mesh)
    # BBox XY exacta ±output_size/2 (64/2 = 32), recentrada en el origen.
    assert stats["x_min"] == pytest.approx(-32.0)
    assert stats["x_max"] == pytest.approx(32.0)
    assert stats["y_min"] == pytest.approx(-32.0)
    assert stats["y_max"] == pytest.approx(32.0)
    assert stats["x_center"] == pytest.approx(0.0, abs=1e-9)
    assert stats["y_center"] == pytest.approx(0.0, abs=1e-9)

    # Alturas invertidas y escaladas (la inversión usa el máximo GLOBAL del
    # DEM, antes del subsample, como Maps4FS): z = −(max − v)×zf sobre los
    # valores muestreados.
    zf = background_z_scaling_factor(project)
    inverted = (dem.max() - dem)[::4, ::4].astype(np.float64)
    assert stats["z_max"] == pytest.approx(-inverted.min() * zf, abs=1e-4)
    assert stats["z_min"] == pytest.approx(-inverted.max() * zf, rel=1e-6)

    # Orientación: el gradiente crece con la columna (x tras las rotaciones),
    # así que z debe crecer con x (mínimo del DEM → z más negativo).
    verts = np.asarray(mesh.vertices)
    x_lo = verts[verts[:, 0] == verts[:, 0].min()]
    x_hi = verts[verts[:, 0] == verts[:, 0].max()]
    assert x_lo[:, 2].max() < x_hi[:, 2].min()

    # Las normales del grid apuntan hacia arriba (+Z) tras las rotaciones
    # (el gradiente sintético es muy empinado; basta el signo).
    assert np.all(np.asarray(mesh.face_normals)[:, 2] > 0)


def test_mesh_con_remove_center(tmp_path):
    project = make_project(tmp_path, size=32)
    project.settings.background.remove_center = True
    proc = project.settings.background.procedural
    proc.resize_factor = 4
    proc.apply_decimation = False

    dem = synthetic_dem(64)
    mesh = build_background_mesh(project, dem)

    full_project = make_project(tmp_path, size=32)
    full_project.settings.background.remove_center = False
    full_project.settings.background.procedural.resize_factor = 4
    full_project.settings.background.procedural.apply_decimation = False
    full_mesh = build_background_mesh(full_project, dem)

    # Se han eliminado caras (las del cuadrado central de lado map_size=32).
    assert len(mesh.faces) < len(full_mesh.faces)
    assert len(mesh.faces) > 0

    # Ninguna cara restante tiene sus TRES vértices estrictamente dentro del
    # cuadrado central abierto.
    verts = np.asarray(mesh.vertices)
    inside = np.all(np.abs(verts[:, :2]) < 16.0, axis=1)
    assert not np.any(inside[np.asarray(mesh.faces)].all(axis=1))

    # La bbox exterior no cambia.
    assert mesh_stats(mesh)["x_min"] == pytest.approx(-32.0)
    assert mesh_stats(mesh)["x_max"] == pytest.approx(32.0)


def test_mesh_decimacion_reduce_a_porcentaje_por_eje(tmp_path):
    project = make_project(tmp_path, size=64)
    project.settings.background.remove_center = False
    proc = project.settings.background.procedural
    proc.resize_factor = 2
    proc.apply_decimation = True
    proc.decimation_percent = 25
    proc.decimation_aggression = 3

    rng = np.random.default_rng(7)
    side = 256
    dem = (rng.random((side, side)) * 20000 + 2000).astype(np.uint16)
    mesh = build_background_mesh(project, dem)

    grid = side // 2
    original_faces = 2 * (grid - 1) ** 2
    target = original_faces * (25 / 100.0) ** 2
    # Llega al objetivo (con la tolerancia de _decimate) y no lo pulveriza.
    assert len(mesh.faces) <= target * 1.15
    assert len(mesh.faces) >= target * 0.25


def test_mesh_requiere_height_scale(tmp_path):
    project = make_project(tmp_path)
    project.height_scale = None
    with pytest.raises(ValueError, match="height_scale"):
        build_background_mesh(project, synthetic_dem(64))


def test_mesh_dem_no_cuadrado(tmp_path):
    project = make_project(tmp_path)
    with pytest.raises(ValueError, match="cuadrado"):
        build_background_mesh(project, np.zeros((64, 32), dtype=np.uint16))


# ------------------------------------------------------------------- textura


def test_textura_determinista_por_seed(tmp_path):
    p1 = make_project(tmp_path, seed=123)
    p2 = make_project(tmp_path, seed=123)
    p3 = make_project(tmp_path, seed=124)
    for p in (p1, p2, p3):
        p.settings.background.procedural.texture_size = 64

    t1 = generate_background_texture(p1)
    t2 = generate_background_texture(p2)
    t3 = generate_background_texture(p3)

    assert t1.shape == (64, 64, 3)
    assert t1.dtype == np.uint8
    np.testing.assert_array_equal(t1, t2)  # misma seed → idéntica
    assert not np.array_equal(t1, t3)  # seed distinta → distinta
    # No es un color plano.
    assert t1.std() > 1.0


def test_textura_paleta(tmp_path):
    project = make_project(tmp_path)
    proc = project.settings.background.procedural
    proc.texture_size = 64
    proc.palette_low = (200, 0, 0)
    proc.palette_high = (0, 0, 200)
    tex = generate_background_texture(project)
    # Con esta paleta el canal G queda esencialmente a cero y R/B dominan.
    assert tex[..., 1].max() <= 10
    assert tex[..., 0].max() > 100
    assert tex[..., 2].max() > 100


def test_textura_por_relieve_bandas(tmp_path):
    """Verde abajo, roca en la cota media y nieve arriba."""
    project = make_project(tmp_path, size=32)
    proc = project.settings.background.procedural
    proc.texture_size = 128
    proc.band_noise = 0.0  # sin jitter: bandas exactas
    proc.band_blend = 0.0  # sin mezcla: escalón duro
    proc.slope_rock_deg = 1e6  # anula la contribución de la pendiente
    proc.snow_slope_limit_deg = 1e6
    proc.palette_low = (0, 200, 0)
    proc.palette_high = (0, 200, 0)
    proc.palette_rock = (128, 128, 128)
    proc.palette_snow = (255, 255, 255)
    proc.rock_height = 130.0
    proc.snow_height = 200.0

    # Rampa vertical 0 → 255 m (con height_scale=255, zf = 1/257).
    side = 128
    column = np.linspace(0, 65535, side).astype(np.uint16)
    dem = np.repeat(column[:, None], side, axis=1)  # varía con la FILA
    tex = generate_background_texture(project, dem).astype(int)

    metres = column * background_z_scaling_factor(project)
    # Fila claramente verde, claramente roca y claramente nieve.
    green = tex[np.argmin(np.abs(metres - 50))].mean(axis=0)
    rock = tex[np.argmin(np.abs(metres - 165))].mean(axis=0)
    snow = tex[np.argmin(np.abs(metres - 240))].mean(axis=0)

    assert green[1] > green[0] + 50 and green[1] > green[2] + 50  # verde domina
    assert abs(rock[0] - rock[1]) < 25 and abs(rock[1] - rock[2]) < 25  # gris
    assert rock.mean() < 190
    assert snow.mean() > 200  # nieve: mucho más claro que la roca
    assert snow.mean() > rock.mean() + 40


def test_textura_sin_dem_es_el_modo_plano(tmp_path):
    """Sin DEM se cae al ruido plano: nada de roca ni nieve."""
    project = make_project(tmp_path)
    proc = project.settings.background.procedural
    proc.texture_size = 64
    proc.palette_low = (200, 0, 0)
    proc.palette_high = (0, 0, 200)
    proc.palette_rock = (0, 255, 0)
    proc.palette_snow = (0, 255, 0)
    tex = generate_background_texture(project, None)
    assert tex[..., 1].max() <= 10  # el verde de roca/nieve no aparece


def test_textura_height_texture_off(tmp_path):
    """``height_texture: false`` ignora el DEM (misma salida que sin DEM)."""
    project = make_project(tmp_path)
    proc = project.settings.background.procedural
    proc.texture_size = 64
    proc.height_texture = False
    dem = synthetic_dem(64)
    np.testing.assert_array_equal(
        generate_background_texture(project, dem),
        generate_background_texture(project, None),
    )


def test_textura_flip_v(tmp_path):
    project = make_project(tmp_path)
    proc = project.settings.background.procedural
    proc.texture_size = 64
    normal = generate_background_texture(project)
    proc.texture_flip_v = True
    np.testing.assert_array_equal(generate_background_texture(project), normal[::-1])


def test_textura_alineada_con_el_mesh(tmp_path):
    """La UV de cada vértice cae sobre el píxel de textura de SU altura.

    Test de extremo a extremo de la convención de alineación documentada en
    ``mapforge.background.texture``: se construye el mesh y la textura desde el
    mismo DEM y se comprueba que el color muestreado en la UV de un vértice
    corresponde a la banda que le toca por altura. Detectaría un volteo o una
    transposición del raster.
    """
    side = 128
    # map.size pequeño: remove_center recorta poco y quedan vértices en todas
    # las zonas del DEM.
    project = make_project(tmp_path, size=16)
    proc = project.settings.background.procedural
    proc.resize_factor = 1
    proc.apply_decimation = False
    proc.texture_size = 128
    proc.band_noise = 0.0
    proc.band_blend = 0.0
    proc.slope_rock_deg = 1e6
    proc.snow_slope_limit_deg = 1e6
    proc.palette_low = (0, 0, 0)
    proc.palette_high = (0, 0, 0)
    proc.palette_rock = (255, 255, 255)
    proc.palette_snow = (255, 255, 255)
    proc.rock_height = 130.0
    proc.snow_height = 1e6  # solo dos bandas: negro abajo, blanco arriba

    # DEM asimétrico en LAS DOS direcciones, para que un volteo vertical, uno
    # horizontal o una transposición rompan el test.
    dem = np.full((side, side), 10000, dtype=np.uint16)
    dem[: side // 2, :] = 60000  # mitad norte alta
    dem[:, : side // 4] = 60000  # cuarto oeste alto

    mesh = build_background_mesh(project, dem)

    tex = generate_background_texture(project, dem)
    # Blanco = roca (alto), negro = bajo. Mapa esperado directo desde el DEM.
    zf = background_z_scaling_factor(project)
    expected_high = (dem * zf) >= proc.rock_height

    # UVs del exportador (misma fórmula que BackgroundExporter.run).
    v = np.asarray(mesh.vertices)
    lo = mesh.bounds[0][:2]
    span = np.where(mesh.extents[:2] > 0, mesh.extents[:2], 1.0)
    uv = np.clip((v[:, :2] - lo) / span, 0.0, 1.0)

    # Convención OBJ/GL: v=1 → fila 0 del array de textura.
    rows = np.clip(((1.0 - uv[:, 1]) * (proc.texture_size - 1)).round().astype(int),
                   0, proc.texture_size - 1)
    cols = np.clip((uv[:, 0] * (proc.texture_size - 1)).round().astype(int),
                   0, proc.texture_size - 1)
    sampled_white = tex[rows, cols].mean(axis=1) > 127

    # Altura real del vértice: world_y = dem × zf = (dem.max() − mesh_z_inv)…
    # el mesh guarda z = −(dem.max() − dem) × zf, luego dem×zf = z + max×zf.
    vertex_height = v[:, 2] + float(dem.max()) * zf
    expected = vertex_height >= proc.rock_height

    # Se ignoran los vértices justo en la frontera (un píxel de textura de
    # holgura por el redondeo del muestreo).
    agreement = (sampled_white == expected).mean()
    assert agreement > 0.97, f"alineación textura/mesh rota: {agreement:.3f}"
    assert expected_high.any() and not expected_high.all()


# ----------------------------------------------------------------- exportador


def test_exporter_genera_obj_textura_y_4_i3d(tmp_path):
    project = make_project(tmp_path, size=32)
    project.settings.background.remove_center = True
    proc = project.settings.background.procedural
    proc.resize_factor = 4
    proc.apply_decimation = False
    proc.texture_size = 64

    dem = synthetic_dem(64)
    mesh = build_background_mesh(project, dem)
    result = BackgroundExporter(project).run(mesh, dem)

    out = project.paths.output_dir
    assert (out / "background" / "decimated_background.obj").is_file()
    assert (out / "assets" / "background" / "background_texture.png").is_file()
    assert len(result["parts"]) == 4

    # translation Y = max_input × z_scaling_factor (regla del artefacto).
    expected_ty = float(dem.max()) * background_z_scaling_factor(project)
    assert result["translation_y"] == pytest.approx(expected_ty)

    total_faces = 0
    for index, part in enumerate(result["parts"], start=1):
        name = f"background_terrain_part_{index:02d}"
        assert part["name"] == name
        i3d_path = out / part["i3d"]
        assert i3d_path.is_file()

        # XML bien formado y con la estructura esperada.
        root = ET.parse(i3d_path).getroot()
        assert root.tag == "i3D"
        assert root.get("name") == name
        assert root.get("version") == "1.6"

        file_el = root.find("./Files/File")
        assert file_el is not None
        assert file_el.get("filename") == "background_texture.png"

        its = root.find("./Shapes/IndexedTriangleSet")
        assert its is not None
        vertices_el = its.find("Vertices")
        triangles_el = its.find("Triangles")
        subset_el = its.find("Subsets/Subset")
        assert vertices_el is not None and triangles_el is not None
        assert subset_el is not None

        v_elems = vertices_el.findall("v")
        t_elems = triangles_el.findall("t")
        assert int(vertices_el.get("count")) == len(v_elems) == part["vertices"]
        assert int(triangles_el.get("count")) == len(t_elems) == part["faces"]
        assert int(subset_el.get("numIndices")) == 3 * part["faces"]
        assert int(subset_el.get("numVertices")) == part["vertices"]
        total_faces += part["faces"]

        # UVs planas dentro de [0, 1]; normales unitarias hacia arriba (+Y en
        # la convención i3d tras el swap (x, z, −y)).
        for v in v_elems[:16]:
            u, vv = (float(c) for c in v.get("t0").split())
            assert 0.0 <= u <= 1.0
            assert 0.0 <= vv <= 1.0
            n = np.array([float(c) for c in v.get("n").split()])
            assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-4)
            assert n[1] > 0.0

        # Shape recolocado con translation Y.
        shape = root.find("./Scene/TransformGroup/Shape")
        assert shape is not None
        ty = float(shape.get("translation").split()[1])
        assert ty == pytest.approx(expected_ty)

    # El split por caras es una partición: no se pierde ninguna.
    assert total_faces == len(mesh.faces)


def test_exporter_translation_y_del_golden(tmp_path):
    """29540 × (255/65535) = 114.94163424124514 (valor exacto del artefacto)."""
    project = make_project(tmp_path, size=32, height_scale=255)
    ty = 29540.0 * background_z_scaling_factor(project)
    assert repr(ty) == "114.94163424124514"
