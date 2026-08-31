"""Tests de la Fase 1 (terrain/dem.py): pipeline DEM S2.

Cubre los dos casos del §53 del doc madre (DEM plano y DEM gradiente) más el
modo custom_dem, el hook de agua, el blur, el clamp de spline_z y los errores
de entrada. Cada test aísla una sola variable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import DEMSettings, GenerationSettings  # noqa: E402
from mapforge.terrain.dem import DemPipeline, spline_z  # noqa: E402

MAP_SIZE = 16  # background_size = 16 + 4096 = 4112


def make_project(
    tmp_path: Path,
    heightmap: np.ndarray,
    map_size: int = MAP_SIZE,
    dem_settings: DEMSettings | None = None,
    rotation: float = 0,
) -> Project:
    """Proyecto mínimo con un heightmap sintético escrito en tmp_path."""
    heightmap_path = tmp_path / "heightmap.png"
    cv2.imwrite(str(heightmap_path), heightmap)

    settings = GenerationSettings()
    if dem_settings is not None:
        settings.dem = dem_settings

    paths = ProjectPaths(
        heightmap=heightmap_path,
        osm=tmp_path / "custom.osm",
        template=tmp_path / "template.zip",
        texture_schema=tmp_path / "texture_schema.json",
        grle_schema=tmp_path / "grle_schema.json",
        output_dir=tmp_path / "salida",
    )
    return Project(
        name="test",
        map_params=MapParams(size=map_size, rotation=rotation),
        paths=paths,
        settings=settings,
    )


def background_size(map_size: int = MAP_SIZE) -> int:
    return map_size + 2 * 2048


# ------------------------------------------------------------- DEM plano


def test_dem_plano(tmp_path: Path) -> None:
    """§53: DEM constante → shift a plateau+water_depth y salida constante."""
    bg = background_size()
    flat = np.full((bg, bg), 1000, dtype=np.uint16)
    project = make_project(tmp_path, flat)  # settings del golden (defaults)

    pipeline = DemPipeline(project)
    height_scale = pipeline.run()

    # height_scale: max tras el shift = 30 m; ceil(max(255, 30+15)) = 255.
    assert height_scale == 255
    assert project.height_scale == 255
    assert pipeline.mesh_z_scaling_factor == pytest.approx(65535 / 255)
    assert pipeline.height_scale_multiplier == pytest.approx(1.0)

    # Salida constante: el blur de una constante es la constante.
    full = pipeline.dem_full
    assert full is not None
    assert full.shape == (bg, bg) and full.dtype == np.uint16
    assert full.min() == full.max()
    # min = plateau + water_depth = 30 m → 30 × 257 = 7710 (±1 por truncado).
    assert abs(int(full.min()) - 7710) <= 1

    # Crop y resize conservan la constante y las formas.
    assert pipeline.dem_not_resized is not None
    assert pipeline.dem_not_resized.shape == (MAP_SIZE, MAP_SIZE)
    assert pipeline.dem_not_resized.min() == full.min()
    assert pipeline.dem_map is not None
    assert pipeline.dem_map.shape == (MAP_SIZE + 1, MAP_SIZE + 1)
    assert pipeline.dem_map.min() == pipeline.dem_map.max() == full.min()

    # Ficheros en disco.
    out = project.paths.output_dir
    for rel in (
        "background/FULL.png",
        "background/not_substracted.png",
        "background/not_resized.png",
        "map/data/dem.png",
    ):
        assert (out / rel).is_file(), f"falta {rel}"
    dem_png = cv2.imread(str(project.paths.dem_png), cv2.IMREAD_UNCHANGED)
    assert dem_png.dtype == np.uint16
    assert dem_png.shape == (MAP_SIZE + 1, MAP_SIZE + 1)
    np.testing.assert_array_equal(dem_png, pipeline.dem_map)

    # dem_info.json expone el height_scale para el escritor i3d.
    info = json.loads((out / "dem_info.json").read_text(encoding="utf-8"))
    assert info["mode"] == "raw"
    assert info["height_scale"]["adjusted_height_scale"] == 255
    assert info["height_scale"]["mesh_z_scaling_factor"] == pytest.approx(257.0)

    # spline_z de un DEM plano = altura constante en metros (~30 m).
    assert pipeline.spline_z(0, 0) == pytest.approx(30.0, abs=0.01)


# --------------------------------------------------------- DEM gradiente


def test_dem_gradiente_sin_ajuste(tmp_path: Path) -> None:
    """§53: gradiente horizontal sin shift ni blur → normalización exacta."""
    bg = background_size()
    ramp = np.tile(
        np.linspace(0, 25700, bg, dtype=np.uint16), (bg, 1)
    )  # 0..100 m de oeste a este
    dem_settings = DEMSettings(
        adjust_terrain_to_ground_level=False, blur_radius=0, water_depth=0, plateau=0
    )
    project = make_project(tmp_path, ramp, dem_settings=dem_settings)

    pipeline = DemPipeline(project)
    height_scale = pipeline.run()

    # max = 100 m; ceil(max(255, 100+15)) = 255.
    assert height_scale == 255

    full = pipeline.dem_full
    assert full is not None
    # Round-trip v → metros → normalizado: idéntico ±1 (truncado de astype).
    delta = full.astype(np.int64) - ramp.astype(np.int64)
    assert int(np.abs(delta).max()) <= 1
    # Monótono no decreciente a lo largo de X.
    assert bool(np.all(np.diff(full[0, :].astype(np.int64)) >= 0))
    # El crop central coincide con la zona central del gradiente.
    center0 = (bg - MAP_SIZE) // 2
    np.testing.assert_array_equal(
        pipeline.dem_not_resized, full[center0 : center0 + MAP_SIZE, center0 : center0 + MAP_SIZE]
    )
    # El resize INTER_LINEAR de un gradiente sigue siendo monótono.
    assert bool(np.all(np.diff(pipeline.dem_map[0, :].astype(np.int64)) >= 0))


def test_dem_gradiente_con_ajuste_y_ceiling(tmp_path: Path) -> None:
    """Gradiente que fuerza el ceiling: height_scale > minimum_height_scale."""
    bg = background_size()
    ramp = np.tile(np.linspace(0, 65535, bg, dtype=np.uint16), (bg, 1))  # 0..255 m
    dem_settings = DEMSettings(
        adjust_terrain_to_ground_level=True,
        plateau=15,
        water_depth=15,
        ceiling=15,
        blur_radius=0,
    )
    project = make_project(tmp_path, ramp, dem_settings=dem_settings)

    pipeline = DemPipeline(project)
    height_scale = pipeline.run()

    # min → 30 m, max → 285 m; ceil(max(255, 285+15)) = 300.
    assert height_scale == 300
    full = pipeline.dem_full
    assert full is not None
    # min = 30/300 × 65535 = 6553.5 → 6553; max = 285/300 × 65535 = 62258.25.
    assert abs(int(full.min()) - 6553) <= 1
    assert abs(int(full.max()) - 62258) <= 1

    # spline_z usa height_scale/65535: el mínimo del crop en metros.
    z = pipeline.spline_z(0, 0)
    expected = float(pipeline.dem_not_resized[0, 0]) * 300 / 65535
    assert z == pytest.approx(expected)


def test_multiplier_escala_y_spline_z_lo_deshace(tmp_path: Path) -> None:
    """multiplier=2 dobla los metros; spline_z divide por multiplier."""
    bg = background_size()
    flat = np.full((bg, bg), 2570, dtype=np.uint16)  # 10 m
    dem_settings = DEMSettings(
        adjust_terrain_to_ground_level=False,
        multiplier=2,
        blur_radius=0,
        plateau=0,
        water_depth=0,
    )
    project = make_project(tmp_path, flat, dem_settings=dem_settings)

    pipeline = DemPipeline(project)
    assert pipeline.run() == 255  # 20 m + 15 < 255

    # 20 m → 20 × 257 = 5140 en uint16.
    assert abs(int(pipeline.dem_full.min()) - 5140) <= 1
    # spline_z: 5140 × (1/2) × (255/65535) ≈ 10 m (la altura original).
    assert pipeline.spline_z(0, 0) == pytest.approx(10.0, abs=0.01)


# ------------------------------------------------------------------ blur


def test_blur_radius_par_se_fuerza_a_impar(tmp_path: Path) -> None:
    """blur_radius par r equivale a r+1 (regla de Maps4FS)."""
    assert DemPipeline._effective_blur_radius(0) == 0
    assert DemPipeline._effective_blur_radius(-3) == 0
    assert DemPipeline._effective_blur_radius(None) == 0
    assert DemPipeline._effective_blur_radius(3) == 3
    assert DemPipeline._effective_blur_radius(4) == 5

    bg = background_size()
    rng = np.random.default_rng(7)
    noisy = rng.integers(0, 30000, (bg, bg), dtype=np.uint16)

    outputs = []
    for radius in (4, 5):
        tmp = tmp_path / f"r{radius}"
        tmp.mkdir()
        project = make_project(
            tmp, noisy, dem_settings=DEMSettings(blur_radius=radius)
        )
        pipeline = DemPipeline(project)
        pipeline.run()
        outputs.append(pipeline.dem_full)
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_blur_suaviza_tras_normalizar(tmp_path: Path) -> None:
    """El blur actúa sobre el uint16 normalizado (reduce el rango local)."""
    bg = background_size()
    checker = np.zeros((bg, bg), dtype=np.uint16)
    checker[::2, :] = 25700  # franjas 0/100 m
    dem_settings = DEMSettings(
        adjust_terrain_to_ground_level=False,
        blur_radius=3,
        plateau=0,
        water_depth=0,
    )
    project = make_project(tmp_path, checker, dem_settings=dem_settings)
    pipeline = DemPipeline(project)
    pipeline.run()

    expected = cv2.GaussianBlur(
        np.clip(
            (checker.astype(np.float64) * project.settings.input_height_scale / 255)
            * 65535,
            0,
            65535,
        ).astype(np.uint16),
        (3, 3),
        sigmaX=10,
        sigmaY=10,
    )
    np.testing.assert_array_equal(pipeline.dem_full, expected)


# ------------------------------------------------------------ custom_dem


def test_custom_dem_passthrough(tmp_path: Path) -> None:
    """Modo custom_dem: FULL = input, crop y resize sin reprocesado."""
    bg = background_size()
    rng = np.random.default_rng(42)
    data = rng.integers(2000, 29540, (bg, bg), dtype=np.uint16)
    project = make_project(tmp_path, data)  # settings del golden

    pipeline = DemPipeline(project, custom_dem=True)
    height_scale = pipeline.run()

    # Fórmula del golden: ceil(max(255, max×255/65535 + 15)) = 255.
    assert height_scale == 255
    np.testing.assert_array_equal(pipeline.dem_full, data)

    center0 = (bg - MAP_SIZE) // 2
    expected_crop = data[center0 : center0 + MAP_SIZE, center0 : center0 + MAP_SIZE]
    np.testing.assert_array_equal(pipeline.dem_not_resized, expected_crop)

    expected_dem = cv2.resize(
        expected_crop, (MAP_SIZE + 1, MAP_SIZE + 1), interpolation=cv2.INTER_LINEAR
    )
    np.testing.assert_array_equal(pipeline.dem_map, expected_dem)

    info = json.loads(
        (project.paths.output_dir / "dem_info.json").read_text(encoding="utf-8")
    )
    assert info["mode"] == "custom_dem"


def test_custom_dem_exige_background_size(tmp_path: Path) -> None:
    wrong = np.full((100, 100), 1000, dtype=np.uint16)
    project = make_project(tmp_path, wrong)
    with pytest.raises(ValueError, match="background_size"):
        DemPipeline(project, custom_dem=True).run()


# ------------------------------------------------------------ hook agua


def test_water_mask_resta_bajo_mascara(tmp_path: Path) -> None:
    """El hook de agua resta water_depth×257 solo bajo la máscara erosionada."""
    bg = background_size()
    flat = np.full((bg, bg), 20000, dtype=np.uint16)
    project = make_project(tmp_path, flat)  # water_depth=15 (golden)

    mask = np.zeros((bg, bg), dtype=np.uint8)
    c = bg // 2
    mask[c - 4 : c + 4, c - 4 : c + 4] = 255  # 8×8 de agua en el centro

    pipeline = DemPipeline(project, custom_dem=True)
    pipeline.run(water_mask=mask)

    subtract_by = int(15 * 65535 / 255)  # 3855
    eroded = cv2.erode(
        (mask == 255).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    assert eroded.sum() == 36  # 6×6 tras la erosión

    full = pipeline.dem_full
    assert int(full[c, c]) == 20000 - subtract_by
    assert (full == 20000 - subtract_by).sum() == 36
    # not_resized y not_substracted quedan SIN resta (orden de Maps4FS).
    assert int(pipeline.dem_not_resized[MAP_SIZE // 2, MAP_SIZE // 2]) == 20000
    not_sub = cv2.imread(
        str(project.paths.background_dir / "not_substracted.png"), cv2.IMREAD_UNCHANGED
    )
    assert int(not_sub.max()) == 20000 and int(not_sub.min()) == 20000
    # dem.png sí hereda la resta (se recorta de FULL post-resta).
    center_dem = pipeline.dem_map[MAP_SIZE // 2, MAP_SIZE // 2]
    assert int(center_dem) == 20000 - subtract_by


def test_sin_water_mask_no_resta(tmp_path: Path) -> None:
    bg = background_size()
    flat = np.full((bg, bg), 20000, dtype=np.uint16)
    project = make_project(tmp_path, flat)
    pipeline = DemPipeline(project, custom_dem=True)
    pipeline.run()
    np.testing.assert_array_equal(pipeline.dem_full, flat)


# -------------------------------------------------------------- spline_z


def test_spline_z_clamp_a_bordes() -> None:
    dem = np.array([[100, 200], [300, 40000]], dtype=np.uint16)
    # Dentro de rango: dem[y=1, x=0] = 300 → 300 × 255/65535.
    assert spline_z(dem, 0, 1, 1, 255) == pytest.approx(300 * 255 / 65535)
    # Fuera de rango: clamp a los bordes.
    assert spline_z(dem, -5, -5, 1, 255) == pytest.approx(100 * 255 / 65535)
    assert spline_z(dem, 10, 10, 1, 255) == pytest.approx(40000 * 255 / 65535)
    # Coordenadas float se truncan (int()).
    assert spline_z(dem, 1.9, 0.9, 1, 255) == pytest.approx(200 * 255 / 65535)
    # multiplier ≠ 1 divide.
    assert spline_z(dem, 0, 0, 2, 255) == pytest.approx(100 * 255 / 65535 / 2)


def test_spline_z_requiere_run(tmp_path: Path) -> None:
    bg = background_size()
    project = make_project(tmp_path, np.full((bg, bg), 1, dtype=np.uint16))
    pipeline = DemPipeline(project)
    with pytest.raises(RuntimeError):
        pipeline.spline_z(0, 0)


# --------------------------------------------------------------- errores


def test_heightmap_multicanal_rechazado(tmp_path: Path) -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    project = make_project(tmp_path, rgb)
    with pytest.raises(ValueError, match="1 canal|uint16"):
        DemPipeline(project).run()


def test_heightmap_todo_cero_rechazado(tmp_path: Path) -> None:
    bg = background_size()
    project = make_project(tmp_path, np.zeros((bg, bg), dtype=np.uint16))
    with pytest.raises(ValueError, match="cero"):
        DemPipeline(project).run()


def test_rotacion_no_soportada(tmp_path: Path) -> None:
    bg = background_size()
    project = make_project(
        tmp_path, np.full((bg, bg), 1000, dtype=np.uint16), rotation=25
    )
    with pytest.raises(ValueError, match="rotation"):
        DemPipeline(project).run()


def test_heightmap_pequeno_se_redimensiona_en_modo_raw(tmp_path: Path) -> None:
    """En modo raw un heightmap de otro tamaño se reescala a background_size."""
    small = np.full((512, 512), 1000, dtype=np.uint16)
    project = make_project(tmp_path, small)
    pipeline = DemPipeline(project)
    pipeline.run()
    bg = background_size()
    assert pipeline.dem_full.shape == (bg, bg)
    assert pipeline.dem_map.shape == (MAP_SIZE + 1, MAP_SIZE + 1)
