"""Tests de la hierba base (``GRLE._add_plants`` de Maps4FS 1.8.242).

Cubren: el valor de planta va al canal R del ``densityMap_fruits.png`` (FACT
del artefacto golden), la máscara es la capa ``usage == "grass"`` fusionada con
la de ``usage == "forest"``, la erosión 3×3 + marco de 1 px, ``base_grass``
desconocido → meadow, ``add_grass: false`` y las islas de ``random_plants``
(deterministas con el seed del proyecto).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from mapforge.fs25.grle_layers import create_empty_grle_layers  # noqa: E402
from mapforge.fs25.plants import (  # noqa: E402
    DENSITY_MAP_FRUITS,
    PLANT_PIXEL_VALUES,
    PlantsWriter,
    plant_to_pixel_value,
)
from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import GenerationSettings  # noqa: E402

SIZE = 128

GRLE_SCHEMA_PATH = REPO_ROOT / "config" / "grle_schema.json"

#: Schema mínimo con una capa de hierba (base) y una de bosque.
SCHEMA = [
    {"name": "forestGrass", "count": 1, "priority": 5, "usage": "forest"},
    {"name": "grass", "count": 1, "priority": 0, "usage": "grass"},
]


# ------------------------------------------------------------------ helpers


def make_project(tmp_path: Path, schema: list[dict] | None = None, **grle) -> Project:
    tmp_path.mkdir(parents=True, exist_ok=True)
    schema_path = tmp_path / "texture_schema.json"
    schema_path.write_text(json.dumps(schema if schema is not None else SCHEMA), encoding="utf-8")
    paths = ProjectPaths(
        heightmap=tmp_path / "hm.png",
        osm=tmp_path / "test.osm",
        template=tmp_path / "template",
        texture_schema=schema_path,
        grle_schema=GRLE_SCHEMA_PATH,
        output_dir=tmp_path / "out",
    )
    params = MapParams(size=SIZE, rotation=0, latitude=43.0, longitude=-95.0)
    settings = GenerationSettings()
    for key, value in grle.items():
        setattr(settings.grle, key, value)
    project = Project("test", params, paths, settings)
    create_empty_grle_layers(project)
    return project


def write_weight(project: Project, filename: str, mask: np.ndarray) -> None:
    cv2.imwrite(str(project.paths.map_data_dir / filename), mask)


def square_mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def read_fruits(project: Project) -> np.ndarray:
    """densityMap_fruits.png como array BGR de OpenCV (canal 2 = R del PNG)."""
    img = cv2.imread(str(project.paths.map_data_dir / DENSITY_MAP_FRUITS), cv2.IMREAD_UNCHANGED)
    assert img is not None
    return img


# -------------------------------------------------------------------- tests


def test_plant_to_pixel_value() -> None:
    assert plant_to_pixel_value("meadow") == 131
    assert plant_to_pixel_value("smallDenseMix") == 33
    assert plant_to_pixel_value("noExiste") is None
    assert PLANT_PIXEL_VALUES["meadow"] == 131


def test_grass_written_to_red_channel(tmp_path: Path) -> None:
    """El valor de planta va al canal R del PNG (FACT del artefacto golden) y
    la máscara se reescala ×2 respecto al weight."""
    project = make_project(tmp_path)
    write_weight(project, "grass01_weight.png", square_mask(20, 20, 100, 100))

    result = PlantsWriter(project).run()
    assert result["status"] == "ok"
    assert result["plant_value"] == 131
    assert result["grass_layer"] == "grass"

    fruits = read_fruits(project)
    assert fruits.shape == (SIZE * 2, SIZE * 2, 3)
    # Canal 2 del array de cv2 == canal R del fichero.
    assert fruits[:, :, 2].max() == 131
    assert not fruits[:, :, 0].any() and not fruits[:, :, 1].any()
    # Centro del cuadrado (×2) pintado; fuera del cuadrado, cero.
    assert fruits[120, 120, 2] == 131
    assert fruits[10, 10, 2] == 0
    # La máscara erosionada recorta 1 px del borde del cuadrado (×2 → y=40).
    assert fruits[40, 120, 2] == 0
    assert fruits[42, 120, 2] == 131


def test_forest_merged_into_grass(tmp_path: Path) -> None:
    """La capa usage='forest' se fusiona con la máscara de hierba."""
    project = make_project(tmp_path)
    write_weight(project, "grass01_weight.png", square_mask(10, 10, 40, 40))
    write_weight(project, "forestGrass01_weight.png", square_mask(60, 60, 100, 100))

    result = PlantsWriter(project).run()
    assert result["forest_layer"] == "forestGrass"

    fruits = read_fruits(project)[:, :, 2]
    assert fruits[50, 50] == 131  # dentro de la hierba
    assert fruits[160, 160] == 131  # dentro del bosque
    assert fruits[110, 110] == 0  # entre ambos


def test_edge_pixels_removed(tmp_path: Path) -> None:
    """Marco de 1 px a cero aunque la hierba llegue al borde del mapa."""
    project = make_project(tmp_path)
    write_weight(project, "grass01_weight.png", np.full((SIZE, SIZE), 255, np.uint8))

    PlantsWriter(project).run()
    fruits = read_fruits(project)[:, :, 2]
    assert not fruits[0, :].any() and not fruits[-1, :].any()
    assert not fruits[:, 0].any() and not fruits[:, -1].any()
    assert fruits[100, 100] == 131


def test_unknown_base_grass_falls_back_to_meadow(tmp_path: Path) -> None:
    project = make_project(tmp_path, base_grass="noExiste")
    write_weight(project, "grass01_weight.png", square_mask(20, 20, 100, 100))
    assert PlantsWriter(project).run()["plant_value"] == 131


def test_small_dense_mix(tmp_path: Path) -> None:
    project = make_project(tmp_path, base_grass="smallDenseMix")
    write_weight(project, "grass01_weight.png", square_mask(20, 20, 100, 100))
    assert PlantsWriter(project).run()["plant_value"] == 33
    assert read_fruits(project)[:, :, 2].max() == 33


def test_add_grass_disabled(tmp_path: Path) -> None:
    project = make_project(tmp_path, add_grass=False)
    write_weight(project, "grass01_weight.png", square_mask(20, 20, 100, 100))
    assert PlantsWriter(project).run()["status"] == "skipped"
    assert not read_fruits(project).any()


def test_missing_grass_layer_is_skipped(tmp_path: Path) -> None:
    """Sin capa usage='grass' en el schema no se pinta nada (no revienta)."""
    project = make_project(tmp_path, schema=[{"name": "grass", "count": 1, "priority": 0}])
    result = PlantsWriter(project).run()
    assert result["status"] == "skipped"
    assert result["reason"] == "sin capa usage=grass"


def test_missing_weight_file_is_skipped(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    result = PlantsWriter(project).run()
    assert result["status"] == "skipped"
    assert not read_fruits(project).any()


def test_dissolve_preview_is_used(tmp_path: Path) -> None:
    """Con dissolve, la máscara sale del ``*_preview.png`` (máscara original)."""
    project = make_project(tmp_path)
    write_weight(project, "grass01_weight.png", square_mask(20, 20, 40, 40))
    write_weight(project, "grass01_weight_preview.png", square_mask(20, 20, 100, 100))

    PlantsWriter(project).run()
    fruits = read_fruits(project)[:, :, 2]
    assert fruits[150, 150] == 131  # solo está en el preview


def test_random_plants_islands_are_deterministic(tmp_path: Path) -> None:
    """Las islas usan el RNG sembrado: mismo seed → mismo resultado, y nunca
    salen de la máscara de hierba."""
    results = []
    for run in range(2):
        project = make_project(
            tmp_path / f"run{run}",
            random_plants=True,
            plants_island_minimum_size=10,
            plants_island_maximum_size=30,
            plants_island_percent=100,
        )
        write_weight(project, "grass01_weight.png", square_mask(20, 20, 100, 100))
        stats = PlantsWriter(project).run()
        assert stats["islands"] == SIZE  # map_size × 100 // 100
        results.append(read_fruits(project)[:, :, 2])

    assert np.array_equal(results[0], results[1])
    values = set(np.unique(results[0]).tolist())
    assert values - {0, 131}, "random_plants no pintó ninguna isla"
    # Nada fuera de la máscara de hierba erosionada (cuadrado ×2 con 1 px menos).
    assert not results[0][:41, :].any()
    assert not results[0][201:, :].any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
