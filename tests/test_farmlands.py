"""Tests de la Fase 5: capas GRLE en cero y farmlands.

Cubren los casos del plan (§53 del doc madre): farmland de 2 fields
sintéticos, tope de 254 IDs, fill_empty_farmlands (0→255), margin mitre,
orden farmyards→fields y sincronización con farmlands.xml.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from mapforge.farmlands import (  # noqa: E402
    EMPTY_FARMLAND_VALUE,
    FARMLAND_ID_LIMIT,
    FARMLANDS_FILENAME,
    FarmlandsWriter,
)
from mapforge.fs25.grle_layers import create_empty_grle_layers  # noqa: E402
from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import GenerationSettings  # noqa: E402

SIZE = 128

GRLE_SCHEMA_PATH = REPO_ROOT / "config" / "grle_schema.json"


# ------------------------------------------------------------------ helpers


def make_project(
    tmp_path: Path,
    size: int = SIZE,
    farmland_margin: int = 0,
    add_farmyards: bool = True,
    fill_empty: bool = True,
    base_price: int = 12500,
) -> Project:
    paths = ProjectPaths(
        heightmap=tmp_path / "hm.png",
        osm=tmp_path / "test.osm",
        template=tmp_path / "template",
        texture_schema=tmp_path / "texture_schema.json",
        grle_schema=GRLE_SCHEMA_PATH,
        output_dir=tmp_path / "out",
    )
    params = MapParams(size=size, rotation=0, latitude=43.0, longitude=-95.0)
    settings = GenerationSettings()
    settings.grle.farmland_margin = farmland_margin
    settings.grle.add_farmyards = add_farmyards
    settings.grle.fill_empty_farmlands = fill_empty
    settings.grle.base_price = base_price
    return Project("test", params, paths, settings)


def write_textures_json(project: Project, data: dict) -> None:
    project.paths.textures_json.parent.mkdir(parents=True, exist_ok=True)
    project.paths.textures_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8"
    )


def square(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    """Polígono cuadrado cerrado (repite el primer punto, como textures.json)."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def run_writer(project: Project, data: dict) -> tuple[np.ndarray, ET.Element, dict]:
    """textures.json + capas GRLE en cero + FarmlandsWriter → (raster, xml, stats)."""
    write_textures_json(project, data)
    create_empty_grle_layers(project)
    writer = FarmlandsWriter(project)
    stats = writer.run()
    image = cv2.imread(str(writer.farmlands_png_path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    root = ET.parse(writer.farmlands_xml_path).getroot()
    return image, root, stats


# -------------------------------------------------------------- grle_layers


def test_grle_layers_created_zero_with_schema_sizes(tmp_path: Path) -> None:
    """Todos los rasters del schema se crean en cero con tamaño
    map_size × multiplier y el dtype/canales declarados."""
    project = make_project(tmp_path, size=64)
    created = create_empty_grle_layers(project)

    schema = json.loads(GRLE_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert created == [entry["name"] for entry in schema]
    assert len(created) == 21

    for entry in schema:
        path = project.paths.map_data_dir / entry["name"]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert img is not None, entry["name"]
        expected_h = int(64 * entry["height_multiplier"])
        expected_w = int(64 * entry["width_multiplier"])
        assert img.shape[:2] == (expected_h, expected_w), entry["name"]
        channels = img.shape[2] if img.ndim == 3 else 1
        assert channels == entry["channels"], entry["name"]
        assert img.dtype == np.dtype(entry["data_type"]), entry["name"]
        assert not img.any(), entry["name"]


def test_grle_farmlands_layer_is_half_resolution(tmp_path: Path) -> None:
    project = make_project(tmp_path, size=SIZE)
    create_empty_grle_layers(project)
    img = cv2.imread(
        str(project.paths.map_data_dir / FARMLANDS_FILENAME), cv2.IMREAD_UNCHANGED
    )
    assert img.shape == (SIZE // 2, SIZE // 2)
    assert img.dtype == np.uint8


# ---------------------------------------------------------------- farmlands


def test_two_fields_get_sequential_ids(tmp_path: Path) -> None:
    """§53: dos fields sintéticos → IDs 1 y 2 en el raster ÷2, resto 255,
    y farmlands.xml con 2 entradas sincronizadas."""
    project = make_project(tmp_path, farmland_margin=0)
    data = {
        "fields": [square(20, 20, 50, 50), square(70, 70, 110, 110)],
    }
    image, root, stats = run_writer(project, data)

    assert stats == {"added": 2, "skipped": 0}
    # Coordenadas ÷2: el centro de cada cuadrado cae dentro de su ID.
    assert image[17, 17] == 1  # (35, 35) ÷ 2
    assert image[45, 45] == 2  # (90, 90) ÷ 2
    # Fuera de los fields: relleno 255, nunca 0.
    assert image[5, 5] == EMPTY_FARMLAND_VALUE
    assert not (image == 0).any()
    assert sorted(np.unique(image).tolist()) == [1, 2, EMPTY_FARMLAND_VALUE]

    farmlands_node = root.find("farmlands")
    assert farmlands_node is not None
    assert farmlands_node.get("pricePerHa") == "12500"
    entries = farmlands_node.findall("farmland")
    assert [e.get("id") for e in entries] == ["1", "2"]
    assert {e.get("priceScale") for e in entries} == {"1"}
    assert {e.get("npcName") for e in entries} == {"FORESTER"}


def test_farmland_id_limit_254(tmp_path: Path) -> None:
    """Tope 254: con 300 fields solo se dibujan/declaran los primeros 254."""
    project = make_project(tmp_path, size=256, farmland_margin=0, fill_empty=False)
    fields = []
    for i in range(300):
        x = (i % 20) * 12 + 2
        y = (i // 20) * 12 + 2
        fields.append(square(x, y, x + 6, y + 6))
    image, root, stats = run_writer(project, {"fields": fields})

    assert stats["added"] == FARMLAND_ID_LIMIT == 254
    entries = root.find("farmlands").findall("farmland")
    assert len(entries) == 254
    assert entries[-1].get("id") == "254"

    values = np.unique(image)
    assert values.max() == 254  # sin fill_empty no aparece el 255
    # Todos los IDs 1..254 presentes, ninguno más.
    assert set(values.tolist()) == {0, *range(1, 255)}


def test_fill_empty_farmlands_zero_to_255(tmp_path: Path) -> None:
    """fill_empty_farmlands: los 0 restantes pasan a 255; apagado, quedan 0."""
    data = {"fields": [square(40, 40, 80, 80)]}

    filled_project = make_project(tmp_path / "on", fill_empty=True)
    image_on, _, _ = run_writer(filled_project, data)
    assert not (image_on == 0).any()
    assert (image_on == EMPTY_FARMLAND_VALUE).any()

    raw_project = make_project(tmp_path / "off", fill_empty=False)
    image_off, _, _ = run_writer(raw_project, data)
    assert (image_off == 0).any()
    assert not (image_off == EMPTY_FARMLAND_VALUE).any()


def test_farmland_margin_mitre_expands_polygon(tmp_path: Path) -> None:
    """farmland_margin: buffer mitre en px (1 px = 1 m) ANTES de ÷2."""
    data = {"fields": [square(40, 40, 80, 80)]}

    no_margin, _, _ = run_writer(make_project(tmp_path / "m0", farmland_margin=0), data)
    margin, _, _ = run_writer(make_project(tmp_path / "m8", farmland_margin=8), data)

    # Sin margin: (40..80) ÷ 2 = filas/cols 20..40; fuera queda 255.
    assert no_margin[18, 30] == EMPTY_FARMLAND_VALUE
    assert no_margin[20, 30] == 1
    # Con margin 8: el cuadrado crece a (32..88) ÷ 2 = 16..44.
    assert margin[16, 30] == 1
    assert margin[44, 30] == 1
    assert margin[14, 30] == EMPTY_FARMLAND_VALUE
    # El área con ID crece estrictamente.
    assert (margin == 1).sum() > (no_margin == 1).sum()


def test_farmyards_get_ids_before_fields(tmp_path: Path) -> None:
    """Orden 1.8: farmyards primero (si add_farmyards), luego fields."""
    data = {
        "farmyards": [square(10, 10, 30, 30)],
        "fields": [square(60, 60, 100, 100)],
    }
    image, root, stats = run_writer(make_project(tmp_path / "yes", add_farmyards=True), data)
    assert stats["added"] == 2
    assert image[10, 10] == 1  # farmyard (20,20)÷2 → ID 1
    assert image[40, 40] == 2  # field (80,80)÷2 → ID 2

    # add_farmyards=False: el farmyard no participa y el field es el ID 1.
    image2, root2, stats2 = run_writer(
        make_project(tmp_path / "no", add_farmyards=False), data
    )
    assert stats2["added"] == 1
    assert image2[40, 40] == 1
    assert image2[10, 10] == EMPTY_FARMLAND_VALUE
    assert len(root2.find("farmlands").findall("farmland")) == 1


def test_out_of_bounds_polygon_skipped_without_consuming_id(tmp_path: Path) -> None:
    """Un polígono fuera del mapa se salta sin consumir ID (paridad 1.8)."""
    data = {
        "fields": [
            square(500, 500, 600, 600),  # fuera de un mapa de 128
            square(40, 40, 80, 80),
        ],
    }
    image, root, stats = run_writer(make_project(tmp_path, farmland_margin=0), data)
    assert stats == {"added": 1, "skipped": 1}
    assert image[30, 30] == 1
    entries = root.find("farmlands").findall("farmland")
    assert [e.get("id") for e in entries] == ["1"]


def test_price_per_ha_and_regeneration(tmp_path: Path) -> None:
    """pricePerHa = base_price; una segunda ejecución no duplica entradas."""
    project = make_project(tmp_path, base_price=99999)
    data = {"fields": [square(40, 40, 80, 80)]}
    _, root, _ = run_writer(project, data)
    node = root.find("farmlands")
    assert node.get("pricePerHa") == "99999"
    assert node.get("infoLayer") == "farmlands"
    assert len(node.findall("farmland")) == 1

    # Re-ejecutar sobre el mismo output: el XML se regenera, no se acumula.
    create_empty_grle_layers(project)
    FarmlandsWriter(project).run()
    root2 = ET.parse(FarmlandsWriter(project).farmlands_xml_path).getroot()
    assert len(root2.find("farmlands").findall("farmland")) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
