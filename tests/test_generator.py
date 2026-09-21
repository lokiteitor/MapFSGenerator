"""Tests de la Fase 9: orquestador del pipeline completo (generator.py).

Mini-E2E con inputs sintéticos pequeños (mapa 128², template FS25 real,
schemas reales): el pipeline completo debe terminar sin excepción, producir
la estructura completa del mod y dejar telemetría en ``generation_info.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from mapforge.generator import STAGE_ORDER, Generator  # noqa: E402
from mapforge.osm.projection import MapProjection  # noqa: E402
from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import GenerationSettings  # noqa: E402

TEMPLATE_ZIP = REPO_ROOT / "templates" / "fs25-map-template.zip"
TEXTURE_SCHEMA = REPO_ROOT / "config" / "texture_schema.json"
GRLE_SCHEMA = REPO_ROOT / "config" / "grle_schema.json"

LAT, LON = 43.0, -95.0
SIZE = 128


# ------------------------------------------------------------------ helpers


def _pixel_to_latlon(
    x: float, y: float, proj: MapProjection
) -> tuple[float, float]:
    north, south, east, west = proj.bbox
    lon = west + ((x + 0.5) / proj.image_size) * (east - west)
    lat = north + ((y + 0.5) / proj.image_size) * (south - north)
    return lat, lon


def _write_osm(path: Path, ways: list[dict], proj: MapProjection) -> None:
    """OSM sintético: cada way = {points: [(x,y)px], tags: {}, closed: bool}."""
    node_id = 1
    parts: list[str] = [
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<osm version="0.6" generator="test">\n'
    ]
    way_parts: list[str] = []
    for way_index, way in enumerate(ways, start=1):
        refs = []
        for x, y in way["points"]:
            lat, lon = _pixel_to_latlon(x, y, proj)
            parts.append(f'  <node id="{node_id}" lat="{lat!r}" lon="{lon!r}" />\n')
            refs.append(node_id)
            node_id += 1
        if way.get("closed"):
            refs.append(refs[0])
        way_parts.append(f'  <way id="{way_index}">\n')
        for ref in refs:
            way_parts.append(f'    <nd ref="{ref}" />\n')
        for key, value in way.get("tags", {}).items():
            way_parts.append(f'    <tag k="{key}" v="{value}" />\n')
        way_parts.append("  </way>\n")
    parts.extend(way_parts)
    parts.append("</osm>\n")
    path.write_text("".join(parts), encoding="utf-8")


def make_mini_project(tmp_path: Path, **settings_overrides) -> Project:
    """Proyecto 128² con heightmap gradiente y OSM (field+farmyard+carretera).

    Los settings de background se rebajan (resize_factor 32, textura 256²)
    para que el mini-E2E corra en ~1 s.
    """
    proj_px = MapProjection(LAT, LON, SIZE)

    # Heightmap gradiente 256² uint16 (modo raw lo redimensiona a background²).
    row = np.linspace(0, 20000, 256, dtype=np.uint16)
    heightmap = np.tile(row, (256, 1))
    cv2.imwrite(str(tmp_path / "hm.png"), heightmap)

    _write_osm(
        tmp_path / "in.osm",
        [
            {
                "points": [(20, 20), (60, 20), (60, 60), (20, 60)],
                "closed": True,
                "tags": {"landuse": "farmland"},
            },
            {
                "points": [(70, 70), (100, 70), (100, 100), (70, 100)],
                "closed": True,
                "tags": {"landuse": "farmyard"},
            },
            {"points": [(10, 90), (120, 90)], "tags": {"highway": "secondary"}},
        ],
        proj_px,
    )

    paths = ProjectPaths(
        heightmap=tmp_path / "hm.png",
        osm=tmp_path / "in.osm",
        template=TEMPLATE_ZIP,
        texture_schema=TEXTURE_SCHEMA,
        grle_schema=GRLE_SCHEMA,
        output_dir=tmp_path / "out",
    )
    params = MapParams(size=SIZE, rotation=0, latitude=LAT, longitude=LON)
    settings = GenerationSettings()
    settings.background.procedural.resize_factor = 32
    settings.background.procedural.texture_size = 256
    settings.background.procedural.noise_octaves = 3
    for key, value in settings_overrides.items():
        group, _, name = key.partition("__")
        setattr(getattr(settings, group), name, value)
    return Project("MiniE2E", params, paths, settings)


# --------------------------------------------------------------------- E2E


@pytest.fixture(scope="module")
def mini_run(tmp_path_factory: pytest.TempPathFactory):
    """Corre el pipeline completo una vez y comparte el resultado."""
    tmp_path = tmp_path_factory.mktemp("mini_e2e")
    project = make_mini_project(tmp_path)
    info = Generator(project).run()
    return project, info


def test_pipeline_produces_complete_structure(mini_run) -> None:
    project, _ = mini_run
    out = project.paths.output_dir
    expected = [
        "map/map.i3d",
        "map/map.xml",
        "modDesc.xml",
        "map/data/dem.png",
        "map/data/infoLayer_farmlands.png",
        "map/config/farmlands.xml",
        "map/splines.i3d",
        "map/data/masks/BLOCKMASK.png",
        "background/FULL.png",
        "background/not_resized.png",
        "background/not_substracted.png",
        "background/decimated_background.obj",
        "info_layers/textures.json",
        "assets/background/background_texture.png",
        "generation_info.json",
        "dem_info.json",
    ]
    for rel in expected:
        assert (out / rel).is_file(), f"falta {rel}"

    parts = sorted((out / "assets" / "background").glob("background_terrain_part_*.i3d"))
    assert len(parts) == 4

    # dem.png con el tamaño FS25: (map_size+1)² uint16.
    dem = cv2.imread(str(out / "map/data/dem.png"), cv2.IMREAD_UNCHANGED)
    assert dem.shape == (SIZE + 1, SIZE + 1)
    assert dem.dtype == np.uint16

    # Al menos un weight dibujado (la carretera secondary → gravelSmall).
    weight = cv2.imread(
        str(out / "map/data/gravelSmall01_weight.png"), cv2.IMREAD_UNCHANGED
    )
    assert weight is not None and weight.shape == (SIZE, SIZE)
    assert np.any(weight == 255)


def test_pipeline_components_integrate_in_i3d(mini_run) -> None:
    project, _ = mini_run
    out = project.paths.output_dir

    root = ET.parse(out / "map/map.i3d").getroot()
    # Fields insertados por FieldsWriter (Fase 4).
    fields_node = root.find(".//TransformGroup[@name='fields']")
    assert fields_node is not None and len(fields_node) == 1
    # Terrain/DisplacementLayer/sun escritos por I3dWriter (Fase 7).
    terrain = root.find(".//Scene/TerrainTransformGroup")
    assert terrain.get("heightScale") == "255"
    assert terrain.get("lodTextureSize") == str(SIZE)
    # Background integrado (Files + ReferenceNodes, Fase 8).
    ref_names = {
        node.get("name") for node in root.find("Scene").iter("ReferenceNode")
    }
    assert {
        f"background_terrain_part_{i:02d}" for i in range(1, 5)
    } <= ref_names

    # Splines de tráfico: 1 original + 1 reversed.
    splines_root = ET.parse(out / "map/splines.i3d").getroot()
    curves = splines_root.findall(".//NurbsCurve")
    assert len(curves) == 2

    # Farmlands: farmyard + field declarados en el XML.
    farmlands_root = ET.parse(out / "map/config/farmlands.xml").getroot()
    entries = farmlands_root.findall(".//farmland")
    assert [e.get("id") for e in entries] == ["1", "2"]

    # infoLayer_farmlands: IDs 1, 2 y 255 (fill_empty_farmlands).
    farmlands_png = cv2.imread(
        str(out / "map/data/infoLayer_farmlands.png"), cv2.IMREAD_UNCHANGED
    )
    assert farmlands_png.shape == (SIZE // 2, SIZE // 2)
    assert set(np.unique(farmlands_png)) == {1, 2, 255}


def test_generation_info_telemetry(mini_run) -> None:
    project, info = mini_run
    # El fichero y el dict devuelto llevan la misma telemetría.
    on_disk = json.loads(
        (project.paths.output_dir / "generation_info.json").read_text("utf-8")
    )
    assert on_disk["components"] == info["components"]
    assert on_disk["timings_s"] == info["timings_s"]
    assert on_disk["height_scale"] == info["height_scale"]

    assert info["software"].startswith("mapforge ")
    assert info["project"]["size"] == SIZE
    assert info["project"]["background_size"] == SIZE + 4096
    assert info["height_scale"] == 255
    assert info["pipeline_order"] == list(STAGE_ORDER)
    assert info["skipped_stages"] == []

    components = info["components"]
    assert set(components) == set(STAGE_ORDER)
    assert components["dem"]["height_scale"] == 255
    assert components["textures"]["info_layers"]["fields"] == 1
    assert components["textures"]["info_layers"]["roads_polylines"] == 1
    assert components["grle_layers"]["created"] == 21
    assert components["farmlands"] == {"added": 2, "skipped": 0}
    assert components["fields"] == {"added": 1, "skipped": 0}
    assert components["splines"] == {"written_curves": 2}
    assert components["i3d"]["map_xml"] == {"width": "128", "height": "128"}
    assert len(components["background"]["parts"]) == 4
    assert len(components["background"]["i3d_references"]) == 4

    timings = info["timings_s"]
    assert set(STAGE_ORDER) <= set(timings)
    assert timings["total"] > 0


# ------------------------------------------------------------- skip stages


def test_background_disabled_by_setting(tmp_path: Path) -> None:
    project = make_mini_project(tmp_path, background__generate_background=False)
    info = Generator(project).run()

    assert info["components"]["background"] == {
        "status": "skipped",
        "reason": "generate_background=false",
    }
    assert not (project.paths.output_dir / "assets").exists()
    root = ET.parse(project.paths.map_i3d).getroot()
    assert root.find(".//ReferenceNode[@name='background_terrain_part_01']") is None


def test_skip_stages_records_and_warns(tmp_path: Path) -> None:
    """Una etapa saltada queda registrada y el resto del pipeline sigue."""
    project = make_mini_project(tmp_path)
    info = Generator(project, skip_stages={"background"}).run()

    assert info["skipped_stages"] == ["background"]
    assert info["components"]["background"] == {
        "status": "skipped",
        "reason": "skip_stages",
    }
    # Las demás etapas corrieron.
    assert info["components"]["splines"] == {"written_curves": 2}
    assert "background" not in info["timings_s"]


def test_skip_stages_unknown_name_raises(tmp_path: Path) -> None:
    project = make_mini_project(tmp_path)
    with pytest.raises(ValueError, match="skip_stages desconocidas"):
        Generator(project, skip_stages={"no_existe"})


def test_missing_inputs_fail_fast(tmp_path: Path) -> None:
    project = make_mini_project(tmp_path)
    project.paths.osm.unlink()
    with pytest.raises(FileNotFoundError, match="osm"):
        Generator(project).run()


# --------------------------------------------------------------------- CLI


def test_cli_runs_full_pipeline(tmp_path: Path) -> None:
    """La CLI ejecuta el pipeline completo desde un config.yaml."""
    project = make_mini_project(tmp_path)  # crea hm.png + in.osm
    out_dir = tmp_path / "out_cli"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
project:
  name: "Mini CLI"
  output_dir: "{out_dir}"
map:
  size: {SIZE}
  rotation: 0
  latitude: {LAT}
  longitude: {LON}
inputs:
  heightmap: "{tmp_path / 'hm.png'}"
  osm: "{tmp_path / 'in.osm'}"
  template: "{TEMPLATE_ZIP}"
  texture_schema: "{TEXTURE_SCHEMA}"
  grle_schema: "{GRLE_SCHEMA}"
settings:
  seed: 123
  background:
    procedural:
      resize_factor: 32
      texture_size: 256
      noise_octaves: 3
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "mapforge", "generate", "-c", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "map" / "map.i3d").exists()
    assert (out_dir / "map" / "data" / "dem.png").exists()
    assert (out_dir / "generation_info.json").exists()

    info = json.loads((out_dir / "generation_info.json").read_text("utf-8"))
    assert info["project"]["name"] == "Mini CLI"
    assert info["settings"]["seed"] == 123
    assert info["skipped_stages"] == []
