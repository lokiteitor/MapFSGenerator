"""Tests de humo de la Fase 0: config, settings, template, CLI y harness."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "FS25_Valle_Bonito"
TEMPLATE_ZIP = REPO_ROOT / "maps4fs-1.8.242" / "data" / "fs25-map-template.zip"
VALLE_BONITO_YAML = REPO_ROOT / "config" / "valle_bonito.yaml"

sys.path.insert(0, str(REPO_ROOT))

from mapforge.fs25.template import REQUIRED_ENTRIES, deploy_template  # noqa: E402
from mapforge.project import Project  # noqa: E402
from mapforge.settings import (  # noqa: E402
    DEFAULT_INPUT_HEIGHT_SCALE,
    GenerationSettings,
)


# ------------------------------------------------------------------ config


def test_valle_bonito_config_loads() -> None:
    project = Project.from_yaml(VALLE_BONITO_YAML)

    assert project.name == "Valle Bonito"
    assert project.map.size == 8192
    assert project.map.rotation == 0
    assert project.map.rotated_size == 8192
    assert project.map.background_size == 12288
    assert project.map.coordinates == pytest.approx(
        (43.145692357357156, -95.1450786604236)
    )

    # Rutas absolutas y apuntando a los inputs del golden.
    assert project.paths.heightmap.is_absolute()
    assert project.paths.heightmap == GOLDEN_DIR / "valle_bonito.png"
    assert project.paths.osm == GOLDEN_DIR / "custom_osm.osm"
    assert project.paths.template == TEMPLATE_ZIP
    assert project.validate_inputs() == []

    # Settings espejo del generation_settings.json del golden.
    s = project.settings
    assert s.seed == 42
    assert s.input_height_scale == pytest.approx(DEFAULT_INPUT_HEIGHT_SCALE)
    assert s.dem.plateau == 15
    assert s.dem.minimum_height_scale == 255
    assert s.dem.blur_radius == 3
    assert s.background.remove_center is True
    assert s.background.procedural.resize_factor == 8
    assert s.background.procedural.decimation_percent == 25
    assert s.grle.farmland_margin == 5
    assert s.grle.base_price == 12500
    assert s.i3d.spline_density == 2
    assert s.i3d.displacement_layer_max_height == 0.2
    assert s.i3d.displacement_layer_size_factor == 8
    assert s.i3d.displacement_layer_cell_size_base == 16384
    assert s.texture.dissolve is False
    assert s.texture.fields_padding == 3


def test_config_example_loads() -> None:
    project = Project.from_yaml(REPO_ROOT / "config" / "config.example.yaml")
    assert project.map.size == 8192
    assert project.settings.grle.fill_empty_farmlands is True


def test_settings_load_from_maps4fs_generation_settings_json() -> None:
    """El espejo debe tragar directamente el generation_settings.json del golden."""
    with open(GOLDEN_DIR / "generation_settings.json", encoding="utf-8") as f:
        raw = json.load(f)
    settings = GenerationSettings.from_dict(raw)

    assert settings.dem.plateau == 15
    assert settings.dem.water_depth == 15
    assert settings.background.flatten_roads is True
    assert settings.grle.add_farmyards is True
    assert settings.grle.base_grass == "meadow"
    assert settings.i3d.add_reversed_splines is True
    assert settings.i3d.license_plate_prefix == "M4F"
    assert settings.texture.skip_drains is True
    # Los campos propios conservan sus defaults.
    assert settings.seed == 42
    assert settings.input_height_scale == pytest.approx(DEFAULT_INPUT_HEIGHT_SCALE)


def test_rng_is_deterministic() -> None:
    project_a = Project.from_yaml(VALLE_BONITO_YAML)
    project_b = Project.from_yaml(VALLE_BONITO_YAML)
    assert project_a.rng.integers(0, 1 << 30) == project_b.rng.integers(0, 1 << 30)
    assert (
        project_a.fresh_rng(7).integers(0, 1 << 30)
        == project_b.fresh_rng(7).integers(0, 1 << 30)
    )


# ---------------------------------------------------------------- template


def test_template_deploys_from_zip(tmp_path: Path) -> None:
    out = tmp_path / "mapa"
    deploy_template(TEMPLATE_ZIP, out)

    for entry in REQUIRED_ENTRIES:
        assert (out / entry).exists(), f"falta {entry}"
    assert (out / "map" / "data").is_dir()
    assert (out / "map" / "splines.i3d").exists()
    assert (out / "map" / "config" / "farmlands.xml").exists()


def test_template_deploys_from_directory(tmp_path: Path) -> None:
    extracted = tmp_path / "template_dir"
    with zipfile.ZipFile(TEMPLATE_ZIP) as zf:
        zf.extractall(extracted)

    out = tmp_path / "mapa"
    deploy_template(extracted, out)
    assert (out / "map" / "map.i3d").exists()
    assert (out / "modDesc.xml").exists()


def test_template_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        deploy_template(tmp_path / "no_existe.zip", tmp_path / "out")


# --------------------------------------------------------------------- CLI


def test_cli_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mapforge", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "generate" in result.stdout


def test_cli_generate_deploys_template(tmp_path: Path) -> None:
    """La CLI corre el pipeline completo (Fase 9) sobre un mapa mínimo.

    Con el pipeline completo conectado, este smoke usa un mapa 64² con
    heightmap sintético y OSM vacío (el E2E del golden 8192² corre aparte:
    ``docs/validacion_golden.md``) para mantenerlo en segundos.
    """
    import numpy as np

    cv2 = pytest.importorskip("cv2")

    heightmap = np.tile(np.linspace(0, 20000, 64, dtype=np.uint16), (64, 1))
    cv2.imwrite(str(tmp_path / "hm.png"), heightmap)
    (tmp_path / "vacio.osm").write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n<osm version=\"0.6\"></osm>\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "salida"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
project:
  name: "Smoke"
  output_dir: "{out_dir}"
map:
  size: 64
  rotation: 0
  latitude: 43.145692357357156
  longitude: -95.1450786604236
inputs:
  heightmap: "{tmp_path / 'hm.png'}"
  osm: "{tmp_path / 'vacio.osm'}"
  template: "{TEMPLATE_ZIP}"
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
    assert (out_dir / "map" / "data").is_dir()
    assert (out_dir / "map" / "data" / "dem.png").exists()
    assert (out_dir / "generation_info.json").exists()


# ---------------------------------------------------------- compare_golden


@pytest.mark.skipif(not GOLDEN_DIR.is_dir(), reason="golden no disponible")
def test_compare_golden_runs_on_template_deploy(tmp_path: Path) -> None:
    out = tmp_path / "mapa"
    deploy_template(TEMPLATE_ZIP, out)

    json_path = tmp_path / "reporte.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "compare_golden.py"),
            str(out),
            "--golden",
            str(GOLDEN_DIR),
            "--json",
            str(json_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # 0 = identico, 1 = con diferencias; ambos validos, nunca crash.
    assert result.returncode in (0, 1), result.stderr
    assert "Resumen:" in result.stdout

    report = json.loads(json_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["compared"] > 0
    # El template solo trae el esqueleto: la mayoria del golden aun falta.
    assert summary["missing_in_output"] > 0
    # map.i3d del template debe haberse comparado como XML.
    assert report["files"]["map/map.i3d"]["kind"] == "xml"
