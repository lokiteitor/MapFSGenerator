"""Tests de la Fase 6: splines de tráfico (splines/traffic.py).

Cubren el caso del plan (§53 del doc madre): spline de carretera recta
sintética + reversed, más las utilidades ``interpolate_points`` (semántica
EXACTA de component.py 1.8: puntos extra entre cada par, truncados a int),
``fit_linestring_into_bounds`` y ``top_left_to_center``, y el contrato del
i3d resultante (Shapes/Scene/UserAttributes, nodeId desde 5000, formato de
los CV con mezcla float/int como el golden).
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import GenerationSettings  # noqa: E402
from mapforge.splines import (  # noqa: E402
    SPLINES_NODE_ID_STARTING_VALUE,
    TrafficSplinesWriter,
    fit_linestring_into_bounds,
    interpolate_points,
    top_left_to_center,
)

SIZE = 128

#: Esqueleto de map/splines.i3d del template limpio (informe §S1).
SKELETON = (
    '<?xml version="1.0" encoding="iso-8859-1"?>\n'
    '<i3D name="spline01" version="1.6" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xsi:noNamespaceSchemaLocation="http://i3d.giants.ch/schema/i3d-1.6.xsd">\n'
    "    <Shapes>\n"
    "    </Shapes>\n"
    "    <Scene>\n"
    "    </Scene>\n"
    "    <UserAttributes>\n"
    "    </UserAttributes>  \n"
    "</i3D>"
)


# ------------------------------------------------------------------ helpers


def make_project(tmp_path: Path, size: int = SIZE) -> Project:
    """Proyecto sintético mínimo para el writer de splines."""
    paths = ProjectPaths(
        heightmap=tmp_path / "heightmap.png",
        osm=tmp_path / "custom.osm",
        template=tmp_path / "template",
        texture_schema=tmp_path / "texture_schema.json",
        grle_schema=tmp_path / "grle_schema.json",
        output_dir=tmp_path / "out",
    )
    project = Project(
        name="test_splines",
        map_params=MapParams(size=size, rotation=0, latitude=43.0, longitude=-95.0),
        paths=paths,
        settings=GenerationSettings(),
    )
    return project


def write_inputs(project: Project, roads: list[dict]) -> None:
    """Escribe textures.json (roads_polylines) y el esqueleto splines.i3d."""
    import json

    project.paths.info_layers_dir.mkdir(parents=True, exist_ok=True)
    with open(project.paths.textures_json, "w", encoding="utf-8") as f:
        json.dump({"roads_polylines": roads}, f)
    project.paths.map_dir.mkdir(parents=True, exist_ok=True)
    project.paths.splines_i3d.write_text(SKELETON, encoding="iso-8859-1")


def flat_dem(value: int = 3500, size: int = SIZE) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint16)


# ---------------------------------------------------- interpolate_points


class TestInterpolatePoints:
    def test_density_2_adds_two_int_points_per_pair(self):
        # Segmento de 15 px (el patrón de spline_1 del golden: 982→997).
        result = interpolate_points([(7239.0, 982.0), (7239.0, 997.0)], num_points=2)
        assert result == [(7239.0, 982.0), (7239, 987), (7239, 992), (7239.0, 997.0)]

    def test_truncates_to_int_not_rounds(self):
        # 250/3 = 83.33 → int() trunca (golden: 997→1247 da 1080 y 1163).
        result = interpolate_points([(0.0, 997.0), (0.0, 1247.0)], num_points=2)
        assert result == [(0.0, 997.0), (0, 1080), (0, 1163), (0.0, 1247.0)]

    def test_originals_keep_type_interpolated_are_int(self):
        # Mezcla float (originales) / int (interpolados) — patrón del golden.
        result = interpolate_points([(1.5, 2.5), (7.5, 8.5)], num_points=1)
        assert result[0] == (1.5, 2.5)
        assert result[1] == (4, 5)  # int((1.5+7.5)/2), int((2.5+8.5)/2)
        assert result[2] == (7.5, 8.5)
        assert isinstance(result[1][0], int)

    def test_num_points_below_1_returns_unchanged(self):
        points = [(0, 0), (10, 10)]
        assert interpolate_points(points, num_points=0) is points

    def test_empty_polyline_returns_unchanged(self):
        assert interpolate_points([], num_points=2) == []

    def test_multi_segment_count(self):
        # n puntos, densidad d → n + (n-1)×d puntos.
        points = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (90.0, 0.0)]
        result = interpolate_points(points, num_points=2)
        assert len(result) == 4 + 3 * 2

    def test_default_density_is_4(self):
        result = interpolate_points([(0.0, 0.0), (10.0, 0.0)])
        assert len(result) == 2 + 4


# ------------------------------------------- fit_linestring_into_bounds


class TestFitLinestring:
    def test_inside_passthrough_as_floats(self):
        fitted = fit_linestring_into_bounds([(10, 20), (30, 40)], map_size=SIZE)
        assert fitted == [(10.0, 20.0), (30.0, 40.0)]
        assert all(isinstance(v, float) for point in fitted for v in point)

    def test_clips_to_map_box(self):
        # Línea que sale por la derecha: se recorta en x = map_size.
        fitted = fit_linestring_into_bounds([(100, 64), (200, 64)], map_size=SIZE)
        assert fitted == [(100.0, 64.0), (128.0, 64.0)]

    def test_fully_outside_raises(self):
        with pytest.raises(ValueError):
            fit_linestring_into_bounds([(200, 200), (300, 300)], map_size=SIZE)

    def test_split_into_parts_raises(self):
        # Entra, sale y vuelve a entrar → MultiLineString → ValueError (1.8 salta).
        points = [(10, 64), (-20, 64), (-20, 80), (10, 80)]
        with pytest.raises(ValueError):
            fit_linestring_into_bounds(points, map_size=SIZE)

    def test_border_shrinks_bounds(self):
        # bounds = box(border, border, map_size − border, map_size − border).
        fitted = fit_linestring_into_bounds(
            [(0, 64), (127, 64)], map_size=SIZE, border=10
        )
        assert fitted == [(10.0, 64.0), (118.0, 64.0)]


# ------------------------------------------------------ top_left_to_center


def test_top_left_to_center_types_and_values():
    # int → int; float → float (afecta al formato "3143" vs "3143.0" del i3d).
    assert top_left_to_center((7239, 982), 8192) == (3143, -3114)
    assert top_left_to_center((7239.0, 982.0), 8192) == (3143.0, -3114.0)
    cx, cy = top_left_to_center((7239, 982), 8192)
    assert isinstance(cx, int) and isinstance(cy, int)
    fx, fy = top_left_to_center((7239.0, 982.0), 8192)
    assert isinstance(fx, float) and isinstance(fy, float)


# ------------------------------------------------- writer: carretera recta


class TestStraightRoadSpline:
    """§53: spline de carretera recta sintética + reversed."""

    TAGS = "{'highway': ['secondary']}"

    def run_writer(self, tmp_path: Path, **settings_overrides) -> ET.Element:
        project = make_project(tmp_path)
        project.settings.i3d.spline_density = 2
        project.settings.i3d.add_reversed_splines = True
        for key, value in settings_overrides.items():
            setattr(project.settings.i3d, key, value)
        # Carretera recta vertical: x=64 constante, y de 10 a 25 (15 px).
        write_inputs(
            project,
            [{"points": [[64, 10], [64, 25]], "tags": self.TAGS}],
        )
        writer = TrafficSplinesWriter(
            project, dem_not_resized=flat_dem(3500), height_scale=255
        )
        self.written = writer.run()
        self.i3d_text = project.paths.splines_i3d.read_text(encoding="iso-8859-1")
        return ET.parse(project.paths.splines_i3d).getroot()

    def test_original_and_reversed_curves(self, tmp_path):
        root = self.run_writer(tmp_path)
        curves = root.findall(".//Shapes/NurbsCurve")
        assert self.written == 2
        assert [c.get("name") for c in curves] == [
            f"spline_1_original_{self.TAGS}",
            f"spline_1_reversed_{self.TAGS}",
        ]
        for c in curves:
            assert c.get("degree") == "3"
            assert c.get("form") == "open"
        assert [c.get("shapeId") for c in curves] == ["5000", "5001"]
        assert int(curves[0].get("shapeId")) == SPLINES_NODE_ID_STARTING_VALUE

    def test_cv_values_straight_road(self, tmp_path):
        root = self.run_writer(tmp_path)
        curves = root.findall(".//Shapes/NurbsCurve")
        # z plano = 3500 × 255/65535 = 3500/257 (la Y del golden).
        z = "13.618677042801556"
        # fit → floats; interpolate densidad 2 → ints en 15, 20; centro − 64.
        expected = [
            f"0.0, {z}, -54.0",
            f"0, {z}, -49",
            f"0, {z}, -44",
            f"0.0, {z}, -39.0",
        ]
        original = [cv.get("c") for cv in curves[0].findall("cv")]
        reversed_ = [cv.get("c") for cv in curves[1].findall("cv")]
        assert original == expected
        assert reversed_ == expected[::-1]

    def test_scene_shapes_and_user_attributes(self, tmp_path):
        root = self.run_writer(tmp_path)
        shapes = root.findall(".//Scene/Shape")
        assert [(s.get("name"), s.get("translation"), s.get("nodeId"), s.get("shapeId"))
                for s in shapes] == [
            (f"spline_1_original_{self.TAGS}", "0 0 0", "5000", "5000"),
            (f"spline_1_reversed_{self.TAGS}", "0 0 0", "5001", "5001"),
        ]
        uas = root.findall(".//UserAttributes/UserAttribute")
        assert [ua.get("nodeId") for ua in uas] == ["5000", "5001"]
        for ua in uas:
            attrs = [
                (a.get("name"), a.get("type"), a.get("value"))
                for a in ua.findall("Attribute")
            ]
            assert attrs == [
                ("maxSpeedScale", "integer", "1"),
                ("speedLimit", "integer", "100"),
            ]

    def test_output_format_matches_golden_conventions(self, tmp_path):
        self.run_writer(tmp_path)
        lines = self.i3d_text.splitlines()
        # Declaración con comillas dobles, como el golden.
        assert lines[0] == '<?xml version="1.0" encoding="iso-8859-1"?>'
        # Prefijo xsi conservado e indentación de 2 espacios.
        assert 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' in lines[1]
        assert "  <Shapes>" in lines
        assert any(line.startswith("    <NurbsCurve name=") for line in lines)
        assert any(line.startswith('      <cv c="') and line.endswith('" />')
                   for line in lines)

    def test_without_reversed_only_original(self, tmp_path):
        root = self.run_writer(tmp_path, add_reversed_splines=False)
        curves = root.findall(".//Shapes/NurbsCurve")
        assert self.written == 1
        assert [c.get("name") for c in curves] == [f"spline_1_original_{self.TAGS}"]
        assert len(root.findall(".//Scene/Shape")) == 1
        assert len(root.findall(".//UserAttributes/UserAttribute")) == 1


# ----------------------------------------------- writer: casos adicionales


def test_road_out_of_bounds_skipped_but_road_id_advances(tmp_path):
    """Una carretera fuera del mapa se salta, pero el road_id avanza
    (enumerate, como 1.8): la segunda se llama spline_2."""
    project = make_project(tmp_path)
    tags = "{'highway': True}"
    write_inputs(
        project,
        [
            {"points": [[500, 500], [600, 600]], "tags": tags},  # fuera
            {"points": [[10, 10], [40, 10]], "tags": tags},
        ],
    )
    writer = TrafficSplinesWriter(project, dem_not_resized=flat_dem(), height_scale=255)
    written = writer.run()
    root = ET.parse(project.paths.splines_i3d).getroot()
    curves = root.findall(".//Shapes/NurbsCurve")
    assert written == 2  # original + reversed de la única que encaja
    assert [c.get("name") for c in curves] == [
        f"spline_2_original_{tags}",
        f"spline_2_reversed_{tags}",
    ]
    # Los nodeId siguen siendo consecutivos desde 5000 (no se reservan huecos).
    assert [c.get("shapeId") for c in curves] == ["5000", "5001"]


def test_z_sampled_from_dem_per_point_with_clamp(tmp_path):
    """La Z se muestrea del DEM not_resized por punto (sin interpolación) y
    con clamp a los bordes."""
    project = make_project(tmp_path)
    project.settings.i3d.spline_density = 0  # sin interpolación: 2 CVs
    project.settings.i3d.add_reversed_splines = False
    dem = flat_dem(0)
    dem[10, :] = 2570  # fila y=10 → 2570/257 = 10 m
    dem[127, :] = 5140  # última fila → 20 m (clamp de y=128 → 127)
    write_inputs(
        project,
        [{"points": [[64, 10], [64, 128]], "tags": "{'highway': True}"}],
    )
    writer = TrafficSplinesWriter(project, dem_not_resized=dem, height_scale=255)
    writer.run()
    root = ET.parse(project.paths.splines_i3d).getroot()
    cvs = [cv.get("c") for cv in root.findall(".//Shapes/NurbsCurve/cv")]
    assert cvs[0] == "0.0, 10.0, -54.0"
    # y=128 se clampa a la fila 127 del DEM (y la coord del CV se mantiene).
    assert cvs[1] == "0.0, 20.0, 64.0"


def test_multiplier_divides_z(tmp_path):
    project = make_project(tmp_path)
    project.settings.dem.multiplier = 2
    project.settings.i3d.spline_density = 0
    project.settings.i3d.add_reversed_splines = False
    write_inputs(
        project, [{"points": [[10, 10], [20, 10]], "tags": "{'highway': True}"}]
    )
    writer = TrafficSplinesWriter(
        project, dem_not_resized=flat_dem(2570), height_scale=255
    )
    writer.run()
    root = ET.parse(project.paths.splines_i3d).getroot()
    cv = root.find(".//Shapes/NurbsCurve/cv")
    assert cv.get("c") == "-54.0, 5.0, -54.0"  # 10 m / multiplier 2


def test_no_roads_leaves_skeleton_untouched(tmp_path):
    project = make_project(tmp_path)
    write_inputs(project, [])
    original = project.paths.splines_i3d.read_text(encoding="iso-8859-1")
    writer = TrafficSplinesWriter(project, dem_not_resized=flat_dem(), height_scale=255)
    assert writer.run() == 0
    assert project.paths.splines_i3d.read_text(encoding="iso-8859-1") == original


def test_missing_height_scale_raises(tmp_path):
    project = make_project(tmp_path)
    write_inputs(
        project, [{"points": [[10, 10], [20, 10]], "tags": "{'highway': True}"}]
    )
    writer = TrafficSplinesWriter(project, dem_not_resized=flat_dem())
    with pytest.raises(RuntimeError):
        writer.run()


def test_height_scale_taken_from_project(tmp_path):
    """La Fase 1 deja project.height_scale; el writer lo usa por defecto."""
    project = make_project(tmp_path)
    project.height_scale = 255
    project.settings.i3d.spline_density = 0
    project.settings.i3d.add_reversed_splines = False
    write_inputs(
        project, [{"points": [[10, 10], [20, 10]], "tags": "{'highway': True}"}]
    )
    writer = TrafficSplinesWriter(project, dem_not_resized=flat_dem(3500))
    assert writer.run() == 1
    root = ET.parse(project.paths.splines_i3d).getroot()
    assert "13.618677042801556" in root.find(".//Shapes/NurbsCurve/cv").get("c")
