"""Tests de la Fase 2: parser OSM, matching de tags y proyección."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, Polygon

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "FS25_Valle_Bonito"

sys.path.insert(0, str(REPO_ROOT))

from mapforge.osm import (  # noqa: E402
    MapProjection,
    bbox_for_map,
    bbox_from_point,
    features_matching,
    latlon_to_pixel,
    parse_osm,
    project_geometry,
    tags_match,
)

# Centro y tamaño del golden (main_settings.json).
GOLDEN_LAT = 43.145692357357156
GOLDEN_LON = -95.1450786604236
GOLDEN_SIZE = 8192
# Texture.bbox del generation_info.json golden: (north, south, east, west).
GOLDEN_BBOX = (
    43.182528518298945,
    43.10885619641537,
    -95.09459168770303,
    -95.19556563314416,
)

OSM_HEADER = "<?xml version='1.0' encoding='utf-8'?>\n<osm version=\"0.6\" generator=\"test\">\n"


def _write_osm(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "test.osm"
    path.write_text(OSM_HEADER + body + "</osm>\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------- tags


def test_tags_match_exact_value() -> None:
    assert tags_match({"landuse": "farmland"}, {"landuse": "farmland"})
    assert not tags_match({"landuse": "forest"}, {"landuse": "farmland"})


def test_tags_match_list_of_values() -> None:
    wanted = {"natural": ["wood", "tree_row"]}
    assert tags_match({"natural": "wood"}, wanted)
    assert tags_match({"natural": "tree_row"}, wanted)
    assert not tags_match({"natural": "water"}, wanted)


def test_tags_match_true_means_key_present() -> None:
    assert tags_match({"building": "yes"}, {"building": True})
    assert tags_match({"building": "house"}, {"building": True})
    assert not tags_match({"highway": "road"}, {"building": True})


def test_tags_match_union_of_keys() -> None:
    wanted = {"natural": ["wood", "tree_row"], "landuse": "forest"}
    assert tags_match({"landuse": "forest"}, wanted)
    assert tags_match({"natural": "wood"}, wanted)
    assert not tags_match({"landuse": "farmland"}, wanted)


def test_tags_match_empty_never_matches() -> None:
    assert not tags_match({}, {"building": True})
    assert not tags_match({"building": "yes"}, {})


def test_tags_match_rejects_unsupported_definition() -> None:
    with pytest.raises(ValueError):
        tags_match({"lanes": "2"}, {"lanes": 2})


# ----------------------------------------------------------------- parser


def test_parse_open_way_is_linestring(tmp_path: Path) -> None:
    """Una línea OSM sintética (test del plan §53)."""
    osm = _write_osm(
        tmp_path,
        """
  <node id="1" lat="43.10" lon="-95.20" />
  <node id="2" lat="43.11" lon="-95.19" />
  <node id="3" lat="43.12" lon="-95.18" />
  <way id="10">
    <nd ref="1" /><nd ref="2" /><nd ref="3" />
    <tag k="highway" v="secondary" />
  </way>
""",
    )
    data = parse_osm(osm)

    assert len(data.ways) == 1
    features = features_matching(data, {"highway": True})
    assert len(features) == 1
    feature = features[0]
    assert feature.element_type == "way"
    assert feature.element_id == 10
    assert feature.tags == {"highway": "secondary"}
    assert isinstance(feature.geometry, LineString)
    # Convención shapely/osmnx: coords = (lon, lat).
    assert list(feature.geometry.coords) == [
        (-95.20, 43.10),
        (-95.19, 43.11),
        (-95.18, 43.12),
    ]


def test_parse_closed_way_is_polygon(tmp_path: Path) -> None:
    """Un polígono OSM sintético (test del plan §53)."""
    osm = _write_osm(
        tmp_path,
        """
  <node id="1" lat="43.10" lon="-95.20" />
  <node id="2" lat="43.10" lon="-95.19" />
  <node id="3" lat="43.11" lon="-95.19" />
  <node id="4" lat="43.11" lon="-95.20" />
  <way id="20">
    <nd ref="1" /><nd ref="2" /><nd ref="3" /><nd ref="4" /><nd ref="1" />
    <tag k="landuse" v="farmland" />
  </way>
""",
    )
    data = parse_osm(osm)

    features = features_matching(data, {"landuse": ["farmland", "meadow"]})
    assert len(features) == 1
    geometry = features[0].geometry
    assert isinstance(geometry, Polygon)
    assert geometry.is_valid
    # Área de un rectángulo de 0.01° × 0.01°.
    assert geometry.area == pytest.approx(1e-4)


def test_parse_tagged_node_is_point(tmp_path: Path) -> None:
    osm = _write_osm(
        tmp_path,
        """
  <node id="1" lat="43.10" lon="-95.20"><tag k="natural" v="tree" /></node>
  <node id="2" lat="43.11" lon="-95.19" />
""",
    )
    data = parse_osm(osm)

    features = features_matching(data, {"natural": ["tree"]})
    assert len(features) == 1
    assert features[0].element_type == "node"
    assert isinstance(features[0].geometry, Point)
    assert (features[0].geometry.x, features[0].geometry.y) == (-95.20, 43.10)
    # El nodo sin tags no genera feature.
    assert len(data.features) == 1


def test_parse_multipolygon_relation_with_hole(tmp_path: Path) -> None:
    """Relation multipolygon: outer cosido de dos ways abiertos + inner cerrado."""
    osm = _write_osm(
        tmp_path,
        """
  <node id="1" lat="43.10" lon="-95.20" />
  <node id="2" lat="43.10" lon="-95.10" />
  <node id="3" lat="43.20" lon="-95.10" />
  <node id="4" lat="43.20" lon="-95.20" />
  <node id="5" lat="43.13" lon="-95.17" />
  <node id="6" lat="43.13" lon="-95.13" />
  <node id="7" lat="43.17" lon="-95.13" />
  <node id="8" lat="43.17" lon="-95.17" />
  <way id="30"><nd ref="1" /><nd ref="2" /><nd ref="3" /></way>
  <way id="31"><nd ref="3" /><nd ref="4" /><nd ref="1" /></way>
  <way id="32">
    <nd ref="5" /><nd ref="6" /><nd ref="7" /><nd ref="8" /><nd ref="5" />
  </way>
  <relation id="40">
    <member type="way" ref="30" role="outer" />
    <member type="way" ref="31" role="outer" />
    <member type="way" ref="32" role="inner" />
    <tag k="type" v="multipolygon" />
    <tag k="landuse" v="forest" />
  </relation>
""",
    )
    data = parse_osm(osm)

    features = features_matching(data, {"landuse": "forest"})
    assert len(features) == 1
    feature = features[0]
    assert feature.element_type == "relation"
    geometry = feature.geometry
    assert isinstance(geometry, Polygon)
    assert len(geometry.interiors) == 1
    # 0.1×0.1 exterior menos agujero 0.04×0.04.
    assert geometry.area == pytest.approx(0.1 * 0.1 - 0.04 * 0.04)


def test_parse_ignores_deleted_and_missing_refs(tmp_path: Path) -> None:
    osm = _write_osm(
        tmp_path,
        """
  <node id="1" lat="43.10" lon="-95.20" />
  <node id="2" lat="43.11" lon="-95.19" />
  <way id="10" action="delete">
    <nd ref="1" /><nd ref="2" />
    <tag k="highway" v="road" />
  </way>
  <way id="11">
    <nd ref="1" /><nd ref="2" /><nd ref="999" />
    <tag k="highway" v="road" />
  </way>
  <way id="12">
    <nd ref="998" /><nd ref="999" />
    <tag k="highway" v="road" />
  </way>
""",
    )
    data = parse_osm(osm)

    features = features_matching(data, {"highway": True})
    # way 10 borrado; way 12 sin nodos resolubles; way 11 sobrevive con 2 puntos.
    assert [f.element_id for f in features] == [11]
    assert len(list(features[0].geometry.coords)) == 2


# ------------------------------------------------------------- proyección


def test_bbox_from_point_matches_golden() -> None:
    """Texture.bbox del golden debe reproducirse con precisión < 1e-6 grados."""
    bbox = bbox_for_map(GOLDEN_LAT, GOLDEN_LON, GOLDEN_SIZE)
    for ours, golden in zip(bbox, GOLDEN_BBOX):
        assert abs(ours - golden) < 1e-6
    # La fórmula es la de osmnx exacta: el error real es 0.0 en float64.
    assert bbox == pytest.approx(GOLDEN_BBOX, abs=1e-12)


def test_bbox_from_point_is_symmetric() -> None:
    north, south, east, west = bbox_from_point(43.0, -95.0, 1000)
    assert north - 43.0 == pytest.approx(43.0 - south)
    assert east - (-95.0) == pytest.approx(-95.0 - west)
    # delta_lon > delta_lat fuera del ecuador (corrección cos(lat)).
    assert (east - west) > (north - south)


def test_latlon_to_pixel_corners_and_truncation() -> None:
    bbox = (43.2, 43.1, -95.1, -95.2)  # north, south, east, west
    assert latlon_to_pixel(43.2, -95.2, bbox, 1000) == (0, 0)
    assert latlon_to_pixel(43.1, -95.1, bbox, 1000) == (1000, 1000)
    assert latlon_to_pixel(43.175, -95.175, bbox, 1000) == (250, 250)
    # Réplica Maps4FS: int() trunca, no redondea (499.99… → 499).
    x, _y = latlon_to_pixel(43.15, -95.15, bbox, 1000)
    assert x == int((-95.15 - -95.2) / (-95.1 - -95.2) * 1000)


def test_project_geometry_polygon_with_hole() -> None:
    bbox = (43.2, 43.1, -95.1, -95.2)
    polygon = Polygon(
        [(-95.2, 43.1), (-95.1, 43.1), (-95.1, 43.2), (-95.2, 43.2)],
        [[(-95.16, 43.14), (-95.14, 43.14), (-95.14, 43.16), (-95.16, 43.16)]],
    )
    projected = project_geometry(polygon, bbox, 1000)
    assert isinstance(projected, Polygon)
    assert projected.bounds == (0.0, 0.0, 1000.0, 1000.0)
    assert len(projected.interiors) == 1


# ------------------------------------------- validación contra el golden


@pytest.mark.skipif(not GOLDEN_DIR.is_dir(), reason="golden no disponible")
def test_golden_spline1_projection() -> None:
    """El way highway vertical en lon=-95.106333644 debe reproducir los CVs
    ancla de spline_1 del golden map/splines.i3d (x=3143, z=-3114..-1898)."""
    data = parse_osm(GOLDEN_DIR / "custom_osm.osm")
    projection = MapProjection(GOLDEN_LAT, GOLDEN_LON, GOLDEN_SIZE)

    matched = data.features_matching(
        {"highway": ["secondary", "tertiary", "road", "service"]}
    )
    assert matched, "ningún highway casó en el golden OSM"

    # spline_1 = primer way casado en orden de documento.
    first = matched[0]
    assert isinstance(first.geometry, LineString)
    lons = {lon for lon, _lat in first.geometry.coords}
    assert lons == {-95.106333644}

    centered = [
        (x - GOLDEN_SIZE // 2, y - GOLDEN_SIZE // 2)
        for lon, lat in first.geometry.coords
        for x, y in [projection.latlon_to_pixel(lat, lon)]
    ]
    # CVs ancla (los .0 del golden; los intermedios son interpolación de Fase 6).
    assert centered == [
        (3143, -3114),
        (3143, -3099),
        (3143, -2849),
        (3143, -2598),
        (3143, -2348),
        (3143, -2098),
        (3143, -1898),
    ]


@pytest.mark.skipif(not GOLDEN_DIR.is_dir(), reason="golden no disponible")
def test_golden_osm_parses_completely() -> None:
    data = parse_osm(GOLDEN_DIR / "custom_osm.osm")
    assert len(data.nodes) > 29000
    assert len(data.ways) == 282
    assert len(data.features) == 282
    # Todos los ways con tags producen geometría.
    tagged_ways = [w for w in data.ways.values() if w.tags]
    assert len(data.features) == len(tagged_ways) + len(data.node_tags)


def test_repo_osm_parses_completely() -> None:
    """Valida que el OSM del repositorio se parsea sin errores."""
    osm_path = REPO_ROOT / "maps" / "custom.osm"
    assert osm_path.is_file()
    data = parse_osm(osm_path)
    assert len(data.nodes) > 0
    assert len(data.ways) > 0
    assert len(data.features) > 0
