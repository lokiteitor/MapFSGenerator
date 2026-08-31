"""Tests de la Fase 4: fields (polígono sintético del §53 del doc madre).

Cubren: fit_into_bounds (intersección con el box del mapa, border), coords a
sistema centro, centroide/área shapely, el subárbol XML del field (estructura,
Note con hectáreas, numeración de nodeIds de 1.8 con su quirk) y la inserción
en ``gameplay/fields`` de un map.i3d (ElementTree + FieldsWriter).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mapforge.fields import (  # noqa: E402
    FIELDS_ATTRIBUTES,
    NODE_ID_STARTING_VALUE,
    FieldsWriter,
    add_fields_to_i3d,
    add_fields_to_i3d_file,
    build_field_node,
    field_layer_border,
    fit_polygon_into_bounds,
    polygon_area_ha,
    polygon_center,
    top_left_to_center,
)
from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import GenerationSettings  # noqa: E402

SIZE = 200

#: map.i3d mínimo con los nodos que necesita la inserción.
MINIMAL_I3D = """<?xml version="1.0" encoding="iso-8859-1"?>
<i3D name="map.i3d" version="1.6">
  <Scene>
    <TransformGroup name="gameplay" nodeId="35">
      <TransformGroup name="fields" translation="0 0 0" nodeId="36"/>
      <TransformGroup name="guidedTours" nodeId="37"/>
    </TransformGroup>
  </Scene>
  <UserAttributes>
    <UserAttribute nodeId="85">
      <Attribute name="onCreate" type="scriptCallback" value="X.onCreate"/>
    </UserAttribute>
  </UserAttributes>
</i3D>
"""


def square(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ------------------------------------------------------------ fit + geometría


def test_fit_polygon_inside_unchanged() -> None:
    points = square(50, 50, 150, 150)
    fitted = fit_polygon_into_bounds(points, map_size=SIZE)
    # shapely cierra el anillo: mismos vértices, con el primero duplicado al final.
    assert fitted[0] == fitted[-1]
    assert set(fitted) == set(points)
    assert len(fitted) == 5


def test_fit_polygon_clipped_by_bounds_and_border() -> None:
    points = square(-50, -50, 100, 100)  # se sale por arriba/izquierda
    border = 10
    fitted = fit_polygon_into_bounds(points, map_size=SIZE, border=border)
    xs = [p[0] for p in fitted]
    ys = [p[1] for p in fitted]
    assert min(xs) == border and min(ys) == border
    assert max(xs) == 100 and max(ys) == 100


def test_fit_polygon_outside_raises() -> None:
    points = square(300, 300, 400, 400)  # fuera del mapa
    with pytest.raises(ValueError):
        fit_polygon_into_bounds(points, map_size=SIZE)


def test_fit_polygon_rotation_requires_rotated_size() -> None:
    with pytest.raises(ValueError):
        fit_polygon_into_bounds(square(0, 0, 10, 10), map_size=SIZE, angle=45)


def test_top_left_to_center_and_centroid() -> None:
    assert top_left_to_center((0, 0), SIZE) == (-100, -100)
    assert top_left_to_center((SIZE, SIZE), SIZE) == (100, 100)
    # centroide shapely truncado a int
    assert polygon_center(square(10, 10, 110, 110)) == (60, 60)
    assert polygon_center(square(0, 0, 5, 5)) == (2, 2)  # 2.5 → int → 2


def test_polygon_area_ha() -> None:
    # cuadrado de 100×100 px = 10 000 m² = 1 ha
    assert polygon_area_ha(square(0, 0, 100, 100)) == pytest.approx(1.0)


# --------------------------------------------------------------- subárbol XML


def test_build_field_node_structure() -> None:
    # cuadrado 100×100 centrado en (-40, -40) del sistema centro
    ccs = [top_left_to_center(p, SIZE) for p in square(10, 10, 110, 110)]
    ccs.append(ccs[0])  # anillo cerrado como lo devuelve fit
    node, next_id = build_field_node(1, ccs, NODE_ID_STARTING_VALUE)
    assert node is not None

    assert node.tag == "TransformGroup"
    assert node.get("name") == "field1"
    assert node.get("translation") == "-40 0 -40"
    assert node.get("nodeId") == str(NODE_ID_STARTING_VALUE)

    pp = node.find("TransformGroup[@name='polygonPoints']")
    assert pp is not None and pp.get("nodeId") == "2001"
    points = list(pp)
    assert len(points) == 5
    assert [p.get("name") for p in points] == [f"point{i}" for i in range(1, 6)]
    # relativos enteros al centroide: (10,10) − centro (100,100) = (−90,−90) − (−40,−40)
    assert points[0].get("translation") == "-50 0 -50"
    assert points[1].get("translation") == "50 0 -50"
    assert points[2].get("translation") == "50 0 50"
    assert points[3].get("translation") == "-50 0 50"
    assert points[4].get("translation") == points[0].get("translation")
    # quirk de 1.8: el bucle incrementa antes de usar → point1 = 2003 (2002 sin asignar)
    assert points[0].get("nodeId") == "2003"
    assert points[-1].get("nodeId") == "2007"

    name_indicator = node.find("TransformGroup[@name='nameIndicator']")
    assert name_indicator is not None and name_indicator.get("nodeId") == "2008"
    note = name_indicator.find("Note")
    assert note is not None
    assert note.get("nodeId") == "2009"
    # &#xA; LITERAL en el valor del atributo (quirk de Maps4FS, FACT artefacto)
    assert note.get("text") == "field1&#xA;1.00 ha"
    assert note.get("color") == "4278190080"
    assert note.get("fixedSize") == "true"

    teleport = node.find("TransformGroup[@name='teleportIndicator']")
    assert teleport is not None and teleport.get("nodeId") == "2010"
    assert next_id == 2010


def test_build_field_node_degenerate_returns_none() -> None:
    node, node_id = build_field_node(1, [(0, 0), (1, 1)], 2000)
    assert node is None and node_id == 2000


# ---------------------------------------------------- inserción en el map.i3d


def test_add_fields_to_i3d_minimal() -> None:
    root = ET.fromstring(MINIMAL_I3D)
    fields = [
        square(10, 10, 110, 110),  # válido
        square(300, 300, 400, 400),  # fuera de bounds → saltado sin consumir id
        square(120, 120, 180, 180),  # válido → field2
    ]
    stats = add_fields_to_i3d(root, fields, map_size=SIZE)
    assert stats.added == 2 and stats.skipped == 1

    fields_node = root.find(".//TransformGroup[@name='fields']")
    assert fields_node is not None
    names = [c.get("name") for c in fields_node]
    assert names == ["field1", "field2"]  # ids consecutivos solo de los añadidos

    field1, field2 = list(fields_node)
    # UserAttribute por field apuntando al nodeId del TransformGroup del field
    ua_nodes = root.findall(".//UserAttributes/UserAttribute")
    new_uas = ua_nodes[1:]  # el primero es el preexistente
    assert [ua.get("nodeId") for ua in new_uas] == [
        field1.get("nodeId"),
        field2.get("nodeId"),
    ]
    for ua in new_uas:
        attrs = [(a.get("name"), a.get("type"), a.get("value")) for a in ua]
        assert attrs == FIELDS_ATTRIBUTES

    # numeración entre fields: field1 5 puntos → siguiente id = 2000+5+5+1 = 2011
    assert field1.get("nodeId") == "2000"
    assert field2.get("nodeId") == "2011"

    # stats por field: centroide, nº de puntos y hectáreas
    assert stats.fields[0]["centroid"] == [-40, -40]
    assert stats.fields[0]["n_points"] == 5
    assert stats.fields[0]["ha"] == pytest.approx(1.0)
    assert stats.fields[1]["ha"] == pytest.approx(0.36)


def test_add_fields_missing_nodes_raises() -> None:
    root = ET.fromstring("<i3D><Scene/></i3D>")
    with pytest.raises(ValueError):
        add_fields_to_i3d(root, [square(10, 10, 20, 20)], map_size=SIZE)


def test_add_fields_to_i3d_file_roundtrip(tmp_path: Path) -> None:
    i3d_in = tmp_path / "map.i3d"
    i3d_in.write_text(MINIMAL_I3D, encoding="utf-8")
    i3d_out = tmp_path / "out.i3d"

    stats = add_fields_to_i3d_file(
        i3d_in, i3d_out, [square(10, 10, 110, 110)], map_size=SIZE
    )
    assert stats.added == 1

    raw = i3d_out.read_text(encoding="utf-8")
    # Note con &#xA; literal escapado como &amp;#xA; (byte a byte como el golden)
    assert 'text="field1&amp;#xA;1.00 ha"' in raw

    # el fichero reparsea y conserva la estructura previa
    reparsed = ET.parse(i3d_out).getroot()
    assert reparsed.find(".//TransformGroup[@name='guidedTours']") is not None
    note = reparsed.find(".//Note")
    assert note is not None and note.get("text") == "field1&#xA;1.00 ha"
    fields_node = reparsed.find(".//TransformGroup[@name='fields']")
    assert fields_node is not None and len(list(fields_node)) == 1


def test_fields_writer_end_to_end(tmp_path: Path) -> None:
    # proyecto sintético: textures.json + map.i3d mínimo + schema con border
    schema = [
        {
            "name": "mudDark",
            "count": 2,
            "tags": {"landuse": ["farmland"]},
            "priority": 4,
            "info_layer": "fields",
            "usage": "field",
            "border": 10,
        }
    ]
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    paths = ProjectPaths(
        heightmap=tmp_path / "hm.png",
        osm=tmp_path / "test.osm",
        template=tmp_path / "template",
        texture_schema=schema_path,
        grle_schema=tmp_path / "grle.json",
        output_dir=tmp_path / "out",
    )
    project = Project("test", MapParams(size=SIZE), paths, GenerationSettings())

    paths.info_layers_dir.mkdir(parents=True)
    # el cuadrado toca el borde: con border=10 se recorta a x,y ≥ 10
    with open(paths.textures_json, "w", encoding="utf-8") as f:
        json.dump({"fields": [square(0, 0, 110, 110)]}, f)
    paths.map_dir.mkdir(parents=True)
    paths.map_i3d.write_text(MINIMAL_I3D, encoding="utf-8")

    stats = FieldsWriter(project).run()
    assert stats.added == 1

    root = ET.parse(paths.map_i3d).getroot()
    field1 = root.find(".//TransformGroup[@name='field1']")
    assert field1 is not None
    # border=10 del schema aplicado: ningún punto por debajo de 10−100 = −90
    pp = field1.find("TransformGroup[@name='polygonPoints']")
    assert pp is not None
    cx, _, cy = (int(v) for v in field1.get("translation").split())
    for point in pp:
        rx, _, ry = (int(v) for v in point.get("translation").split())
        assert cx + rx >= -90 and cy + ry >= -90


def test_field_layer_border_from_repo_schema() -> None:
    assert field_layer_border(REPO_ROOT / "config" / "texture_schema.json") == 10
