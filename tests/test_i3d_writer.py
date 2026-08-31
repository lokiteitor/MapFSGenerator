"""Tests de la Fase 7: escritor map.i3d (transformación de un i3d sintético).

Cubren: Terrain (heightScale/lodTextureSize), DisplacementLayer
(size/maxHeight/cellSize con la regla 16384/map_size), luz sun
(lastShadowMapSplitBboxMin/Max formato 1.8), Files+ReferenceNodes del
background (ids consecutivos tras el máximo, idempotencia), map.xml
(width/height), modDesc.xml (títulos) y el ``I3dWriter.run`` completo
(encoding iso-8859-1, prefijo xsi, integración de fields sin duplicar).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mapforge.fs25.i3d_writer import (  # noqa: E402
    I3D_XML_DECLARATION,
    I3dWriter,
    add_background_to_i3d,
    update_displacement_layer,
    update_map_xml,
    update_mod_desc,
    update_sun,
    update_terrain,
    write_map_i3d,
)
from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import GenerationSettings  # noqa: E402

SIZE = 2048

#: map.i3d mínimo sintético con todos los nodos que toca el escritor.
MINIMAL_I3D = """<?xml version="1.0" encoding="iso-8859-1"?>
<i3D name="map.i3d" version="1.6" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://i3d.giants.ch/schema/i3d-1.6.xsd">
  <Asset>
    <Export program="GIANTS Editor a\xf1ejo" version="10.0.2"/>
  </Asset>
  <Files>
    <File fileId="1" filename="data/dem.png"/>
    <File fileId="1211" filename="$data/maps/trees/aspen/aspen_stage01.i3d"/>
  </Files>
  <Scene>
    <Light name="sun" translation="0 400 0" nodeId="12" type="directional" lastShadowMapSplitBboxMin="-1024,-128,-1024" lastShadowMapSplitBboxMax="1024,148,1024"/>
    <TerrainTransformGroup name="terrain" nodeId="15" heightMapId="1" patchSize="65" heightScale="255" unitsPerPixel="1" lodTextureSize="2048">
      <Layers>
        <DisplacementLayer name="terrainDisplacement" size="16384" tileSize="16" numChannels="6" cellSize="8" viewDistance="25" blendOutDistance="5" maxHeight="0.2" densityMapShaderNames="terrainDisplacementMap"/>
      </Layers>
    </TerrainTransformGroup>
    <TransformGroup name="gameplay" nodeId="35">
      <TransformGroup name="fields" translation="0 0 0" nodeId="36"/>
    </TransformGroup>
  </Scene>
  <UserAttributes>
    <UserAttribute nodeId="65">
      <Attribute name="onCreate" type="scriptCallback" value="X.onCreate"/>
    </UserAttribute>
  </UserAttributes>
</i3D>
"""

MINIMAL_MAP_XML = """<?xml version="1.0" encoding="utf-8" standalone="no" ?>
<map width="2048" height="2048" imageFilename="map/overview.png">
    <filename>map/map.i3d</filename>
</map>
"""

MINIMAL_MODDESC = """<?xml version="1.0" encoding="utf-8" standalone="no"?>
<modDesc descVersion="92">
    <title>
        <en>Generated in MAPS4FS</en>
    </title>
    <maps>
        <map id="MAPS4S" configFilename="map/map.xml">
            <title>
                <en>Generated in MAPS4FS</en>
            </title>
        </map>
    </maps>
</modDesc>
"""


def parse_minimal() -> ET.Element:
    return ET.fromstring(MINIMAL_I3D)


def square(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ------------------------------------------------------------------ terrain


def test_update_terrain_sets_height_scale_and_lod() -> None:
    root = parse_minimal()
    data = update_terrain(root, map_size=8192, height_scale=300)
    terrain = root.find(".//Scene/TerrainTransformGroup")
    assert terrain.get("heightScale") == "300"
    assert terrain.get("lodTextureSize") == "8192"
    assert data == {"lodTextureSize": "8192", "heightScale": "300"}


def test_update_terrain_without_height_scale_keeps_template_value() -> None:
    root = parse_minimal()
    data = update_terrain(root, map_size=4096, height_scale=None)
    terrain = root.find(".//Scene/TerrainTransformGroup")
    assert terrain.get("heightScale") == "255"  # el del template, intacto
    assert terrain.get("lodTextureSize") == "4096"
    assert "heightScale" not in data


def test_update_terrain_missing_node_raises() -> None:
    root = ET.fromstring("<i3D><Scene/></i3D>")
    with pytest.raises(ValueError):
        update_terrain(root, map_size=2048, height_scale=255)


# ------------------------------------------------------------ displacement


def test_update_displacement_layer_golden_rule() -> None:
    """map_size 8192 → size 65536, cellSize 2 (valores exactos del artefacto)."""
    root = parse_minimal()
    data = update_displacement_layer(root, map_size=8192, max_height=0.2)
    layer = root.find(".//Scene/TerrainTransformGroup/Layers/DisplacementLayer")
    assert layer.get("size") == "65536"
    assert layer.get("cellSize") == "2"
    assert layer.get("maxHeight") == "0.2"
    # el resto de atributos del template no se tocan
    assert layer.get("tileSize") == "16"
    assert layer.get("numChannels") == "6"
    assert data == {"size": "65536", "maxHeight": "0.2", "cellSize": "2"}


def test_update_displacement_layer_template_size_is_fixed_point() -> None:
    """map_size 2048 (tamaño del template) → size 16384, cellSize 8 (idénticos
    al template limpio: la regla reproduce el par conocido)."""
    root = parse_minimal()
    update_displacement_layer(root, map_size=2048, max_height=0.2)
    layer = root.find(".//DisplacementLayer")
    assert layer.get("size") == "16384"
    assert layer.get("cellSize") == "8"


def test_update_displacement_layer_configurable() -> None:
    root = parse_minimal()
    update_displacement_layer(
        root, map_size=4096, max_height=0.5, size_factor=4, cell_size_base=10000
    )
    layer = root.find(".//DisplacementLayer")
    assert layer.get("size") == "16384"  # 4096 × 4
    assert layer.get("maxHeight") == "0.5"
    assert layer.get("cellSize") == "2.44140625"  # 10000/4096, sin redondeo


# ------------------------------------------------------------------- sun


def test_update_sun_bbox_exact_format() -> None:
    root = parse_minimal()
    data = update_sun(root, map_size=8192)
    sun = root.find(".//Scene/Light[@name='sun']")
    # formato exacto de 1.8: sin espacios, -128/148 fijos
    assert sun.get("lastShadowMapSplitBboxMin") == "-4096,-128,-4096"
    assert sun.get("lastShadowMapSplitBboxMax") == "4096,148,4096"
    assert data["lastShadowMapSplitBboxMin"] == "-4096,-128,-4096"


# -------------------------------------------------------------- background


def test_add_background_files_and_reference_nodes() -> None:
    root = parse_minimal()
    parts = [
        {"name": "background_terrain_part_01", "i3d": "assets/background/background_terrain_part_01.i3d"},
        {"name": "background_terrain_part_02", "i3d": "assets/background/background_terrain_part_02.i3d"},
    ]
    added = add_background_to_i3d(root, parts)
    assert len(added) == 2

    files = {f.get("fileId"): f.get("filename") for f in root.iter("File")}
    # fileIds consecutivos tras el máximo existente (1211)
    assert files["1212"] == "../assets/background/background_terrain_part_01.i3d"
    assert files["1213"] == "../assets/background/background_terrain_part_02.i3d"

    scene = root.find("Scene")
    refs = scene.findall("ReferenceNode")
    assert [r.get("name") for r in refs] == [
        "background_terrain_part_01",
        "background_terrain_part_02",
    ]
    # referenceId = fileId de su <File>; nodeId tras el máximo del documento (65)
    assert [r.get("referenceId") for r in refs] == ["1212", "1213"]
    assert [r.get("nodeId") for r in refs] == ["66", "67"]
    # en la raíz de Scene, sin translation propia (patrón del artefacto)
    assert all(r.get("translation") is None for r in refs)


def test_add_background_is_idempotent() -> None:
    root = parse_minimal()
    parts = [
        {"name": "background_terrain_part_01", "i3d": "assets/background/background_terrain_part_01.i3d"}
    ]
    assert len(add_background_to_i3d(root, parts)) == 1
    assert len(add_background_to_i3d(root, parts)) == 0
    assert len(root.find("Scene").findall("ReferenceNode")) == 1


def test_add_background_empty_parts_no_op() -> None:
    root = parse_minimal()
    assert add_background_to_i3d(root, []) == []
    assert len(list(root.iter("File"))) == 2


# --------------------------------------------------------- map.xml/modDesc


def test_update_map_xml(tmp_path: Path) -> None:
    path = tmp_path / "map.xml"
    path.write_text(MINIMAL_MAP_XML, encoding="utf-8")
    data = update_map_xml(path, 8192)
    assert data == {"width": "8192", "height": "8192"}
    root = ET.parse(path).getroot()
    assert root.get("width") == "8192" and root.get("height") == "8192"
    # el resto de atributos e hijos sobreviven
    assert root.get("imageFilename") == "map/overview.png"
    assert root.find("filename").text == "map/map.i3d"


def test_update_mod_desc_sets_both_titles(tmp_path: Path) -> None:
    path = tmp_path / "modDesc.xml"
    path.write_text(MINIMAL_MODDESC, encoding="utf-8")
    updated = update_mod_desc(path, "Valle Bonito")
    assert updated == 2  # título raíz + título de maps/map
    root = ET.parse(path).getroot()
    titles = [en.text for en in root.iter("en")]
    assert titles == ["Valle Bonito", "Valle Bonito"]


# ----------------------------------------------------------- escritura i3d


def test_write_map_i3d_preserves_encoding_and_xsi(tmp_path: Path) -> None:
    tree = ET.ElementTree(parse_minimal())
    out = tmp_path / "map.i3d"
    write_map_i3d(tree, out)

    raw = out.read_bytes()
    assert raw.startswith(I3D_XML_DECLARATION.encode("ascii"))
    # 'añejo' codificado en latin-1 (0xF1), no utf-8
    assert b"a\xf1ejo" in raw
    text = raw.decode("iso-8859-1")
    assert 'xsi:noNamespaceSchemaLocation="http://i3d.giants.ch/schema/i3d-1.6.xsd"' in text
    # y el resultado se puede volver a parsear con su declaración
    assert ET.parse(out).getroot().tag == "i3D"


# ------------------------------------------------------------- I3dWriter


def make_project(tmp_path: Path, size: int = SIZE, name: str = "Mapa Test") -> Project:
    paths = ProjectPaths(
        heightmap=tmp_path / "hm.png",
        osm=tmp_path / "test.osm",
        template=tmp_path / "template",
        texture_schema=tmp_path / "schema.json",  # no existe → border 0
        grle_schema=tmp_path / "grle.json",
        output_dir=tmp_path / "out",
    )
    project = Project(name, MapParams(size=size), paths, GenerationSettings())

    paths.map_dir.mkdir(parents=True)
    paths.map_i3d.write_bytes(MINIMAL_I3D.encode("iso-8859-1"))
    paths.map_xml.write_text(MINIMAL_MAP_XML, encoding="utf-8")
    paths.mod_desc.write_text(MINIMAL_MODDESC, encoding="utf-8")
    return project


def test_writer_run_end_to_end(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    project.height_scale = 412

    # fields de la Fase 3 (contrato textures.json)
    project.paths.info_layers_dir.mkdir(parents=True)
    with open(project.paths.textures_json, "w", encoding="utf-8") as f:
        json.dump({"fields": [square(100, 100, 300, 300)]}, f)

    # partes del background de la Fase 8 (descubrimiento por disco)
    assets = project.paths.output_dir / "assets" / "background"
    assets.mkdir(parents=True)
    (assets / "background_terrain_part_01.i3d").write_text("<i3D/>")
    (assets / "background_terrain_part_02.i3d").write_text("<i3D/>")

    info = I3dWriter(project).run()

    # --- map.i3d ---
    raw = project.paths.map_i3d.read_bytes()
    assert raw.startswith(I3D_XML_DECLARATION.encode("ascii"))
    root = ET.parse(project.paths.map_i3d).getroot()

    terrain = root.find(".//Scene/TerrainTransformGroup")
    assert terrain.get("heightScale") == "412"
    assert terrain.get("lodTextureSize") == str(SIZE)

    layer = root.find(".//DisplacementLayer")
    assert layer.get("size") == str(SIZE * 8)
    assert layer.get("cellSize") == str(16384 // SIZE)
    assert layer.get("maxHeight") == "0.2"

    sun = root.find(".//Scene/Light[@name='sun']")
    assert sun.get("lastShadowMapSplitBboxMin") == f"-{SIZE // 2},-128,-{SIZE // 2}"
    assert sun.get("lastShadowMapSplitBboxMax") == f"{SIZE // 2},148,{SIZE // 2}"

    # fields integrados vía mapforge.fields
    fields_node = root.find(".//TransformGroup[@name='fields']")
    assert len(fields_node) == 1
    assert fields_node[0].get("name") == "field1"
    assert info["fields"]["status"] == "added" and info["fields"]["added"] == 1

    # background: File + ReferenceNode por parte
    files = {f.get("fileId"): f.get("filename") for f in root.iter("File")}
    assert "../assets/background/background_terrain_part_01.i3d" in files.values()
    refs = root.find("Scene").findall("ReferenceNode")
    assert [r.get("name") for r in refs] == [
        "background_terrain_part_01",
        "background_terrain_part_02",
    ]
    for ref in refs:
        assert files[ref.get("referenceId")].endswith(f"{ref.get('name')}.i3d")
    assert len(info["background"]) == 2

    # --- map.xml y modDesc.xml ---
    map_root = ET.parse(project.paths.map_xml).getroot()
    assert map_root.get("width") == str(SIZE) and map_root.get("height") == str(SIZE)

    titles = [en.text for en in ET.parse(project.paths.mod_desc).getroot().iter("en")]
    assert titles == ["Mapa Test", "Mapa Test"]


def test_writer_run_twice_does_not_duplicate(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    project.height_scale = 255

    project.paths.info_layers_dir.mkdir(parents=True)
    with open(project.paths.textures_json, "w", encoding="utf-8") as f:
        json.dump({"fields": [square(100, 100, 300, 300)]}, f)

    assets = project.paths.output_dir / "assets" / "background"
    assets.mkdir(parents=True)
    (assets / "background_terrain_part_01.i3d").write_text("<i3D/>")

    I3dWriter(project).run()
    info2 = I3dWriter(project).run()

    root = ET.parse(project.paths.map_i3d).getroot()
    assert len(root.find(".//TransformGroup[@name='fields']")) == 1
    assert len(root.find("Scene").findall("ReferenceNode")) == 1
    assert info2["fields"]["status"] == "skipped"
    assert info2["background"] == []


def test_writer_run_without_optional_inputs(tmp_path: Path) -> None:
    """Sin textures.json ni assets/background: solo terrain/sun/map.xml/modDesc."""
    project = make_project(tmp_path)
    project.height_scale = None  # DEM no ejecutado: heightScale intacto

    info = I3dWriter(project).run()

    root = ET.parse(project.paths.map_i3d).getroot()
    assert root.find(".//Scene/TerrainTransformGroup").get("heightScale") == "255"
    assert info["fields"]["status"] == "skipped"
    assert info["background"] == []
    assert len(root.find(".//TransformGroup[@name='fields']")) == 0
