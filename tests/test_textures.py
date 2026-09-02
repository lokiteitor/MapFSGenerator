"""Tests de la Fase 3: schema de texturas, rasterizador y motor de dibujo.

Cubren los casos del plan (§53 del doc madre): solape de 2 texturas con
prioridades, línea con width (width = RADIO del buffer), más el contrato de
``info_layers/textures.json``, fields_padding, dissolve determinista, borders
y máscaras procedural.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from mapforge.osm.projection import MapProjection  # noqa: E402
from mapforge.project import MapParams, Project, ProjectPaths  # noqa: E402
from mapforge.settings import GenerationSettings  # noqa: E402
from mapforge.textures import (  # noqa: E402
    Layer,
    TextureEngine,
    get_base_layer,
    layers_by_priority,
    load_texture_schema,
    rasterize,
)

# Centro arbitrario para los mapas sintéticos de test.
LAT, LON = 43.0, -95.0
SIZE = 128

OSM_HEADER = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    '<osm version="0.6" generator="test">\n'
)


# ------------------------------------------------------------------ helpers


def pixel_to_latlon(x: float, y: float, proj: MapProjection) -> tuple[float, float]:
    """Inversa de latlon_to_pixel: centro del píxel (x, y) → (lat, lon)."""
    north, south, east, west = proj.bbox
    lon = west + ((x + 0.5) / proj.image_size) * (east - west)
    lat = north + ((y + 0.5) / proj.image_size) * (south - north)
    return lat, lon


def write_osm(path: Path, ways: list[dict], proj: MapProjection) -> None:
    """Escribe un OSM sintético. Cada way: {points: [(x,y)px], tags: {},
    closed: bool}."""
    node_id = 1
    parts: list[str] = [OSM_HEADER]
    way_parts: list[str] = []
    for way_index, way in enumerate(ways, start=1):
        refs = []
        for x, y in way["points"]:
            lat, lon = pixel_to_latlon(x, y, proj)
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


def make_project(
    tmp_path: Path,
    schema: list[dict],
    size: int = SIZE,
    fields_padding: float = 0.0,
    dissolve: bool = False,
    seed: int = 42,
) -> Project:
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
    params = MapParams(size=size, rotation=0, latitude=LAT, longitude=LON)
    settings = GenerationSettings()
    settings.seed = seed
    settings.texture.dissolve = dissolve
    settings.texture.fields_padding = fields_padding
    return Project("test", params, paths, settings)


def read_weight(project: Project, filename: str) -> np.ndarray:
    img = cv2.imread(str(project.paths.map_data_dir / filename), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"no existe {filename}"
    return img


def square(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ------------------------------------------------------------------- schema


def test_load_texture_schema_golden() -> None:
    layers = load_texture_schema(REPO_ROOT / "config" / "texture_schema.json")
    assert len(layers) == 43
    by_name = {layer.name: layer for layer in layers}

    mud_dark = by_name["mudDark"]
    assert mud_dark.priority == 4
    assert mud_dark.info_layer == "fields"
    assert mud_dark.usage == "field"
    assert mud_dark.procedural == ["PG_acres"]
    assert mud_dark.border == 10
    assert mud_dark.tags == {"landuse": ["farmland", "meadow"]}

    base = get_base_layer(layers)
    assert base is not None and base.name == "grass"


def test_layer_weight_filenames() -> None:
    layer = Layer(name="mudDark", count=2)
    assert layer.weight_filenames() == ["mudDark01_weight.png", "mudDark02_weight.png"]
    assert layer.path("/w").name == "mudDark01_weight.png"
    assert layer.path_preview("/w").name == "mudDark01_weight_preview.png"

    excluded = Layer(name="forestRockRoots", count=2, exclude_weight=True)
    assert excluded.weight_filenames() == ["forestRockRoots01.png", "forestRockRoots02.png"]
    assert excluded.path("/w").name == "forestRockRoots01.png"

    single = Layer(name="solo", count=0)
    assert single.weight_filenames() == ["solo_weight.png"]
    assert single.path("/w").name == "solo_weight.png"


def test_layers_by_priority_order() -> None:
    layers = [
        Layer(name="base", count=1, priority=0),
        Layer(name="p5", count=1, priority=5),
        Layer(name="none_a", count=1),
        Layer(name="p2", count=1, priority=2),
        Layer(name="none_b", count=1),
        Layer(name="p5_bis", count=1, priority=5),
    ]
    ordered = [layer.name for layer in layers_by_priority(layers)]
    # None primero (orden de schema), luego descendente con empates estables.
    assert ordered == ["none_a", "none_b", "p5", "p5_bis", "p2", "base"]


# --------------------------------------------------------------- rasterizer


def test_rasterize_line_width_is_radius() -> None:
    """width = RADIO del buffer → ancho total ≈ 2×width (§S3)."""
    mask = rasterize(LineString([(20, 64), (100, 64)]), (SIZE, SIZE), width=4)
    column = mask[:, 60]
    filled = int(np.count_nonzero(column))
    assert 2 * 4 <= filled <= 2 * 4 + 2
    assert column[64] == 255 and column[61] == 255 and column[67] == 255
    assert column[58] == 0 and column[70] == 0


def test_rasterize_line_without_width_is_empty() -> None:
    """Sin width → buffer(0) colapsa y no se dibuja nada (paridad 1.8)."""
    mask = rasterize(LineString([(10, 10), (100, 100)]), (SIZE, SIZE))
    assert not mask.any()


def test_rasterize_polygon_ignores_holes() -> None:
    """1.8 solo rasteriza el anillo exterior (los agujeros se ignoran)."""
    poly = Polygon(square(10, 10, 60, 60), [square(30, 30, 40, 40)])
    mask = rasterize(poly, (SIZE, SIZE))
    assert mask[35, 35] == 255  # el agujero queda relleno, como en 1.8


def test_rasterize_point_with_width() -> None:
    mask = rasterize(Point(64, 64), (SIZE, SIZE), width=5)
    assert mask[64, 64] == 255
    assert mask[64, 60] == 255
    assert mask[64, 55] == 0


# ------------------------------------------------------------------- engine


def test_two_layers_overlap_priority(tmp_path: Path) -> None:
    """§53: solape de 2 texturas — la de mayor prioridad se dibuja primero y
    reclama el solape; la otra queda recortada; la base rellena el resto."""
    schema = [
        {"name": "alta", "count": 1, "priority": 2, "tags": {"landuse": "residential"}},
        {"name": "baja", "count": 1, "priority": 1, "tags": {"landuse": "industrial"}},
        {"name": "base", "count": 1, "priority": 0, "tags": {"natural": "grassland"}},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [
            {"points": square(30, 30, 70, 70), "tags": {"landuse": "industrial"}, "closed": True},
            {"points": square(10, 10, 50, 50), "tags": {"landuse": "residential"}, "closed": True},
        ],
        proj,
    )
    TextureEngine(project).run()

    alta = read_weight(project, "alta01_weight.png")
    baja = read_weight(project, "baja01_weight.png")
    base = read_weight(project, "base01_weight.png")

    # Zona solo de alta.
    assert alta[20, 20] == 255 and baja[20, 20] == 0 and base[20, 20] == 0
    # Solape: gana la prioridad más alta (primer claim).
    assert alta[40, 40] == 255 and baja[40, 40] == 0 and base[40, 40] == 0
    # Zona solo de baja.
    assert alta[60, 60] == 0 and baja[60, 60] == 255 and base[60, 60] == 0
    # Resto: base = NOT cumulative.
    assert alta[100, 100] == 0 and baja[100, 100] == 0 and base[100, 100] == 255
    # Exclusividad: las máscaras no se solapan y cubren todo el mapa.
    overlap = (alta > 0) & (baja > 0)
    assert not overlap.any()
    assert ((alta > 0) | (baja > 0) | (base > 0)).all()


def test_engine_line_with_width(tmp_path: Path) -> None:
    """§53: una línea OSM con width=4 produce un pasillo de ~2×width px."""
    schema = [
        {
            "name": "camino",
            "count": 1,
            "priority": 1,
            "width": 4,
            "tags": {"highway": "track"},
            "info_layer": "roads",
        },
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [{"points": [(20, 64), (100, 64)], "tags": {"highway": "track"}}],
        proj,
    )
    TextureEngine(project).run()

    camino = read_weight(project, "camino01_weight.png")
    column = camino[:, 60]
    filled = int(np.count_nonzero(column))
    assert 2 * 4 <= filled <= 2 * 4 + 2, f"ancho de línea inesperado: {filled}"
    assert camino[64, 60] == 255
    assert camino[64, 10] == 0  # fuera del tramo

    # Contrato textures.json: roads_polylines con points + tags (str del dict).
    data = json.loads(project.paths.textures_json.read_text(encoding="utf-8"))
    polylines = data["roads_polylines"]
    assert len(polylines) == 1
    assert polylines[0]["tags"] == str({"highway": "track"})
    assert polylines[0]["points"] == [[20, 64], [100, 64]]
    # Y el polígono bufferizado va a la lista "roads".
    assert len(data["roads"]) == 1


def test_fields_padding_and_textures_json(tmp_path: Path) -> None:
    """fields_padding = buffer(-padding) antes de rasterizar y de volcar el
    polígono a textures.json (fields en píxeles)."""
    schema = [
        {
            "name": "campo",
            "count": 1,
            "priority": 4,
            "tags": {"landuse": "farmland"},
            "info_layer": "fields",
        },
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema, fields_padding=3.0)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [{"points": square(20, 20, 80, 80), "tags": {"landuse": "farmland"}, "closed": True}],
        proj,
    )
    TextureEngine(project).run()

    campo = read_weight(project, "campo01_weight.png")
    assert campo[50, 50] == 255
    assert campo[21, 21] == 0  # el marco de 3 px se encogió

    data = json.loads(project.paths.textures_json.read_text(encoding="utf-8"))
    fields = data["fields"]
    assert len(fields) == 1
    xs = [pt[0] for pt in fields[0]]
    ys = [pt[1] for pt in fields[0]]
    assert min(xs) == 23 and max(xs) == 77
    assert min(ys) == 23 and max(ys) == 77
    # El polígono guardado repite el punto de cierre (paridad con Maps4FS).
    assert fields[0][0] == fields[0][-1]


def test_polygon_with_less_than_3_points_skipped(tmp_path: Path) -> None:
    """Un way de 2 nodos sin width bufferiza a vacío y se salta sin romper."""
    schema = [
        {"name": "linea", "count": 1, "priority": 1, "tags": {"highway": "path"}},
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [{"points": [(10, 10), (100, 100)], "tags": {"highway": "path"}}],
        proj,
    )
    TextureEngine(project).run()
    linea = read_weight(project, "linea01_weight.png")
    assert not linea.any()
    base = read_weight(project, "base01_weight.png")
    assert (base == 255).all()


def test_invisible_layer_feeds_info_layer_only(tmp_path: Path) -> None:
    """Capa invisible (farmyards): alimenta textures.json pero no pinta."""
    schema = [
        {
            "name": "corral",
            "count": 1,
            "tags": {"landuse": "farmyard"},
            "info_layer": "farmyards",
            "invisible": True,
        },
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [{"points": square(30, 30, 60, 60), "tags": {"landuse": "farmyard"}, "closed": True}],
        proj,
    )
    TextureEngine(project).run()

    corral = read_weight(project, "corral01_weight.png")
    assert not corral.any()
    data = json.loads(project.paths.textures_json.read_text(encoding="utf-8"))
    assert len(data["farmyards"]) == 1


def test_dissolve_deterministic(tmp_path: Path) -> None:
    """Dissolve: reparto por píxel entre sublayers con rng(seed); la unión
    reproduce la máscara original (guardada como _preview) y es determinista."""

    def build(seed: int) -> Project:
        schema = [
            {"name": "zona", "count": 2, "priority": 1, "tags": {"landuse": "residential"}},
            {"name": "base", "count": 1, "priority": 0},
        ]
        root = tmp_path / f"seed{seed}_{build.counter}"
        build.counter += 1
        root.mkdir()
        project = make_project(root, schema, dissolve=True, seed=seed)
        proj = MapProjection(LAT, LON, SIZE)
        write_osm(
            project.paths.osm,
            [{"points": square(20, 20, 90, 90), "tags": {"landuse": "residential"}, "closed": True}],
            proj,
        )
        TextureEngine(project).run()
        return project

    build.counter = 0
    project_a = build(seed=7)
    sub1 = read_weight(project_a, "zona01_weight.png")
    sub2 = read_weight(project_a, "zona02_weight.png")
    preview = read_weight(project_a, "zona01_weight_preview.png")

    assert preview.any()
    # Sublayers disjuntas cuya unión es la máscara original.
    assert not ((sub1 > 0) & (sub2 > 0)).any()
    union = np.where((sub1 > 0) | (sub2 > 0), 255, 0).astype(np.uint8)
    assert np.array_equal(union, preview)
    # Ambas sublayers reciben píxeles (reparto aleatorio).
    assert sub1.any() and sub2.any()

    # Determinismo: misma seed → mismas sublayers; otra seed → distintas.
    project_b = build(seed=7)
    assert np.array_equal(sub1, read_weight(project_b, "zona01_weight.png"))
    project_c = build(seed=8)
    assert not np.array_equal(sub1, read_weight(project_c, "zona01_weight.png"))


def test_border_transferred_to_base(tmp_path: Path) -> None:
    """border=N: el marco de N px de la capa pasa a 0 y se transfiere a la base."""
    schema = [
        {
            "name": "campo",
            "count": 1,
            "priority": 1,
            "tags": {"landuse": "farmland"},
            "border": 10,
        },
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [{"points": square(0, 0, 127, 127), "tags": {"landuse": "farmland"}, "closed": True}],
        proj,
    )
    TextureEngine(project).run()

    campo = read_weight(project, "campo01_weight.png")
    base = read_weight(project, "base01_weight.png")
    assert campo[5, 64] == 0 and base[5, 64] == 255  # marco superior
    assert campo[64, 125] == 0 and base[64, 125] == 255  # marco derecho
    assert campo[64, 64] == 255 and base[64, 64] == 0  # interior intacto


def test_procedural_masks_and_blockmask(tmp_path: Path) -> None:
    """Capas procedural → masks/PG_*.png (copia o fusión) + BLOCKMASK vacío."""
    schema = [
        {
            "name": "a",
            "count": 1,
            "priority": 2,
            "tags": {"landuse": "residential"},
            "procedural": ["PG_test", "PG_merged"],
        },
        {
            "name": "b",
            "count": 1,
            "priority": 1,
            "tags": {"landuse": "industrial"},
            "procedural": ["PG_merged"],
        },
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [
            {"points": square(10, 10, 40, 40), "tags": {"landuse": "residential"}, "closed": True},
            {"points": square(60, 60, 90, 90), "tags": {"landuse": "industrial"}, "closed": True},
        ],
        proj,
    )
    TextureEngine(project).run()

    masks_dir = project.paths.map_data_dir / "masks"
    blockmask = cv2.imread(str(masks_dir / "BLOCKMASK.png"), cv2.IMREAD_UNCHANGED)
    assert blockmask is not None and not blockmask.any()
    assert blockmask.shape == (SIZE, SIZE)

    pg_test = cv2.imread(str(masks_dir / "PG_test.png"), cv2.IMREAD_UNCHANGED)
    a_weight = read_weight(project, "a01_weight.png")
    assert np.array_equal(pg_test, a_weight)

    pg_merged = cv2.imread(str(masks_dir / "PG_merged.png"), cv2.IMREAD_UNCHANGED)
    b_weight = read_weight(project, "b01_weight.png")
    expected = np.zeros((SIZE, SIZE), dtype=np.uint8)
    expected[a_weight == 255] = 255
    expected[b_weight == 255] = 255
    assert np.array_equal(pg_merged, expected)


def test_skip_drains(tmp_path: Path) -> None:
    """skip_drains=true (default) salta las capas usage=drain."""
    schema = [
        {
            "name": "dren",
            "count": 1,
            "priority": 1,
            "width": 4,
            "usage": "drain",
            "tags": {"waterway": "ditch"},
        },
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [{"points": [(20, 64), (100, 64)], "tags": {"waterway": "ditch"}}],
        proj,
    )
    TextureEngine(project).run()
    dren = read_weight(project, "dren01_weight.png")
    assert not dren.any()


def test_all_weights_created_in_zero(tmp_path: Path) -> None:
    """Todas las capas del schema reciben sus weight files (en cero si no
    tienen contenido), con los nombres {name}{NN}_weight.png."""
    schema = [
        {"name": "conTag", "count": 2, "priority": 1, "tags": {"landuse": "residential"}},
        {"name": "sinTag", "count": 2},
        {"name": "excluida", "count": 2, "exclude_weight": True},
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(project.paths.osm, [], proj)
    TextureEngine(project).run()

    data_dir = project.paths.map_data_dir
    expected = [
        "conTag01_weight.png",
        "conTag02_weight.png",
        "sinTag01_weight.png",
        "sinTag02_weight.png",
        "excluida01.png",
        "excluida02.png",
        "base01_weight.png",
    ]
    for filename in expected:
        img = cv2.imread(str(data_dir / filename), cv2.IMREAD_UNCHANGED)
        assert img is not None, filename
        assert img.shape == (SIZE, SIZE)
        assert img.dtype == np.uint8
    # Sin OSM que casar: el segundo sublayer y las capas sin tags quedan a 0.
    assert not read_weight(project, "conTag01_weight.png").any()
    assert not read_weight(project, "conTag02_weight.png").any()

def test_textures_json_stale_data_is_overwritten(tmp_path: Path) -> None:
    """Regenerar sobre el mismo output_dir NO debe conservar los polígonos de
    la pasada anterior.

    Maps4FS 1.8 hace ``info_layer_data.update(fichero_existente)`` (gana el
    fichero viejo); allí no se nota porque cada generación escribe en un
    directorio nuevo. Reutilizando output_dir eso dejaba fields/farmyards/
    roads rancios y descuadraba i3d, farmlands y splines respecto a las
    texturas recién dibujadas.
    """
    schema = [
        {
            "name": "campo",
            "count": 1,
            "priority": 4,
            "tags": {"landuse": "farmland"},
            "info_layer": "fields",
        },
        {"name": "base", "count": 1, "priority": 0},
    ]
    project = make_project(tmp_path, schema)
    proj = MapProjection(LAT, LON, SIZE)
    write_osm(
        project.paths.osm,
        [{"points": square(20, 20, 80, 80), "tags": {"landuse": "farmland"}, "closed": True}],
        proj,
    )

    # textures.json de una "pasada anterior" con datos que ya no existen.
    project.paths.textures_json.parent.mkdir(parents=True, exist_ok=True)
    project.paths.textures_json.write_text(
        json.dumps({"fields": [[[0, 0], [1, 0], [1, 1], [0, 0]]], "otra_clave": [1]}),
        encoding="utf-8",
    )

    TextureEngine(project).run()

    data = json.loads(project.paths.textures_json.read_text(encoding="utf-8"))
    fields = data["fields"]
    assert len(fields) == 1
    xs = [pt[0] for pt in fields[0]]
    assert min(xs) == 20 and max(xs) == 80  # el polígono de ESTA pasada
    # Las claves que esta pasada no produce sí se conservan.
    assert data["otra_clave"] == [1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
