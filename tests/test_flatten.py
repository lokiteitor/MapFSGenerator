"""Tests del aplanado de terreno (terrain/flatten.py).

Cubre ``background.flatten_roads`` (réplica funcional de la feature 3.x) y
``dem.flatten_farmyard`` (extensión propia de MapForge). No hay golden para
comparar el resultado píxel a píxel — el criterio es geométrico —, así que cada
test aísla una propiedad comprobable sobre DEM sintéticos: núcleo plano,
exterior intacto, transición monótona y sin escalón, suavizado longitudinal,
composición en los cruces y determinismo.

Convenio de los DEM sintéticos: ``dem[y, x]`` con un gradiente en Y (el terreno
baja/sube hacia el sur) de ``SLOPE`` unidades uint16 por píxel, que es lo que
hace visible cualquier corte en el margen de una vía horizontal.
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
from mapforge.settings import (  # noqa: E402
    BackgroundSettings,
    DEMSettings,
    GenerationSettings,
)
from mapforge.terrain.dem import DemPipeline  # noqa: E402
from mapforge.terrain.flatten import (  # noqa: E402
    DEFAULT_FEATHER_RATIO,
    flatten_farmyards,
    flatten_roads,
    flatten_terrain,
    load_flatten_geometry,
    road_widths_from_schema,
)
from mapforge.textures.schema import Layer  # noqa: E402

SIZE = 200
SLOPE = 10  # unidades uint16 por píxel de gradiente del terreno sintético
TAGS = "{'highway': ['secondary']}"


def sloped_dem(size: int = SIZE, slope: int = SLOPE) -> np.ndarray:
    """DEM con una rampa constante en Y: ``dem[y, x] = 1000 + slope × y``."""
    column = 1000 + slope * np.arange(size, dtype=np.int64)
    return np.tile(column.reshape(-1, 1), (1, size)).astype(np.uint16)


def horizontal_road(y: int = 100, tags: str = TAGS) -> list[dict]:
    """Una vía recta de este a oeste a la altura ``y``, sin tocar los bordes."""
    return [{"points": [[10, y], [SIZE - 10, y]], "tags": tags}]


# ------------------------------------------------------------ geometría base


def test_nucleo_plano_y_radio_del_corredor() -> None:
    """El núcleo es plano a ±width y nada cambia más allá de width+feather."""
    dem = sloped_dem()
    out, stats = flatten_roads(dem, horizontal_road(), {TAGS: 4}, feather=4, smooth=0)

    assert stats["applied"] == 1 and stats["skipped"] == 0
    # Núcleo: constante en la sección transversal (y = 96..104 = 100 ± width).
    core = out[96:105, 100]
    assert core.min() == core.max()
    # Y su altura es la del eje (el terreno bajo el eje, sin suavizar).
    assert int(core[0]) == int(dem[100, 100])

    delta = out.astype(np.int64) - dem.astype(np.int64)
    rows = np.nonzero(delta.any(axis=1))[0]
    # width + feather = 8 → filas 92..108; el borde exacto puede quedar en 0 por
    # redondeo, así que se comprueba la cota superior.
    assert rows.min() >= 100 - 8 and rows.max() <= 100 + 8


def test_exterior_intacto() -> None:
    """Fuera del corredor el DEM es idéntico bit a bit."""
    dem = sloped_dem()
    out, _ = flatten_roads(dem, horizontal_road(), {TAGS: 4}, feather=4, smooth=0)
    np.testing.assert_array_equal(out[:90], dem[:90])
    np.testing.assert_array_equal(out[112:], dem[112:])


def test_transicion_monotona_y_sin_escalon() -> None:
    """El talud es monótono y no introduce ningún salto brusco.

    Es el test que codifica el requisito: el terreno debe integrarse en el
    margen. Un aplanado con corte duro dejaría un escalón de ``width × SLOPE``
    (40 unidades) en un solo píxel; con el feather el salto máximo se queda en
    unas pocas veces el gradiente propio del terreno.
    """
    dem = sloped_dem()
    out, _ = flatten_roads(dem, horizontal_road(), {TAGS: 4}, feather=8, smooth=0)

    profile = out[80:121, 100].astype(np.int64)
    steps = np.diff(profile)
    assert (steps >= 0).all(), "el perfil transversal debe ser monótono"
    # Sin feather el salto sería de 40 unidades de golpe; con él se reparte.
    assert steps.max() < 3 * SLOPE
    assert steps.max() < 4 * SLOPE  # cota holgada y explícita del "sin corte"


def test_feather_mas_ancho_suaviza_el_talud() -> None:
    """A más feather, menor pendiente máxima del talud (justifica el default)."""
    dem = sloped_dem()
    peaks = []
    for feather in (4, 8, 12):
        out, _ = flatten_roads(
            dem, horizontal_road(), {TAGS: 4}, feather=feather, smooth=0
        )
        profile = out[70:131, 100].astype(np.int64)
        peaks.append(int(np.abs(np.diff(profile)).max()))
    assert peaks[0] > peaks[1] > peaks[2]
    # El default (feather = 2 × width) es el punto donde deja de compensar
    # ensanchar: de 4→8 se gana bastante más que de 8→12.
    assert (peaks[0] - peaks[1]) > (peaks[1] - peaks[2])


def test_feather_por_defecto_es_proporcional_al_ancho() -> None:
    """``feather=None`` ⇒ ``DEFAULT_FEATHER_RATIO × width``."""
    dem = sloped_dem()
    auto, _ = flatten_roads(dem, horizontal_road(), {TAGS: 4}, feather=None, smooth=0)
    explicit, _ = flatten_roads(
        dem,
        horizontal_road(),
        {TAGS: 4},
        feather=4 * DEFAULT_FEATHER_RATIO,
        smooth=0,
    )
    np.testing.assert_array_equal(auto, explicit)


def test_feather_cero_es_corte_duro() -> None:
    """``feather=0`` deja el escalón completo: es la referencia a evitar."""
    dem = sloped_dem()
    out, _ = flatten_roads(dem, horizontal_road(), {TAGS: 4}, feather=0, smooth=0)
    profile = out[90:111, 100].astype(np.int64)
    # El borde del núcleo cae de golpe los width × SLOPE = 40 acumulados.
    assert np.abs(np.diff(profile)).max() >= 4 * SLOPE


# ------------------------------------------------ suavizado longitudinal


def test_suavizado_longitudinal_elimina_bache() -> None:
    """Un bache puntual en el eje desaparece del perfil de la calzada."""
    dem = np.full((SIZE, SIZE), 2000, dtype=np.uint16)
    dem[98:103, 90:95] = 2600  # bache de 600 unidades sobre el eje
    road = horizontal_road()

    sin_suavizar, _ = flatten_roads(dem, road, {TAGS: 4}, feather=4, smooth=0)
    suavizado, _ = flatten_roads(dem, road, {TAGS: 4}, feather=4, smooth=25)

    eje_sin = sin_suavizar[100, 85:100].astype(np.int64)
    eje_con = suavizado[100, 85:100].astype(np.int64)
    assert eje_sin.max() > 2500, "sin suavizado el bache sigue en la calzada"
    assert eje_con.max() < 2300, "con suavizado el bache queda muy amortiguado"
    # Y el perfil longitudinal resultante es mucho más suave.
    assert np.abs(np.diff(eje_con)).max() < np.abs(np.diff(eje_sin)).max()


def test_suavizado_conserva_la_pendiente_real() -> None:
    """Una rampa constante sobrevive intacta a la media móvil."""
    # Rampa en X para que el eje de una vía horizontal la recorra de frente.
    row = 1000 + SLOPE * np.arange(SIZE, dtype=np.int64)
    dem = np.tile(row.reshape(1, -1), (SIZE, 1)).astype(np.uint16)
    out, _ = flatten_roads(dem, horizontal_road(), {TAGS: 4}, feather=4, smooth=25)
    eje = out[100, 60:140].astype(np.int64)
    np.testing.assert_array_equal(np.diff(eje), np.full(eje.size - 1, SLOPE))


# ------------------------------------------------------ cruces y composición


def test_cruce_de_vias_sin_discontinuidad() -> None:
    """En la intersección los dos corredores se promedian, no se pisan."""
    dem = sloped_dem()
    roads = [
        {"points": [[10, 100], [SIZE - 10, 100]], "tags": TAGS},
        {"points": [[100, 10], [100, SIZE - 10]], "tags": TAGS},
    ]
    out, stats = flatten_roads(dem, roads, {TAGS: 4}, feather=8, smooth=0)
    assert stats["applied"] == 2

    # Un corte que atraviesa el cruce no debe tener saltos mayores que los del
    # talud de una vía aislada.
    aislada, _ = flatten_roads(
        dem, horizontal_road(), {TAGS: 4}, feather=8, smooth=0
    )
    salto_cruce = int(np.abs(np.diff(out[70:131, 100].astype(np.int64))).max())
    salto_aislada = int(np.abs(np.diff(aislada[70:131, 100].astype(np.int64))).max())
    assert salto_cruce <= salto_aislada


def test_determinismo_y_orden_independiente() -> None:
    """Misma entrada ⇒ misma salida, y el orden de las vías no influye."""
    dem = sloped_dem()
    roads = [
        {"points": [[10, 100], [SIZE - 10, 100]], "tags": TAGS},
        {"points": [[100, 10], [100, SIZE - 10]], "tags": TAGS},
        {"points": [[20, 40], [180, 160]], "tags": TAGS},
    ]
    a, _ = flatten_roads(dem, roads, {TAGS: 4}, smooth=25)
    b, _ = flatten_roads(dem, roads, {TAGS: 4}, smooth=25)
    c, _ = flatten_roads(dem, list(reversed(roads)), {TAGS: 4}, smooth=25)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_via_diagonal_tiene_seccion_plana() -> None:
    """La sección transversal también es plana en una vía diagonal."""
    dem = sloped_dem()
    roads = [{"points": [[20, 20], [180, 180]], "tags": TAGS}]
    out, stats = flatten_roads(dem, roads, {TAGS: 4}, feather=4, smooth=0)
    assert stats["applied"] == 1
    # Perpendicular a la diagonal en el centro: (100,100) ± k·(1,-1)/√2.
    ks = np.arange(-3, 4)
    values = [int(out[100 + k, 100 - k]) for k in ks]
    assert max(values) - min(values) <= 1  # plano salvo redondeo del Voronoi


# --------------------------------------------------------------- robustez


def test_entradas_vacias_son_identidad() -> None:
    """Sin geometría el DEM sale intacto (y como copia, no como alias)."""
    dem = sloped_dem()
    out, stats = flatten_roads(dem, [], {TAGS: 4})
    np.testing.assert_array_equal(out, dem)
    assert out is not dem
    assert stats["input"] == 0 and stats["applied"] == 0

    out, stats = flatten_farmyards(dem, [])
    np.testing.assert_array_equal(out, dem)
    assert stats["input"] == 0


def test_tags_desconocidos_se_descartan() -> None:
    """Una vía cuyo tag no está en el schema no aplana nada, pero no rompe."""
    dem = sloped_dem()
    out, stats = flatten_roads(dem, horizontal_road(tags="{'highway': ['x']}"), {})
    np.testing.assert_array_equal(out, dem)
    assert stats["skipped"] == 1 and stats["applied"] == 0
    assert stats["unknown_tags"] == ["{'highway': ['x']}"]


def test_default_width_rescata_tags_desconocidos() -> None:
    """Con ``default_width`` sí se aplana pese a no estar en el schema."""
    dem = sloped_dem()
    out, stats = flatten_roads(
        dem, horizontal_road(tags="{'x': 1}"), {}, default_width=3, smooth=0
    )
    assert stats["applied"] == 1
    assert not np.array_equal(out, dem)


def test_via_fuera_de_bounds() -> None:
    """Una vía enteramente fuera del DEM se descarta sin tocar nada."""
    dem = sloped_dem()
    roads = [{"points": [[-500, -500], [-400, -400]], "tags": TAGS}]
    out, stats = flatten_roads(dem, roads, {TAGS: 4})
    np.testing.assert_array_equal(out, dem)
    assert stats["applied"] == 0


def test_via_que_cruza_el_borde() -> None:
    """Una vía que se sale por un lado aplana solo la parte visible."""
    dem = sloped_dem()
    roads = [{"points": [[-50, 100], [SIZE + 50, 100]], "tags": TAGS}]
    out, stats = flatten_roads(dem, roads, {TAGS: 4}, feather=4, smooth=0)
    assert stats["applied"] == 1
    assert not np.array_equal(out[:, 0], dem[:, 0])  # llega hasta el borde
    np.testing.assert_array_equal(out[:80], dem[:80])


def test_polilinea_degenerada() -> None:
    """Polilíneas de 1 punto o con todos los puntos repetidos no rompen."""
    dem = sloped_dem()
    for points in ([[100, 100]], [[100, 100], [100, 100], [100, 100]]):
        out, stats = flatten_roads(
            dem, [{"points": points, "tags": TAGS}], {TAGS: 4}, smooth=0
        )
        assert out.shape == dem.shape and out.dtype == dem.dtype
        assert stats["applied"] + stats["skipped"] == 1


def test_polilinea_larga_se_trocea_sin_costuras() -> None:
    """El troceado interno (MAX_CHUNK_SAMPLES) no deja costuras en el perfil."""
    import mapforge.terrain.flatten as flatten_mod

    dem = sloped_dem()
    road = horizontal_road()
    entero, _ = flatten_roads(dem, road, {TAGS: 4}, feather=4, smooth=0)

    original = flatten_mod.MAX_CHUNK_SAMPLES
    try:
        flatten_mod.MAX_CHUNK_SAMPLES = 40  # fuerza ~5 trozos solapados
        troceado, _ = flatten_roads(dem, road, {TAGS: 4}, feather=4, smooth=0)
    finally:
        flatten_mod.MAX_CHUNK_SAMPLES = original

    # El troceado promedia solapes: se admite 1 unidad uint16 de diferencia.
    assert np.abs(
        troceado.astype(np.int64) - entero.astype(np.int64)
    ).max() <= 1


# --------------------------------------------------------------- farmyards


def square(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    """Anillo cerrado, como los que escribe el motor de texturas."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def test_farmyard_interior_a_la_media() -> None:
    """El interior queda plano exactamente en la media del área."""
    dem = sloped_dem()
    ring = square(50, 50, 150, 150)
    out, stats = flatten_farmyards(dem, [ring], feather=8)

    assert stats["applied"] == 1
    interior = out[60:141, 60:141]
    assert interior.min() == interior.max()

    mask = np.zeros(dem.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(ring, dtype=np.int32)], 255)
    esperado = float(np.round(cv2.mean(dem, mask=mask)[0]))
    assert int(interior[0, 0]) == int(esperado)


def test_farmyard_feather_monotono_y_exterior_intacto() -> None:
    """La transición sale del polígono hacia fuera, monótona y acotada."""
    dem = sloped_dem()
    out, _ = flatten_farmyards(dem, [square(50, 50, 150, 150)], feather=10)

    # Corte vertical por el borde norte del recinto.
    profile = out[35:65, 100].astype(np.int64)
    assert (np.diff(profile) >= 0).all()
    np.testing.assert_array_equal(out[:38], dem[:38])  # más allá del feather
    # Y sin escalón: el salto máximo es un múltiplo pequeño del terreno.
    assert np.abs(np.diff(profile)).max() < 12 * SLOPE


def test_farmyard_feather_cero_deja_escalon() -> None:
    """Referencia: sin feather el borde del recinto es un corte vertical."""
    dem = sloped_dem()
    out, _ = flatten_farmyards(dem, [square(50, 50, 150, 150)], feather=0)
    np.testing.assert_array_equal(out[:50], dem[:50])
    salto = int(out[50, 100]) - int(out[49, 100])
    assert abs(salto) > 20 * SLOPE


def test_farmyard_parcialmente_fuera_de_bounds() -> None:
    """Un recinto que se sale del DEM se aplana en su parte visible."""
    dem = sloped_dem()
    out, stats = flatten_farmyards(dem, [square(-50, -50, 60, 60)], feather=8)
    assert stats["applied"] == 1
    interior = out[5:55, 5:55]
    assert interior.min() == interior.max()


def test_farmyard_degenerado_se_descarta() -> None:
    """Anillos con menos de 3 puntos o fuera del DEM no aplican nada."""
    dem = sloped_dem()
    out, stats = flatten_farmyards(
        dem,
        [[[10, 10], [20, 20]], square(-500, -500, -400, -400)],
        feather=8,
    )
    np.testing.assert_array_equal(out, dem)
    assert stats["applied"] == 0 and stats["skipped"] == 2


def test_farmyard_max_relief_descarta_recintos_con_mucho_desnivel() -> None:
    """La salvaguarda evita convertir una ladera entera en una meseta.

    ``landuse=farmyard`` se usa en OSM con mucha manga ancha: en el mapa de
    validación hay recintos de 513 ha con 78 m de desnivel. Aplanarlos a su
    media no mejoraría el terreno, lo destruiría.
    """
    dem = sloped_dem()  # 10 u16/px ⇒ 100 px de recinto = 1000 u16 ≈ 3.9 m
    ring = square(20, 20, 180, 180)  # 160 px de alto ⇒ 1600 u16 ≈ 6.2 m

    descartado, stats = flatten_farmyards(dem, [ring], feather=8, max_relief=1.0)
    np.testing.assert_array_equal(descartado, dem)
    assert stats["applied"] == 0 and stats["skipped"] == 1
    assert stats["skipped_by_relief"][0]["index"] == 0
    assert stats["skipped_by_relief"][0]["relief_m"] == pytest.approx(1590 / 257, abs=0.1)

    aplicado, stats = flatten_farmyards(dem, [ring], feather=8, max_relief=100.0)
    assert stats["applied"] == 1 and not stats["skipped_by_relief"]
    assert not np.array_equal(aplicado, dem)


def test_farmyard_max_relief_none_desactiva_la_salvaguarda() -> None:
    """``max_relief=None`` (o 0) aplana todos los recintos."""
    dem = sloped_dem()
    ring = square(20, 20, 180, 180)
    for limite in (None, 0):
        out, stats = flatten_farmyards(dem, [ring], feather=8, max_relief=limite)
        assert stats["applied"] == 1, f"max_relief={limite!r}"
        assert not np.array_equal(out, dem)


def test_farmyard_max_relief_usa_la_escala_del_dem() -> None:
    """El límite es en metros: depende de ``units_per_metre`` del DEM."""
    dem = sloped_dem()
    ring = square(20, 20, 180, 180)  # ~1590 unidades uint16 de desnivel
    # Con 257 u16/m son ~6.2 m → por debajo del límite de 10 m.
    _, permisivo = flatten_farmyards(dem, [ring], max_relief=10.0, units_per_metre=257)
    # Con 1 u16/m son 1590 m → muy por encima.
    _, estricto = flatten_farmyards(dem, [ring], max_relief=10.0, units_per_metre=1)
    assert permisivo["applied"] == 1
    assert estricto["applied"] == 0


def test_farmyards_solapados_se_promedian() -> None:
    """Dos recintos que se solapan no dependen del orden de iteración."""
    dem = sloped_dem()
    rings = [square(40, 40, 110, 110), square(90, 90, 160, 160)]
    a, _ = flatten_farmyards(dem, rings, feather=6)
    b, _ = flatten_farmyards(dem, list(reversed(rings)), feather=6)
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------ orquestación


def test_orden_farmyard_luego_carretera() -> None:
    """La vía se muestrea sobre el DEM ya aplanado por el farmyard.

    Dentro del recinto la calzada debe heredar la altura plana de la era; si el
    orden fuera el contrario, la vía llevaría la altura del terreno original.
    """
    dem = sloped_dem()
    # Recinto asimétrico respecto de la vía: su media (la altura de y=80) no
    # coincide con el terreno bajo el eje (y=100), así que la diferencia entre
    # los dos órdenes posibles es observable.
    ring = square(60, 30, 140, 130)
    roads = horizontal_road(y=100)

    out, stats = flatten_terrain(
        dem,
        offset=0,
        roads=roads,
        road_widths={TAGS: 4},
        farmyards=[ring],
        roads_feather=4,
        roads_smooth=0,
        farmyard_feather=8,
    )
    assert stats["farmyards"]["applied"] == 1 and stats["roads"]["applied"] == 1

    solo_farmyard, _ = flatten_farmyards(dem, [ring], feather=8)
    altura_era = int(solo_farmyard[100, 100])
    # La calzada dentro del recinto está a la altura de la era, no a la del
    # terreno original (que ahí valía dem[100, 100]).
    assert int(out[100, 100]) == altura_era
    assert altura_era != int(dem[100, 100])


def test_flatten_terrain_sin_geometria_devuelve_copia() -> None:
    """Sin roads ni farmyards se devuelve una copia intacta y stats vacías."""
    dem = sloped_dem()
    out, stats = flatten_terrain(dem, offset=0)
    np.testing.assert_array_equal(out, dem)
    assert out is not dem
    assert stats == {}


def test_offset_desplaza_la_geometria() -> None:
    """``offset`` traslada las coordenadas del marco del mapa al del background.

    El desplazamiento se aplica a los dos ejes, que es como se pasa del marco
    del mapa (``map_size²``) al del DEM del background (``background_size²``).
    """
    dem = sloped_dem()
    sin_offset, _ = flatten_roads(
        dem,
        [{"points": [[40, 50], [160, 50]], "tags": TAGS}],
        {TAGS: 4},
        feather=4,
        smooth=0,
    )
    con_offset, _ = flatten_roads(
        dem,
        [{"points": [[10, 20], [130, 20]], "tags": TAGS}],
        {TAGS: 4},
        offset=30,
        feather=4,
        smooth=0,
    )
    np.testing.assert_array_equal(sin_offset, con_offset)


def test_road_widths_from_schema() -> None:
    """Solo las capas con ``info_layer='roads'`` y ``width`` entran al mapa."""
    layers = [
        Layer(name="asphaltDusty", count=2, tags={"highway": ["motorway"]},
              width=8, info_layer="roads"),
        Layer(name="gravelSmall", count=2, tags={"highway": ["secondary"]},
              width=4, info_layer="roads"),
        Layer(name="concrete", count=2, tags={"building": True}, width=8,
              info_layer="buildings"),
        Layer(name="sinWidth", count=2, tags={"highway": ["x"]},
              info_layer="roads"),
    ]
    widths = road_widths_from_schema(layers)
    assert widths == {
        "{'highway': ['motorway']}": 8,
        "{'highway': ['secondary']}": 4,
    }


def test_road_widths_del_schema_real() -> None:
    """El schema del repo resuelve los tres tipos de vía con sus radios."""
    from mapforge.textures.schema import load_texture_schema

    widths = road_widths_from_schema(
        load_texture_schema(REPO_ROOT / "config" / "texture_schema.json")
    )
    assert widths["{'highway': ['motorway', 'trunk', 'primary']}"] == 8
    assert widths["{'highway': ['secondary', 'tertiary', 'road', 'service']}"] == 4
    assert widths["{'highway': ['unclassified', 'residential', 'track']}"] == 2


def test_load_flatten_geometry_sin_fichero(tmp_path: Path) -> None:
    """Sin ``textures.json`` se devuelven listas vacías, no una excepción."""
    roads, widths, farmyards = load_flatten_geometry(
        tmp_path / "no-existe.json", REPO_ROOT / "config" / "texture_schema.json"
    )
    assert roads == [] and widths == {} and farmyards == []


# ----------------------------------------------------- integración con el DEM


MAP_SIZE = 256
BG_SIZE = MAP_SIZE + 2 * 2048


def make_project(
    tmp_path: Path,
    heightmap: np.ndarray,
    *,
    flatten_roads_on: bool = True,
    flatten_farmyard_on: bool = False,
    textures: dict | None = None,
) -> Project:
    """Proyecto en modo ``custom_dem`` con un ``textures.json`` sintético."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    heightmap_path = tmp_path / "heightmap.png"
    cv2.imwrite(str(heightmap_path), heightmap)

    settings = GenerationSettings()
    settings.dem = DEMSettings(
        custom_dem=True, flatten_farmyard=flatten_farmyard_on
    )
    settings.background = BackgroundSettings(
        flatten_roads=flatten_roads_on, flatten_roads_smooth=0
    )

    paths = ProjectPaths(
        heightmap=heightmap_path,
        osm=tmp_path / "custom.osm",
        template=tmp_path / "template.zip",
        texture_schema=REPO_ROOT / "config" / "texture_schema.json",
        grle_schema=tmp_path / "grle_schema.json",
        output_dir=tmp_path / "salida",
    )
    if textures is not None:
        info_layers = paths.output_dir / "info_layers"
        info_layers.mkdir(parents=True, exist_ok=True)
        (info_layers / "textures.json").write_text(
            json.dumps(textures), encoding="utf-8"
        )
    return Project(
        name="test",
        map_params=MapParams(size=MAP_SIZE),
        paths=paths,
        settings=settings,
    )


def bg_heightmap() -> np.ndarray:
    """DEM del background con rampa en Y (custom_dem: se usa tal cual)."""
    column = 2000 + 5 * np.arange(BG_SIZE, dtype=np.int64)
    return np.tile(column.reshape(-1, 1), (1, BG_SIZE)).astype(np.uint16)


PRIMARY = "{'highway': ['motorway', 'trunk', 'primary']}"


def test_pipeline_escribe_el_dem_aplanado(tmp_path: Path) -> None:
    """La etapa DEM aplana sobre el DEM completo y escribe el intermedio."""
    textures = {
        "roads_polylines": [
            {"points": [[10, 128], [MAP_SIZE - 10, 128]], "tags": PRIMARY}
        ],
        "farmyards": [],
    }
    project = make_project(tmp_path, bg_heightmap(), textures=textures)
    pipeline = DemPipeline(project)
    pipeline.run()

    flattened_path = (
        project.paths.background_dir / "not_resized_with_flattened_roads.png"
    )
    assert flattened_path.is_file()

    crudo = cv2.imread(
        str(project.paths.background_dir / "not_resized.png"), cv2.IMREAD_UNCHANGED
    )
    aplanado = cv2.imread(str(flattened_path), cv2.IMREAD_UNCHANGED)
    assert not np.array_equal(crudo, aplanado)

    # FACT del golden: crop(FULL) == not_resized_with_flattened_roads.
    full = cv2.imread(
        str(project.paths.background_dir / "FULL.png"), cv2.IMREAD_UNCHANGED
    )
    half = MAP_SIZE // 2
    centre = full.shape[0] // 2
    np.testing.assert_array_equal(
        full[centre - half : centre + half, centre - half : centre + half], aplanado
    )

    # not_resized.png (sin aplanar) y el crop crudo en memoria coinciden.
    np.testing.assert_array_equal(crudo, pipeline.dem_not_resized_raw)
    # Y dem_not_resized (lo que muestrean las splines) es el aplanado.
    np.testing.assert_array_equal(aplanado, pipeline.dem_not_resized)

    stats = pipeline.flatten_stats
    assert stats["roads"]["applied"] == 1
    assert stats["changed_pixels"] > 0


def test_pipeline_flatten_desactivado_es_identidad(tmp_path: Path) -> None:
    """Con ambos flags a false no se aplana ni se escribe el intermedio."""
    textures = {
        "roads_polylines": [
            {"points": [[10, 128], [MAP_SIZE - 10, 128]], "tags": PRIMARY}
        ],
        "farmyards": [square(40, 40, 100, 100)],
    }
    project = make_project(
        tmp_path,
        bg_heightmap(),
        flatten_roads_on=False,
        flatten_farmyard_on=False,
        textures=textures,
    )
    pipeline = DemPipeline(project)
    pipeline.run()

    assert not (
        project.paths.background_dir / "not_resized_with_flattened_roads.png"
    ).is_file()
    assert pipeline.flatten_stats == {}
    np.testing.assert_array_equal(
        pipeline.dem_not_resized, pipeline.dem_not_resized_raw
    )


def test_pipeline_sin_textures_json_no_rompe(tmp_path: Path) -> None:
    """Si falta ``textures.json`` (etapa saltada) el DEM sale sin aplanar."""
    project = make_project(tmp_path, bg_heightmap(), textures=None)
    pipeline = DemPipeline(project)
    pipeline.run()
    assert pipeline.flatten_stats == {}
    np.testing.assert_array_equal(
        pipeline.dem_not_resized, pipeline.dem_not_resized_raw
    )


def test_pipeline_farmyard_opcional(tmp_path: Path) -> None:
    """``flatten_farmyard`` aplana los recintos y por defecto está apagado."""
    textures = {"roads_polylines": [], "farmyards": [square(40, 40, 120, 120)]}

    apagado = make_project(
        tmp_path / "off", bg_heightmap(), flatten_farmyard_on=False, textures=textures
    )
    DemPipeline(apagado).run()
    assert not (
        apagado.paths.background_dir / "not_resized_with_flattened_roads.png"
    ).is_file()

    encendido = make_project(
        tmp_path / "on", bg_heightmap(), flatten_farmyard_on=True, textures=textures
    )
    pipeline = DemPipeline(encendido)
    pipeline.run()
    assert pipeline.flatten_stats["farmyards"]["applied"] == 1
    interior = pipeline.dem_not_resized[50:110, 50:110]
    assert interior.min() == interior.max()


def test_settings_por_defecto() -> None:
    """Defaults documentados: roads on, farmyard off, feather proporcional."""
    settings = GenerationSettings()
    assert settings.background.flatten_roads is True
    assert settings.background.flatten_roads_feather is None
    assert settings.background.flatten_roads_smooth == pytest.approx(25.0)
    assert settings.dem.flatten_farmyard is False
    assert settings.dem.flatten_farmyard_feather == pytest.approx(8.0)
    assert settings.dem.flatten_farmyard_max_relief == pytest.approx(10.0)
