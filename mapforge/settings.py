"""Settings de generación de MapForge.

Dataclasses espejo de los grupos de ``generation_settings.json`` del artefacto
golden (``FS25_Valle_Bonito/generation_settings.json``, Maps4FS 3.1.2) más los
campos propios de MapForge (``seed``, ``input_height_scale`` y
``background.procedural``).

Convenciones:

- Los valores por defecto son los del artefacto golden (la configuración
  validada por el análisis forense), no los defaults de Maps4FS 1.8 (§S7 del
  informe: difieren en farmland_margin, add_reversed_splines, ...).
- ``from_dict`` ignora claves desconocidas (con aviso vía logging) para poder
  cargar directamente un ``generation_settings.json`` de Maps4FS.
- ``GenerationSettings.from_dict`` acepta tanto las claves estilo Maps4FS
  (``DEMSettings``, ``BackgroundSettings``, ...) como las claves cortas de
  nuestro ``config.yaml`` (``dem``, ``background``, ...). Los grupos fuera de
  alcance del proyecto (Preprocessor, Satellite, Building) se ignoran.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mapforge.settings")

#: Escala por defecto del heightmap de entrada: metros = valor_uint16 × escala.
#: 255/65535 (= 1/257), la convención del artefacto (informe forense §C).
DEFAULT_INPUT_HEIGHT_SCALE = 255.0 / 65535.0

#: Distancia extra del background a cada lado del mapa (Parameters.BACKGROUND_DISTANCE).
BACKGROUND_DISTANCE = 2048


def _filtered_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Filtra ``data`` a los campos del dataclass ``cls``, avisando de sobrantes."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        logger.debug("%s: claves ignoradas: %s", cls.__name__, ", ".join(unknown))
    return {k: v for k, v in data.items() if k in known}


class _SettingsBase:
    """Base común: construcción desde dict y volcado a dict."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "_SettingsBase":
        data = dict(data or {})
        return cls(**_filtered_kwargs(cls, data))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[call-overload]


@dataclass
class DEMSettings(_SettingsBase):
    """Espejo de ``DEMSettings`` (pipeline S2 del informe forense).

    ``custom_dem`` (espejo del ``custom_dem`` de ``main_settings.json`` de
    Maps4FS, no de ``DEMSettings``): el heightmap de entrada YA es el DEM
    final uint16 normalizado y el pipeline S2 se omite — es el hook que
    :class:`mapforge.terrain.dem.DemPipeline` lee vía
    ``settings.dem.custom_dem`` y el modo con el que se generó el golden.

    ``flatten_farmyard`` / ``flatten_farmyard_feather`` son **extensión propia
    de MapForge**, sin equivalente en Maps4FS (ver
    ``mapforge.terrain.flatten``): aplanan el terreno dentro de cada polígono
    ``landuse=farmyard`` a la media del área, con una transición suave de
    ``flatten_farmyard_feather`` metros hacia el terreno circundante.
    ``flatten_farmyard_max_relief`` (metros, ``None`` = sin límite) descarta los
    recintos con demasiado desnivel interior: ``landuse=farmyard`` se usa en OSM
    con mucha manga ancha y aplanar un polígono de cientos de hectáreas dejaría
    una meseta artificial rodeada de un talud enorme.
    """

    custom_dem: bool = False
    adjust_terrain_to_ground_level: bool = True
    multiplier: float = 1
    minimum_height_scale: int = 255
    plateau: int = 15
    ceiling: int = 15
    water_depth: int = 15
    blur_radius: int = 3
    add_foundations: bool = False  # fuera de alcance; se conserva por espejo
    water_bank_steepness: float = 2
    # --- campos propios de MapForge ---
    flatten_farmyard: bool = False
    flatten_farmyard_feather: float = 8.0
    flatten_farmyard_max_relief: float | None = 10.0


@dataclass
class ProceduralBackgroundSettings(_SettingsBase):
    """Campos propios de MapForge: background procedural (sustituye al satélite).

    - Mesh (algoritmo S6): subsample, decimación quadric (default 25/3 activada,
      ~167k vértices como el artefacto).
    - Textura: ver :mod:`mapforge.background.texture`. Con
      ``height_texture`` (default) el color sigue el relieve del DEM en tres
      bandas — la rampa tierra→verde (``palette_low``/``palette_high``) abajo,
      ``palette_rock`` desde ``rock_height`` o en las pendientes fuertes, y
      ``palette_snow`` desde ``snow_height``. Con ``height_texture: false`` se
      cae al modo original: fBm plano sobre la rampa tierra→verde.

    Las cotas ``rock_height``/``snow_height`` van en **metros de mundo** (la
    misma escala que la Y del mesh: ``dem × z_scaling_factor``), no en valores
    del uint16. Si una cota queda por encima del techo real del terreno, esa
    banda simplemente no se dibuja y la fase avisa por log.
    """

    resize_factor: int = 8
    apply_decimation: bool = True
    decimation_percent: int = 25
    decimation_aggression: int = 3
    texture_size: int = 4096
    noise_octaves: int = 5
    palette_low: tuple[int, int, int] = (101, 84, 59)  # tierra
    palette_high: tuple[int, int, int] = (78, 105, 54)  # verde
    # --- textura por relieve ---
    height_texture: bool = True
    palette_rock: tuple[int, int, int] = (122, 118, 110)  # gris roca
    palette_snow: tuple[int, int, int] = (238, 242, 247)  # nieve
    rock_height: float = 130.0
    snow_height: float = 200.0
    band_blend: float = 25.0
    band_noise: float = 12.0
    slope_rock_deg: float = 30.0
    slope_rock_blend: float = 12.0
    snow_slope_limit_deg: float = 45.0
    texture_flip_v: bool = False

    def __post_init__(self) -> None:
        # YAML entrega listas; normalizamos a tuplas.
        self.palette_low = tuple(self.palette_low)  # type: ignore[assignment]
        self.palette_high = tuple(self.palette_high)  # type: ignore[assignment]
        self.palette_rock = tuple(self.palette_rock)  # type: ignore[assignment]
        self.palette_snow = tuple(self.palette_snow)  # type: ignore[assignment]


@dataclass
class BackgroundSettings(_SettingsBase):
    """Espejo de ``BackgroundSettings`` + subgrupo propio ``procedural``.

    ``flatten_roads`` aplana el corredor de cada vía sobre el DEM del
    background (ver ``mapforge.terrain.flatten``). Los dos campos
    ``flatten_roads_*`` son propios de MapForge y gobiernan la integración con
    el terreno: ``feather`` es el ancho de la transición en metros (``None`` =
    el doble del ancho de la vía) y ``smooth`` la ventana de suavizado
    longitudinal del eje, también en metros (``0`` = off).
    """

    generate_background: bool = True
    generate_water: bool = False  # fuera de alcance
    water_blurriness: int = 20
    remove_center: bool = True
    flatten_roads: bool = True
    flatten_water: bool = False
    # --- campos propios de MapForge ---
    flatten_roads_feather: float | None = None
    flatten_roads_smooth: float = 25.0
    procedural: ProceduralBackgroundSettings = field(
        default_factory=ProceduralBackgroundSettings
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BackgroundSettings":
        data = dict(data or {})
        procedural = data.pop("procedural", None)
        kwargs = _filtered_kwargs(cls, data)
        kwargs.pop("procedural", None)
        obj = cls(**kwargs)
        if procedural is not None:
            obj.procedural = ProceduralBackgroundSettings.from_dict(procedural)
        return obj


@dataclass
class GRLESettings(_SettingsBase):
    """Espejo de ``GRLESettings`` (farmlands e infoLayers, §S5)."""

    farmland_margin: int = 5
    add_farmyards: bool = True
    base_price: int = 12500
    price_scale: int = 100
    add_grass: bool = True
    base_grass: str = "meadow"
    random_plants: bool = False
    plants_island_minimum_size: int = 10
    plants_island_maximum_size: int = 200
    plants_island_vertex_count: int = 30
    plants_island_rounding_radius: int = 15
    plants_island_percent: int = 100
    fill_empty_farmlands: bool = True


@dataclass
class I3DSettings(_SettingsBase):
    """Espejo de ``I3DSettings`` + campos propios del DisplacementLayer.

    Campos propios (decisiones técnicas del plan):

    - ``displacement_layer_size_factor``: DisplacementLayer.size = map_size × factor
      (FACT-source: 8).
    - ``displacement_layer_cell_size_base``: cellSize = base / map_size (regla
      inferida: 16384/8192 = 2 como el artefacto; producto size×cellSize = 131072).
    """

    add_trees: bool = True  # fuera de alcance (bosques excluidos); espejo
    forest_density: int = 10
    tree_limit: int = 12500
    trees_relative_shift: int = 20
    spline_density: int = 2
    add_reversed_splines: bool = True
    field_splines: bool = False
    license_plate_prefix: str = "M4F"
    self_clear: bool = False
    displacement_layer_max_height: float = 0.2
    # --- campos propios de MapForge ---
    displacement_layer_size_factor: int = 8
    displacement_layer_cell_size_base: int = 16384


@dataclass
class TextureSettings(_SettingsBase):
    """Espejo de ``TextureSettings`` (motor de texturas, §S3)."""

    dissolve: bool = False
    fields_padding: float = 3
    skip_drains: bool = True
    use_cache: bool = False  # sin efecto en MapForge; espejo
    use_precise_tags: bool = False


#: Mapeo grupo → (clave corta config.yaml, clave estilo Maps4FS).
_GROUPS: dict[str, tuple[str, type]] = {
    "dem": ("DEMSettings", DEMSettings),
    "background": ("BackgroundSettings", BackgroundSettings),
    "grle": ("GRLESettings", GRLESettings),
    "i3d": ("I3DSettings", I3DSettings),
    "texture": ("TextureSettings", TextureSettings),
}

#: Grupos de Maps4FS reconocidos pero fuera del alcance del proyecto.
_IGNORED_GROUPS = {
    "PreprocessorSettings",
    "SatelliteSettings",
    "BuildingSettings",
    "SplineSettings",
}


@dataclass
class GenerationSettings:
    """Agregado de todos los settings de generación.

    Campos propios de MapForge a nivel raíz:

    - ``seed``: seed global (dissolve y textura procedural del background,
      ``numpy.random.default_rng(seed)``).
    - ``input_height_scale``: metros = valor_uint16_de_entrada × escala
      (default 255/65535, la convención del artefacto).
    """

    seed: int = 42
    input_height_scale: float = DEFAULT_INPUT_HEIGHT_SCALE
    dem: DEMSettings = field(default_factory=DEMSettings)
    background: BackgroundSettings = field(default_factory=BackgroundSettings)
    grle: GRLESettings = field(default_factory=GRLESettings)
    i3d: I3DSettings = field(default_factory=I3DSettings)
    texture: TextureSettings = field(default_factory=TextureSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationSettings":
        """Construye los settings desde un dict de config.yaml o de un
        ``generation_settings.json`` de Maps4FS (claves ``XxxSettings``)."""
        data = dict(data or {})
        obj = cls()
        if "seed" in data:
            obj.seed = int(data.pop("seed"))
        if "input_height_scale" in data:
            obj.input_height_scale = float(data.pop("input_height_scale"))

        for short_key, (m4fs_key, group_cls) in _GROUPS.items():
            group_data = None
            if short_key in data:
                group_data = data.pop(short_key)
            if m4fs_key in data:
                merged = data.pop(m4fs_key)
                group_data = {**merged, **(group_data or {})}
            if group_data is not None:
                setattr(obj, short_key, group_cls.from_dict(group_data))

        leftovers = sorted(set(data) - _IGNORED_GROUPS)
        if leftovers:
            logger.warning(
                "GenerationSettings: grupos/claves desconocidos ignorados: %s",
                ", ".join(leftovers),
            )
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "input_height_scale": self.input_height_scale,
            **{key: getattr(self, key).to_dict() for key in _GROUPS},
        }
