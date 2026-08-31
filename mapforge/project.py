"""Modelo de proyecto de MapForge.

Un :class:`Project` reúne todo lo que necesita el pipeline: parámetros del
mapa (tamaño, rotación, coordenadas), rutas de entrada (heightmap, OSM,
template, schemas), rutas de salida, settings de generación, logging y el
generador de números aleatorios derivado de la seed global.

Se construye desde un ``config.yaml`` (ver ``config/config.example.yaml``) con
:meth:`Project.from_yaml`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mapforge.settings import BACKGROUND_DISTANCE, GenerationSettings

logger = logging.getLogger("mapforge")

#: Multiplicador del tamaño rotado (map.py de Maps4FS: 1.5 si hay rotación).
ROTATION_SIZE_MULTIPLIER = 1.5


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configura el logging de consola de mapforge (idempotente)."""
    root = logging.getLogger("mapforge")
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level)


@dataclass
class MapParams:
    """Parámetros geométricos/geográficos del mapa."""

    size: int = 2048
    rotation: float = 0
    latitude: float = 0.0
    longitude: float = 0.0
    output_size: int | None = None

    @property
    def coordinates(self) -> tuple[float, float]:
        """(lat, lon) del centro del mapa."""
        return (self.latitude, self.longitude)

    @property
    def rotated_size(self) -> int:
        """Tamaño del lienzo rotado (regla de Maps4FS: ×1.5 si hay rotación)."""
        if self.rotation:
            return int(self.size * ROTATION_SIZE_MULTIPLIER)
        return self.size

    @property
    def background_size(self) -> int:
        """Tamaño del DEM/mesh de background: map_size + 2×2048."""
        return self.size + 2 * BACKGROUND_DISTANCE


@dataclass
class ProjectPaths:
    """Rutas de entrada y salida del proyecto (absolutas tras ``resolve_from``)."""

    heightmap: Path
    osm: Path
    template: Path
    texture_schema: Path
    grle_schema: Path
    output_dir: Path

    def resolve_from(self, base: Path) -> "ProjectPaths":
        """Resuelve las rutas relativas contra ``base`` (dir del config.yaml)."""

        def _abs(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        return ProjectPaths(
            heightmap=_abs(self.heightmap),
            osm=_abs(self.osm),
            template=_abs(self.template),
            texture_schema=_abs(self.texture_schema),
            grle_schema=_abs(self.grle_schema),
            output_dir=_abs(self.output_dir),
        )

    # --- rutas derivadas dentro del directorio de salida (estructura FS25) ---

    @property
    def map_dir(self) -> Path:
        return self.output_dir / "map"

    @property
    def map_data_dir(self) -> Path:
        return self.output_dir / "map" / "data"

    @property
    def map_config_dir(self) -> Path:
        return self.output_dir / "map" / "config"

    @property
    def map_i3d(self) -> Path:
        return self.output_dir / "map" / "map.i3d"

    @property
    def map_xml(self) -> Path:
        return self.output_dir / "map" / "map.xml"

    @property
    def splines_i3d(self) -> Path:
        return self.output_dir / "map" / "splines.i3d"

    @property
    def mod_desc(self) -> Path:
        return self.output_dir / "modDesc.xml"

    @property
    def dem_png(self) -> Path:
        return self.output_dir / "map" / "data" / "dem.png"

    @property
    def background_dir(self) -> Path:
        return self.output_dir / "background"

    @property
    def masks_dir(self) -> Path:
        return self.output_dir / "masks"

    @property
    def info_layers_dir(self) -> Path:
        return self.output_dir / "info_layers"

    @property
    def textures_json(self) -> Path:
        return self.info_layers_dir / "textures.json"

    @property
    def generation_info_json(self) -> Path:
        return self.output_dir / "generation_info.json"


class Project:
    """Proyecto de generación: parámetros + rutas + settings + seed + logging."""

    def __init__(
        self,
        name: str,
        map_params: MapParams,
        paths: ProjectPaths,
        settings: GenerationSettings | None = None,
    ) -> None:
        self.name = name
        self.map = map_params
        self.paths = paths
        self.settings = settings or GenerationSettings()
        self.logger = logging.getLogger("mapforge.project")
        self._rng: np.random.Generator | None = None
        #: height_scale calculado por la fase DEM; lo consume el escritor i3d.
        self.height_scale: int | None = None

    # ------------------------------------------------------------------ rng

    @property
    def rng(self) -> np.random.Generator:
        """RNG determinista derivado de ``settings.seed`` (se crea perezosamente)."""
        if self._rng is None:
            self._rng = np.random.default_rng(self.settings.seed)
        return self._rng

    def fresh_rng(self, stream: int = 0) -> np.random.Generator:
        """RNG independiente y reproducible para un subcomponente concreto."""
        return np.random.default_rng([self.settings.seed, stream])

    # ---------------------------------------------------------------- carga

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Project":
        """Carga un proyecto desde un ``config.yaml``."""
        config_path = Path(config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.from_config_dict(raw, base_dir=config_path.parent)

    @classmethod
    def from_config_dict(cls, raw: dict[str, Any], base_dir: Path) -> "Project":
        """Construye el proyecto desde el dict del YAML.

        Las rutas relativas del config se resuelven contra ``base_dir`` (el
        directorio donde vive el config.yaml).
        """
        project_cfg = dict(raw.get("project") or {})
        map_cfg = dict(raw.get("map") or {})
        inputs_cfg = dict(raw.get("inputs") or {})
        settings_cfg = raw.get("settings") or {}

        name = str(project_cfg.get("name", "mapforge_map"))

        map_params = MapParams(
            size=int(map_cfg.get("size", 2048)),
            rotation=float(map_cfg.get("rotation", 0)),
            latitude=float(map_cfg.get("latitude", 0.0)),
            longitude=float(map_cfg.get("longitude", 0.0)),
            output_size=(
                int(map_cfg["output_size"])
                if map_cfg.get("output_size") is not None
                else None
            ),
        )

        missing = [k for k in ("heightmap", "osm", "template") if not inputs_cfg.get(k)]
        if missing:
            raise ValueError(
                f"config: faltan rutas obligatorias en 'inputs': {', '.join(missing)}"
            )

        default_config_dir = Path(__file__).resolve().parent.parent / "config"
        paths = ProjectPaths(
            heightmap=Path(inputs_cfg["heightmap"]),
            osm=Path(inputs_cfg["osm"]),
            template=Path(inputs_cfg["template"]),
            texture_schema=Path(
                inputs_cfg.get("texture_schema", default_config_dir / "texture_schema.json")
            ),
            grle_schema=Path(
                inputs_cfg.get("grle_schema", default_config_dir / "grle_schema.json")
            ),
            output_dir=Path(
                project_cfg.get("output_dir", "output/" + name.replace(" ", "_"))
            ),
        ).resolve_from(base_dir)

        settings = GenerationSettings.from_dict(settings_cfg)
        return cls(name=name, map_params=map_params, paths=paths, settings=settings)

    # ------------------------------------------------------------ utilidades

    def validate_inputs(self) -> list[str]:
        """Devuelve la lista de problemas encontrados en las rutas de entrada."""
        problems: list[str] = []
        for label, path in (
            ("heightmap", self.paths.heightmap),
            ("osm", self.paths.osm),
            ("template", self.paths.template),
            ("texture_schema", self.paths.texture_schema),
            ("grle_schema", self.paths.grle_schema),
        ):
            if not path.exists():
                problems.append(f"{label}: no existe {path}")
        return problems

    def describe(self) -> str:
        """Resumen legible del proyecto para el log."""
        m = self.map
        return (
            f"proyecto '{self.name}': size={m.size} rotation={m.rotation} "
            f"lat={m.latitude} lon={m.longitude} background={m.background_size} "
            f"seed={self.settings.seed} output={self.paths.output_dir}"
        )
