"""Schema de capas de textura (Fase 3 del plan).

``Layer``: espejo de ``layer.py`` de Maps4FS 1.8.242 con todos sus campos
(``name, count, tags, width, color, exclude_weight, priority, info_layer,
usage, background, invisible, procedural, border``), cargado desde
``config/texture_schema.json`` (copia de ``fs25-texture-schema.json``).

Nombres de weight (FACT-source, ``Layer.path``/``Texture._generate_weights``):

- ``{name}{NN}_weight.png`` con NN = 01..count;
- ``{name}_weight.png`` si ``count == 0``;
- sin sufijo ``_weight`` cuando ``exclude_weight`` (caso ``forestRockRoots``:
  ``forestRockRoots01.png``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Capas cuyos ficheros no siguen el patrón ``{name}{NN}_weight.png``
#: (lista literal de ``Layer.paths`` de Maps4FS 1.8.242).
INCONSISTENT_NAMES = ("forestRockRoot", "waterPuddle")


@dataclass
class Layer:
    """Una capa del texture schema. Los campos espejan layer.py de Maps4FS 1.8.

    ``tags`` usa la semántica osmnx del schema: ``str | list[str] | True`` por
    clave OSM. ``width`` es el RADIO del buffer para LineStrings (§S3).
    """

    name: str
    count: int
    tags: dict[str, Any] | None = None
    width: int | None = None
    color: tuple[int, int, int] | list[int] | None = None
    exclude_weight: bool = False
    priority: int | None = None
    info_layer: str | None = None
    usage: str | None = None
    background: bool = False
    invisible: bool = False
    procedural: list[str] | None = None
    border: int | None = None

    def __post_init__(self) -> None:
        if self.color is None:
            self.color = (255, 255, 255)

    # ------------------------------------------------------------- serialización

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Layer":
        """Crea la capa desde una entrada del schema JSON (``Layer.from_json``)."""
        return cls(**data)

    def to_json(self) -> dict[str, Any]:
        """Vuelca la capa a dict, omitiendo Nones (``Layer.to_json`` de 1.8)."""
        data: dict[str, Any] = {
            "name": self.name,
            "count": self.count,
            "tags": self.tags,
            "width": self.width,
            "color": list(self.color) if self.color is not None else None,
            "exclude_weight": self.exclude_weight,
            "priority": self.priority,
            "info_layer": self.info_layer,
            "usage": self.usage,
            "background": self.background,
            "invisible": self.invisible,
            "procedural": self.procedural,
            "border": self.border,
        }
        return {k: v for k, v in data.items() if v is not None}

    # ------------------------------------------------------------------- rutas

    @property
    def weight_postfix(self) -> str:
        """``_weight`` salvo que la capa lo excluya (``exclude_weight``)."""
        return "_weight" if not self.exclude_weight else ""

    def weight_filenames(self) -> list[str]:
        """Nombres de TODOS los weight files de la capa (los que se crean en
        cero), réplica de ``Texture._generate_weights``: ``{name}{NN}{postfix}.png``
        con NN = 01..count, o ``{name}{postfix}.png`` si ``count == 0``."""
        postfix = f"{self.weight_postfix}.png"
        if self.count == 0:
            return [f"{self.name}{postfix}"]
        return [f"{self.name}{i:02d}{postfix}" for i in range(1, self.count + 1)]

    def path(self, weights_dir: str | Path) -> Path:
        """Ruta del PRIMER weight file de la capa (``Layer.path`` de 1.8)."""
        idx = "01" if self.count > 0 else ""
        return Path(weights_dir) / f"{self.name}{idx}{self.weight_postfix}.png"

    def path_preview(self, weights_dir: str | Path) -> Path:
        """Ruta del preview del primer weight (dissolve): ``*_preview.png``."""
        path = self.path(weights_dir)
        return path.with_name(path.name.replace(".png", "_preview.png"))

    def get_preview_or_path(self, weights_dir: str | Path) -> Path:
        """Preview si existe (tras dissolve), si no el weight original."""
        preview = self.path_preview(weights_dir)
        return preview if preview.is_file() else self.path(weights_dir)

    def paths(self, weights_dir: str | Path) -> list[Path]:
        """Weight files de la capa EXISTENTES en disco (``Layer.paths`` de 1.8:
        lista el directorio). A diferencia de 1.8 (os.listdir, orden arbitrario)
        se devuelven ordenados para que dissolve sea determinista."""
        weights_dir = Path(weights_dir)
        filenames = sorted(p.name for p in weights_dir.iterdir() if p.is_file())

        if self.name in INCONSISTENT_NAMES or self.name.startswith(INCONSISTENT_NAMES):
            return [
                weights_dir / filename
                for filename in filenames
                if filename.startswith(self.name)
            ]

        pattern = rf"{self.name}\d{{2}}_weight\.png"
        return [
            weights_dir / filename
            for filename in filenames
            if re.fullmatch(pattern, filename)
        ]


def load_texture_schema(schema_path: str | Path) -> list[Layer]:
    """Carga ``texture_schema.json`` → lista de :class:`Layer`.

    Réplica de ``Texture.get_schema`` + ``read_layers``: el fichero debe ser
    una lista JSON de dicts de capa.
    """
    schema_path = Path(schema_path)
    if not schema_path.is_file():
        raise FileNotFoundError(f"Texture schema no encontrado: {schema_path}")
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            layers_schema = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Error cargando el texture schema: {exc}") from exc

    if not isinstance(layers_schema, list):
        raise ValueError("El texture schema debe ser una lista de dicts de capa.")

    try:
        return [Layer.from_json(entry) for entry in layers_schema]
    except Exception as exc:
        raise ValueError(f"Error construyendo las capas del schema: {exc}") from exc


def layers_by_priority(layers: list[Layer]) -> list[Layer]:
    """Capas ordenadas para el dibujo (``Texture.layers_by_priority`` de 1.8):
    primero las de priority None (en orden de schema), luego por priority
    DESCENDENTE (empates en orden de schema; sort estable)."""
    return sorted(
        layers,
        key=lambda layer: (
            layer.priority is not None,
            -layer.priority if layer.priority is not None else float("inf"),
        ),
    )


def get_base_layer(layers: list[Layer]) -> Layer | None:
    """La capa base: la primera con ``priority == 0`` (``get_base_layer`` 1.8)."""
    for layer in layers:
        if layer.priority == 0:
            return layer
    return None
