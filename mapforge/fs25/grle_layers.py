"""Creación de infoLayers/densityMaps en cero (Fase 5 del plan; §S5).

Todos los PNG de ``map/data/`` declarados en ``config/grle_schema.json``
(copia de ``fs25-grle-schema.json``) se crean como CEROS con tamaño
``map_size × height/width_multiplier`` según el schema: farmlands ×0.5,
environment ×0.25, fieldType/indoor/tip/densityMaps ×2.0, navigation/placement
×1.0 — coincide 1:1 con el artefacto (4096/2048/16384/8192). Solo PNG, nunca
GRLE/GDM: el motor compila los binarios (FACT).

Réplica de ``GRLE.process()`` de Maps4FS 1.8.242 (component/grle.py): por cada
entrada del schema, ``np.zeros`` con el dtype/canales declarados y
``cv2.imwrite`` en el directorio de weights (``map/data``).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

import cv2

if TYPE_CHECKING:
    from mapforge.project import Project

logger = logging.getLogger("mapforge.fs25.grle_layers")


def create_empty_grle_layers(project: "Project") -> list[str]:
    """Crea todos los rasters del grle_schema en cero; devuelve los nombres.

    Cada entrada del schema define ``name``, ``height_multiplier``,
    ``width_multiplier``, ``channels`` y ``data_type``; el raster resultante
    mide ``int(map_size × multiplier)`` por lado y se escribe con
    ``cv2.imwrite`` en ``map/data/`` (paridad con ``GRLE.process()`` de 1.8).
    """
    with open(project.paths.grle_schema, "r", encoding="utf-8") as f:
        grle_schema = json.load(f)

    data_dir = project.paths.map_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    map_size = project.map.size
    created: list[str] = []

    for info_layer in grle_schema:
        if not isinstance(info_layer, dict):
            logger.warning("entrada de schema GRLE inválida: %r", info_layer)
            continue

        name = info_layer["name"]
        height = int(map_size * info_layer["height_multiplier"])
        width = int(map_size * info_layer["width_multiplier"])
        channels = int(info_layer["channels"])
        data_type = info_layer["data_type"]

        if channels == 1:
            info_layer_data = np.zeros((height, width), dtype=data_type)
        else:
            info_layer_data = np.zeros((height, width, channels), dtype=data_type)

        cv2.imwrite(str(data_dir / name), info_layer_data)
        created.append(name)
        logger.debug("capa GRLE %s creada en cero (%s)", name, info_layer_data.shape)

    logger.info("%d capas GRLE creadas en cero en %s", len(created), data_dir)
    return created
