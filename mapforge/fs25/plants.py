"""Hierba base en ``densityMap_fruits.png`` (réplica de ``GRLE._add_plants``).

Maps4FS 1.8.242 pinta la hierba jugable escribiendo el valor de la planta
(``meadow`` = 131, ``smallDenseMix`` = 33) en el canal R del
``densityMap_fruits.png`` (FACT verificado en el artefacto golden:
``FS25_Valle_Bonito/map/data/densityMap_fruits.png`` tiene 131 SOLO en el rojo).

Algoritmo (``component/grle.py:264-361`` de 1.8.242), en el mismo orden:

1. máscara = weight de la capa con ``usage == "grass"`` (en el schema FS25 es
   ``grass``, la capa BASE: todo lo que ninguna otra capa reclamó), con
   ``*_preview.png`` si hubo dissolve;
2. reescalada ×2 con ``INTER_NEAREST`` (el densityMap es 2× el weight);
3. se le suma la capa ``usage == "forest"`` (``forestGrass``), también ×2;
4. copia con el valor de planta donde la máscara no es cero;
5. si ``random_plants``: islas de valores {65, 97, 129, 161, 193, 225}
   (polígono de N vértices distorsionado y redondeado con ``buffer``);
6. la MÁSCARA se erosiona 3×3 y se pone a cero la copia donde la máscara
   erosionada es cero (las islas que caen fuera de la hierba desaparecen);
7. marco de 1 px a cero;
8. se escribe en el canal R del densityMap (en 1.8 es
   ``density_map_fruits[:, :, 0] = ...`` seguido de ``cvtColor(BGR2RGB)``, que
   acaba dejando el valor en el rojo del PNG).

Diferencia deliberada con 1.8: las islas usan el RNG sembrado del proyecto
(``fresh_rng``) en vez de ``random``/``np.random`` globales, para que la
generación sea reproducible como el resto de MapForge.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import Polygon

import cv2

from mapforge.textures.schema import get_layer_by_usage, load_texture_schema

if TYPE_CHECKING:
    from mapforge.project import Project

logger = logging.getLogger("mapforge.fs25.plants")

#: Fichero del densityMap de frutos dentro de ``map/data`` (grle_schema).
DENSITY_MAP_FRUITS = "densityMap_fruits.png"

#: Valor de píxel por planta (FACT-source grle.py: ``plant_to_pixel_value``).
PLANT_PIXEL_VALUES: dict[str, int] = {"smallDenseMix": 33, "meadow": 131}

#: Valor por defecto si ``base_grass`` no está en la tabla (meadow, como 1.8).
DEFAULT_PLANT_PIXEL_VALUE = 131

#: Valores posibles de las islas de plantas (``create_island_of_plants``).
ISLAND_PLANT_VALUES: tuple[int, ...] = (65, 97, 129, 161, 193, 225)

#: Distorsión de ángulos/radios de las islas (``island_distortion`` de 1.8).
ISLAND_DISTORTION = 0.3

#: Stream del RNG del proyecto reservado para las islas de plantas.
PLANTS_RNG_STREAM = 90


def plant_to_pixel_value(plant_name: str) -> int | None:
    """Valor de píxel de una planta, o ``None`` si no se conoce."""
    return PLANT_PIXEL_VALUES.get(plant_name)


def get_rounded_polygon(
    rng: np.random.Generator,
    num_vertices: int,
    center: tuple[int, int],
    radius: int,
    rounding_radius: int,
) -> list[tuple[float, float]] | None:
    """Polígono de ``num_vertices`` distorsionado y redondeado con ``buffer``
    (réplica de ``GRLE.get_rounded_polygon`` con el RNG del proyecto)."""
    angle_offset = np.pi / num_vertices
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False) + angle_offset
    random_angles = angles + rng.uniform(-ISLAND_DISTORTION, ISLAND_DISTORTION, num_vertices)
    random_radii = radius + rng.uniform(
        -radius * ISLAND_DISTORTION, radius * ISLAND_DISTORTION, num_vertices
    )

    points = [
        (center[0] + np.cos(a) * r, center[1] + np.sin(a) * r)
        for a, r in zip(random_angles, random_radii)
    ]
    buffered_polygon = Polygon(points).buffer(rounding_radius, quad_segs=16)
    if buffered_polygon.is_empty or not hasattr(buffered_polygon, "exterior"):
        return None
    rounded_polygon = list(buffered_polygon.exterior.coords)
    return rounded_polygon or None


def create_islands_of_plants(
    image: np.ndarray,
    count: int,
    rng: np.random.Generator,
    minimum_size: int,
    maximum_size: int,
    vertex_count: int,
    rounding_radius: int,
) -> np.ndarray:
    """Pinta ``count`` islas de plantas sobre ``image`` (in place).

    Réplica de ``GRLE.create_island_of_plants``: valor, tamaño y posición al
    azar; los polígonos que fallan se saltan sin romper.
    """
    for _ in range(count):
        plant_value = int(ISLAND_PLANT_VALUES[rng.integers(len(ISLAND_PLANT_VALUES))])
        island_size = int(rng.integers(minimum_size, maximum_size + 1))
        x = int(rng.integers(0, image.shape[1] - island_size + 1))
        y = int(rng.integers(0, image.shape[0] - island_size + 1))

        try:
            polygon_points = get_rounded_polygon(
                rng,
                num_vertices=vertex_count,
                center=(x + island_size // 2, y + island_size // 2),
                radius=island_size // 2,
                rounding_radius=rounding_radius,
            )
            if not polygon_points:
                continue
            nodes = np.array(polygon_points, np.int32)
            cv2.fillPoly(image, [nodes], (float(plant_value),))
        except Exception as exc:  # noqa: BLE001 — paridad con 1.8
            logger.debug("isla de plantas descartada: %r", exc)
            continue

    return image


def remove_edge_pixel_values(image: np.ndarray) -> np.ndarray:
    """Marco de 1 px a cero (``GRLE.remove_edge_pixel_values``)."""
    image[0, :] = 0
    image[-1, :] = 0
    image[:, 0] = 0
    image[:, -1] = 0
    return image


class PlantsWriter:
    """Pinta la hierba base en ``map/data/densityMap_fruits.png``."""

    def __init__(self, project: "Project") -> None:
        self.project = project
        self.logger = project.logger.getChild("plants")
        self.settings = project.settings.grle

    def run(self) -> dict[str, Any]:
        """Escribe la hierba; devuelve un resumen para ``generation_info``."""
        if not self.settings.add_grass:
            self.logger.info("add_grass desactivado: no se pinta hierba")
            return {"status": "skipped", "reason": "add_grass=false"}

        weights_dir = self.project.paths.map_data_dir
        density_map_path = weights_dir / DENSITY_MAP_FRUITS
        if not density_map_path.is_file():
            self.logger.warning(
                "%s no existe; ejecuta antes la etapa grle_layers", density_map_path
            )
            return {"status": "skipped", "reason": f"{DENSITY_MAP_FRUITS} no existe"}

        layers = load_texture_schema(self.project.paths.texture_schema)
        grass_layer = get_layer_by_usage(layers, "grass")
        if grass_layer is None:
            self.logger.warning("el texture schema no tiene ninguna capa con usage='grass'")
            return {"status": "skipped", "reason": "sin capa usage=grass"}

        grass_path = grass_layer.get_preview_or_path(weights_dir)
        if not grass_path.is_file():
            self.logger.warning("weight de la hierba no encontrado: %s", grass_path)
            return {"status": "skipped", "reason": f"{grass_path.name} no existe"}

        mask = cv2.imread(str(grass_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(
            mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation=cv2.INTER_NEAREST
        )

        forest_layer = get_layer_by_usage(layers, "forest")
        forest_name = None
        if forest_layer is not None:
            forest_path = forest_layer.get_preview_or_path(weights_dir)
            if forest_path.is_file():
                forest = cv2.imread(str(forest_path), cv2.IMREAD_UNCHANGED)
                forest = cv2.resize(
                    forest,
                    (forest.shape[1] * 2, forest.shape[0] * 2),
                    interpolation=cv2.INTER_NEAREST,
                )
                # El bosque también lleva hierba debajo (1.8: se fusionan máscaras).
                mask[forest != 0] = 255
                forest_name = forest_layer.name
                del forest

        plant_value = plant_to_pixel_value(str(self.settings.base_grass))
        if not plant_value:
            self.logger.warning(
                "base_grass=%r desconocido; se usa meadow (%d)",
                self.settings.base_grass,
                DEFAULT_PLANT_PIXEL_VALUE,
            )
            plant_value = DEFAULT_PLANT_PIXEL_VALUE

        plants = np.zeros_like(mask)
        plants[mask != 0] = plant_value

        islands = 0
        if self.settings.random_plants:
            islands = int(
                self.project.map.size * self.settings.plants_island_percent // 100
            )
            plants = create_islands_of_plants(
                plants,
                islands,
                self.project.fresh_rng(PLANTS_RNG_STREAM),
                minimum_size=self.settings.plants_island_minimum_size,
                maximum_size=self.settings.plants_island_maximum_size,
                vertex_count=self.settings.plants_island_vertex_count,
                rounding_radius=self.settings.plants_island_rounding_radius,
            )

        # Máscara erosionada: recorta 1 px el borde de la hierba (y con ella las
        # islas que se salieron), como 1.8.
        cv2.erode(mask, np.ones((3, 3), np.uint8), dst=mask, iterations=1)
        plants[mask == 0] = 0
        remove_edge_pixel_values(plants)
        del mask

        density_map = cv2.imread(str(density_map_path), cv2.IMREAD_UNCHANGED)
        if density_map is None or density_map.ndim != 3:
            self.logger.warning("%s no es un PNG de 3 canales", density_map_path)
            return {"status": "skipped", "reason": f"{DENSITY_MAP_FRUITS} inválido"}
        if density_map.shape[:2] != plants.shape[:2]:
            self.logger.warning(
                "tamaños incompatibles: %s es %s y la máscara ×2 es %s",
                DENSITY_MAP_FRUITS,
                density_map.shape[:2],
                plants.shape[:2],
            )
            return {"status": "skipped", "reason": "tamaños incompatibles"}

        # 1.8 hace density_map[:, :, 0] = plants y luego cvtColor(BGR2RGB); el
        # resultado neto es que la planta acaba en el canal R del PNG.
        density_map[:, :, 0] = plants
        blue = density_map[:, :, 0].copy()
        density_map[:, :, 0] = density_map[:, :, 2]
        density_map[:, :, 2] = blue
        del blue

        cv2.imwrite(str(density_map_path), density_map)

        painted = int(np.count_nonzero(plants))
        total = plants.shape[0] * plants.shape[1]
        self.logger.info(
            "hierba %s (valor %d) en %.1f %% de %s",
            self.settings.base_grass,
            plant_value,
            100 * painted / total,
            DENSITY_MAP_FRUITS,
        )
        return {
            "status": "ok",
            "grass_layer": grass_layer.name,
            "forest_layer": forest_name,
            "base_grass": str(self.settings.base_grass),
            "plant_value": plant_value,
            "islands": islands,
            "pixels": painted,
            "coverage": round(painted / total, 4),
        }
