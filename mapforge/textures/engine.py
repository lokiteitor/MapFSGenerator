"""Motor de texturas (Fase 3 del plan; algoritmo S3 del informe forense).

Réplica del ``Texture.process()`` de Maps4FS 1.8.242, en el mismo orden::

    _prepare_weights → draw → rotate_textures → add_borders
    → dissolve (opcional) → copy_procedural

Dibujo (S3, "el primer claim gana")::

    capas ordenadas: priority None primero, luego priority DESCENDENTE
    la capa base (priority == 0) se pospone al final
    por capa:
        mask = NOT cumulative
        dibujar los polígonos de la capa (fillPoly 255)
        output = imagen_capa AND mask
        cumulative |= output
    base = NOT cumulative

Además: líneas ``buffer(width)`` con width = RADIO; ``fields_padding`` →
``buffer(-padding)`` (si colapsa se ignora); polígonos con <3 puntos se
saltan; ``skip_drains`` salta las capas ``usage == "drain"``; capas
``invisible`` no se dibujan pero sí alimentan su info_layer.

Salidas:

- weights en ``map/data/{name}{NN}_weight.png`` (uint8, map_size², en cero
  las capas sin contenido; ``exclude_weight`` sin sufijo ``_weight``);
- ``info_layers/textures.json`` con el MISMO formato que Maps4FS
  (fields/farmyards/…: listas de puntos px; ``roads_polylines``:
  ``{points, tags}`` con ``tags = str(dict de tags de la capa)``) — contrato
  interno para las fases 4-6;
- máscaras procedural en ``map/data/masks/PG_*.png`` + ``BLOCKMASK.png``;
- con ``dissolve``: reparto aleatorio por píxel entre sublayers usando el
  rng(seed) del proyecto y original → ``*_preview.png``.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np
from shapely.geometry import LineString, Point, Polygon

import cv2

from mapforge.osm.parser import OsmData, parse_osm
from mapforge.osm.projection import MapProjection
from mapforge.textures.rasterizer import (
    MIN_POLYGON_POINTS,
    geometry_to_polygon,
    np_to_polygon_points,
    polygon_to_np,
)
from mapforge.textures.schema import (
    Layer,
    get_base_layer,
    layers_by_priority,
    load_texture_schema,
)

if TYPE_CHECKING:
    from mapforge.project import Project

#: Nombre del fichero de máscara de bloqueo procedural (vacía).
BLOCKMASK_FILENAME = "BLOCKMASK.png"


class TextureEngine:
    """Genera los weight maps, máscaras PG y ``info_layers/textures.json``.

    Réplica del componente ``Texture`` de Maps4FS 1.8.242 sobre el modelo de
    proyecto de MapForge: weights en ``map/data`` (``weights_dir_path`` de
    FS25), máscaras en ``map/data/masks``.
    """

    def __init__(self, project: "Project", osm_data: OsmData | None = None) -> None:
        self.project = project
        self.logger = project.logger.getChild("textures")
        self.settings = project.settings.texture

        self.map_size = project.map.size
        self.map_rotated_size = project.map.rotated_size
        self.rotation = project.map.rotation

        self.weights_dir: Path = project.paths.map_data_dir
        self.masks_dir: Path = self.weights_dir / "masks"
        self.info_layer_path: Path = project.paths.textures_json

        self.layers: list[Layer] = load_texture_schema(project.paths.texture_schema)
        self.projection = MapProjection(
            project.map.latitude, project.map.longitude, self.map_rotated_size
        )
        self._osm_data = osm_data

    # ------------------------------------------------------------------ datos

    @property
    def osm_data(self) -> OsmData:
        """OSM parseado (perezoso si no se inyectó en el constructor)."""
        if self._osm_data is None:
            self._osm_data = parse_osm(self.project.paths.osm)
        return self._osm_data

    def get_base_layer(self) -> Layer | None:
        return get_base_layer(self.layers)

    # -------------------------------------------------------------- pipeline

    def run(self) -> None:
        """Ejecuta el motor completo (orden de ``Texture.process()`` de 1.8)."""
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.info_layer_path.parent.mkdir(parents=True, exist_ok=True)

        self._prepare_weights()
        self.draw()
        self.rotate_textures()
        self.add_borders()
        if self.settings.dissolve:
            self.dissolve()
        self.copy_procedural()
        self.logger.info(
            "texturas generadas: %d capas, weights en %s",
            len(self.layers),
            self.weights_dir,
        )

    # ---------------------------------------------------------------- weights

    def _prepare_weights(self) -> None:
        """Crea TODOS los weight files en cero (``_generate_weights`` de 1.8):
        map_size² para capas sin tags, map_rotated_size² para capas con tags
        (se recortan al rotar)."""
        for layer in self.layers:
            if layer.tags is None:
                size = (self.map_size, self.map_size)
            else:
                size = (self.map_rotated_size, self.map_rotated_size)
            for filename in layer.weight_filenames():
                img = np.zeros(size, dtype=np.uint8)
                cv2.imwrite(str(self.weights_dir / filename), img)
        self.logger.debug("weights en cero creados para %d capas", len(self.layers))

    # ------------------------------------------------------------------ draw

    def draw(self) -> None:
        """Dibuja las capas por prioridad (S3) y guarda textures.json."""
        layers = [
            layer for layer in layers_by_priority(self.layers) if layer.tags is not None
        ]

        cumulative_image: np.ndarray | None = None
        info_layer_data: dict[str, list] = defaultdict(list)

        for layer in layers:
            if self.settings.skip_drains and layer.usage == "drain":
                self.logger.debug("capa %s saltada (usage=drain)", layer.name)
                continue
            if layer.priority == 0:
                self.logger.debug("capa base %s pospuesta al final", layer.name)
                continue

            layer_path = layer.path(self.weights_dir)
            layer_image = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)

            if cumulative_image is None:
                cumulative_image = np.zeros_like(layer_image)

            mask = cv2.bitwise_not(cumulative_image)
            self._draw_layer(layer, info_layer_data, layer_image)
            self._add_roads(layer, info_layer_data)

            output_image = cv2.bitwise_and(layer_image, mask)
            cumulative_image = cv2.bitwise_or(cumulative_image, output_image)

            cv2.imwrite(str(layer_path), output_image)
            self.logger.debug("capa %s dibujada → %s", layer.name, layer_path)

        self._save_info_layer_data(info_layer_data)

        if cumulative_image is not None:
            self.draw_base_layer(cumulative_image)

    def _draw_layer(
        self,
        layer: Layer,
        info_layer_data: dict[str, list],
        layer_image: np.ndarray,
    ) -> None:
        """Dibuja los polígonos de la capa y alimenta su info_layer
        (``_draw_layer`` de 1.8: <3 puntos se salta; ``invisible`` solo
        alimenta el info_layer)."""
        for polygon in self._polygons_generator(layer):
            if not len(polygon) > MIN_POLYGON_POINTS - 1:
                self.logger.debug("polígono con <3 puntos saltado (%s)", layer.name)
                continue
            if layer.info_layer:
                info_layer_data[layer.info_layer].append(np_to_polygon_points(polygon))
            if not layer.invisible:
                try:
                    cv2.fillPoly(layer_image, [polygon], color=255)
                except Exception as exc:  # noqa: BLE001 — paridad con 1.8
                    self.logger.warning("error dibujando polígono: %r", exc)
                    continue

    def _add_roads(self, layer: Layer, info_layer_data: dict[str, list]) -> None:
        """Vuelca las polilíneas de carreteras a ``roads_polylines``
        (``_add_roads`` de 1.8): solo capas con ``info_layer == "roads"``;
        cada entrada es ``{"points": [(x, y), ...], "tags": str(tags)}``."""
        if layer.info_layer != "roads":
            return
        for points in self._linestrings_generator(layer):
            info_layer_data[f"{layer.info_layer}_polylines"].append(
                {"points": points, "tags": str(layer.tags)}
            )

    def _polygons_generator(self, layer: Layer) -> Generator[np.ndarray, None, None]:
        """Genera los arrays de puntos de los polígonos de la capa
        (``objects_generator`` + ``polygons_generator`` de 1.8):

        - matching de tags sobre el OSM (semántica osmnx, orden de documento);
        - proyección a píxeles vértice a vértice (solo anillo exterior);
        - LineString/Point → ``buffer(width)`` con width = RADIO;
        - ``fields_padding`` → ``buffer(-padding)`` (si el resultado no es un
          Polygon con exterior, se conserva el original);
        - tipos no soportados (MultiPolygon, …) se saltan.
        """
        is_fields = layer.info_layer == "fields"
        for feature in self.osm_data.features_matching(layer.tags):
            pixel_geometry = self._to_pixel_geometry(feature.geometry)
            if pixel_geometry is None:
                continue
            polygon = geometry_to_polygon(pixel_geometry, layer.width)
            if polygon is None:
                continue

            if is_fields and self.settings.fields_padding > 0:
                padded_polygon = polygon.buffer(-self.settings.fields_padding)
                if not isinstance(padded_polygon, Polygon) or not list(
                    padded_polygon.exterior.coords
                ):
                    self.logger.debug(
                        "fields_padding demasiado alto, el field no se encoge"
                    )
                else:
                    polygon = padded_polygon

            yield polygon_to_np(polygon)

    def _linestrings_generator(
        self, layer: Layer
    ) -> Generator[list[tuple[int, int]], None, None]:
        """Polilíneas en píxeles de la capa (``linestrings_generator`` de 1.8):
        solo geometrías LineString, puntos truncados a int."""
        for feature in self.osm_data.features_matching(layer.tags):
            geometry = feature.geometry
            if isinstance(geometry, LineString):
                yield [
                    self.projection.latlon_to_pixel(lat, lon)
                    for lon, lat in geometry.coords
                ]

    def _to_pixel_geometry(self, geometry):
        """Proyecta una geometría de grados a píxeles como 1.8: Polygon (solo
        exterior), LineString y Point; el resto no está soportado → None."""
        if isinstance(geometry, Polygon):
            coords_pixel = [
                self.projection.latlon_to_pixel(lat, lon)
                for lon, lat in geometry.exterior.coords
            ]
            return Polygon(coords_pixel)
        if isinstance(geometry, LineString):
            return LineString(
                [
                    self.projection.latlon_to_pixel(lat, lon)
                    for lon, lat in geometry.coords
                ]
            )
        if isinstance(geometry, Point):
            return Point(self.projection.latlon_to_pixel(geometry.y, geometry.x))
        self.logger.debug("geometría %s no soportada", geometry.geom_type)
        return None

    def _save_info_layer_data(self, info_layer_data: dict[str, list]) -> None:
        """Guarda ``info_layers/textures.json`` (mismo formato que Maps4FS:
        json indent=4, ensure_ascii=False; si ya existe se fusiona)."""
        if self.info_layer_path.is_file():
            self.logger.debug("%s ya existe, se fusiona", self.info_layer_path)
            with open(self.info_layer_path, "r", encoding="utf-8") as f:
                info_layer_data.update(json.load(f))

        with open(self.info_layer_path, "w", encoding="utf-8") as f:
            json.dump(info_layer_data, f, ensure_ascii=False, indent=4)
        self.logger.debug("info layers → %s", self.info_layer_path)

    def draw_base_layer(self, cumulative_image: np.ndarray) -> None:
        """Base = NOT cumulative: rellena todo lo no reclamado
        (``draw_base_layer`` de 1.8)."""
        base_layer = self.get_base_layer()
        if base_layer is None:
            return
        layer_path = base_layer.path(self.weights_dir)
        img = cv2.bitwise_not(cumulative_image)
        cv2.imwrite(str(layer_path), img)
        self.logger.debug("capa base %s → %s", base_layer.name, layer_path)

    # ------------------------------------------------------------- rotación

    def rotate_textures(self) -> None:
        """Rota y recorta a map_size² las capas con tags (``rotate_textures``
        de 1.8; sin rotación no hace nada)."""
        if not self.rotation:
            return
        for layer in self.layers:
            if not layer.tags:
                continue
            layer_paths = layer.paths(self.weights_dir)
            layer_paths.append(layer.path_preview(self.weights_dir))
            for layer_path in layer_paths:
                if layer_path.is_file():
                    self._rotate_image(layer_path)

    def _rotate_image(self, image_path: Path) -> None:
        """Rota la imagen alrededor de su centro y recorta el centro a
        map_size² (réplica de ``Component.rotate_image``)."""
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            self.logger.warning("no se pudo leer %s para rotar", image_path)
            return
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

        start_x = center[0] - self.map_size // 2
        start_y = center[1] - self.map_size // 2
        cropped = rotated[
            start_y : start_y + self.map_size, start_x : start_x + self.map_size
        ]
        cv2.imwrite(str(image_path), cropped)

    # -------------------------------------------------------------- borders

    def add_borders(self) -> None:
        """Capas con ``border``: sus píxeles en el marco de ``border`` px se
        ponen a 0 y (si eran 255) se transfieren a la capa base
        (``add_borders`` + ``transfer_border`` de 1.8)."""
        base_layer = self.get_base_layer()
        base_layer_image = None
        if base_layer is not None:
            base_layer_image = cv2.imread(
                str(base_layer.path(self.weights_dir)), cv2.IMREAD_UNCHANGED
            )

        for layer in self.layers:
            if layer.border is None or not layer.border:
                continue
            layer_path = layer.path(self.weights_dir)
            layer_image = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
            self._transfer_border(layer_image, base_layer_image, layer.border)
            cv2.imwrite(str(layer_path), layer_image)
            self.logger.debug("border de %d px aplicado a %s", layer.border, layer.name)

        if base_layer_image is not None and base_layer is not None:
            cv2.imwrite(str(base_layer.path(self.weights_dir)), base_layer_image)

    @staticmethod
    def _transfer_border(
        src_image: np.ndarray, dst_image: np.ndarray | None, border: int
    ) -> None:
        """Réplica exacta de ``ImageComponent.transfer_border`` de 1.8."""
        borders = [
            (slice(None, border), slice(None)),
            (slice(None), slice(-border, None)),
            (slice(-border, None), slice(None)),
            (slice(None), slice(None, border)),
        ]
        for row_slice, col_slice in borders:
            border_slice = (row_slice, col_slice)
            if dst_image is not None:
                dst_image[border_slice][src_image[border_slice] != 0] = 255
            src_image[border_slice] = 0

    # ------------------------------------------------------------- dissolve

    def dissolve(self) -> None:
        """Reparte aleatoriamente cada píxel 255 de la capa entre sus
        ``count`` sublayers (``dissolve`` de 1.8) usando el rng(seed) del
        proyecto; la máscara original se guarda como ``*_preview.png``."""
        rng = self.project.rng
        for layer in self.layers:
            if not layer.tags:
                continue
            layer_path = layer.path(self.weights_dir)
            layer_paths = layer.paths(self.weights_dir)
            if len(layer_paths) < 2:
                self.logger.debug("capa %s con una sola textura, sin dissolve", layer.name)
                continue

            layer_image = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
            if layer_image is None or not np.any(layer_image):
                continue

            cv2.imwrite(str(layer.path_preview(self.weights_dir)), layer_image.copy())

            non_zero_coords = np.column_stack(np.where(layer_image > 0))
            choices = rng.integers(0, layer.count, size=len(non_zero_coords))

            for index, sublayer_path in enumerate(layer_paths[: layer.count]):
                sublayer = np.zeros_like(layer_image)
                selected = non_zero_coords[choices == index]
                sublayer[selected[:, 0], selected[:, 1]] = 255
                cv2.imwrite(str(sublayer_path), sublayer)

            self.logger.debug("capa %s disuelta en %d sublayers", layer.name, layer.count)

    # ----------------------------------------------------------- procedural

    def copy_procedural(self) -> None:
        """Copia/fusiona las capas ``procedural`` a ``masks/PG_*.png`` y crea
        ``BLOCKMASK.png`` vacío si no existe (``copy_procedural`` de 1.8)."""
        blockmask_path = self.masks_dir / BLOCKMASK_FILENAME
        if not blockmask_path.is_file():
            img = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
            cv2.imwrite(str(blockmask_path), img)

        pg_layers_by_type: dict[str, list[Path]] = defaultdict(list)
        for layer in self.layers:
            if layer.procedural:
                texture_path = layer.get_preview_or_path(self.weights_dir)
                for procedural_layer_name in layer.procedural:
                    pg_layers_by_type[procedural_layer_name].append(texture_path)

        for procedural_layer_name, texture_paths in pg_layers_by_type.items():
            save_path = self.masks_dir / f"{procedural_layer_name}.png"
            if len(texture_paths) > 1:
                merged = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
                for texture_path in texture_paths:
                    texture = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
                    merged[texture == 255] = 255
                cv2.imwrite(str(save_path), merged)
            elif len(texture_paths) == 1:
                shutil.copyfile(texture_paths[0], save_path)
            self.logger.debug("máscara procedural %s escrita", save_path.name)
