"""Pipeline DEM completo (Fase 1 del plan; algoritmo S2 del informe forense).

Replica el pipeline de Maps4FS 1.8.242 (``dem.py`` + ``component/background.py``)
con la semántica de entrada de MapForge: el heightmap es un PNG uint16 a
1 px/m del tamaño background (``map_size + 2×2048``) y
``metros = valor × input_height_scale`` (default 255/65535, la convención del
artefacto golden).

Orden exacto del modo raw (S2, FACT-source):

1. cargar heightmap uint16 (se redimensiona a ``background_size²`` con
   INTER_LINEAR si no viene ya a ese tamaño)
2. a metros: ``valor × input_height_scale``
3. ``× multiplier`` (omitido si es 1, como Maps4FS)
4. ``adjust_terrain_to_ground_level``: shift para que
   ``min = plateau + water_depth``
5. ``height_scale = ceil(max(minimum_height_scale, max + ceiling))`` → se
   guarda en ``project.height_scale`` (lo consume el escritor i3d, Fase 7) y
   en ``dem_info.json``
6. normalizar ``clip(v / height_scale × 65535, 0, 65535)`` → uint16 (astype,
   trunca — idéntico a Maps4FS)
7. blur gaussiano DESPUÉS de normalizar: kernel ``(r, r)`` con r impar
   (r par → r+1; r ≤ 0 → sin blur), ``sigmaX = sigmaY = 10``

Y la fase de background (``component/background.py.process``), que produce los
ficheros en este orden (el mismo estado final que Maps4FS):

- ``background/not_substracted.png`` = DEM completo ANTES de restar agua
- ``background/not_resized.png`` = crop central ``map_size²`` (pre-resta)
- [hook] si hay máscara de agua y ``water_depth`` > 0: restar
  ``water_depth × 65535/height_scale`` bajo la máscara (erosión 3×3 ×1)
- ``background/FULL.png`` = DEM completo (post-resta si hubo)
- ``map/data/dem.png`` = crop central de FULL → resize ``(map_size+1)²``
  INTER_LINEAR

Modo ``custom_dem`` (espejo de ``custom_dem: true`` del ``main_settings.json``
del golden): el heightmap YA es el DEM final uint16 normalizado — se omiten
multiplier/shift/normalización/blur y el array pasa tal cual a la fase de
background. ``height_scale`` se calcula con la misma fórmula del paso 5 sobre
``valor × input_height_scale`` (para el golden: ceil(max(255, 114.94+15)) =
255, exactamente el ``adjusted_height_scale`` de su ``generation_info.json``).

Utilidades reutilizables (Fase 6, splines):

- :func:`spline_z` / :meth:`DemPipeline.spline_z`:
  ``z = dem_not_resized[y, x] × 1/multiplier × height_scale/65535`` con clamp
  a bordes (réplica exacta de ``i3d.py.get_z_coordinate_from_dem`` +
  ``get_z_scaling_factor(ignore_height_scale_multiplier=True)``).

Limitaciones documentadas:

- ``rotation != 0`` no está soportado en esta fase (el golden usa 0).
- ``flatten_roads`` es una feature 3.x no replicada (ver
  ``docs/analisis_forense_maps4fs.md`` §S9): el dem.png del golden se generó
  desde el DEM con carreteras aplanadas y con un resize tipo NEAREST de 3.x;
  nuestro INTER_LINEAR (FACT-source 1.8) produce diferencias sub-métricas que
  la validación de fase cuantifica.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from mapforge.project import Project

logger = logging.getLogger("mapforge.dem")

#: Dtypes de heightmap aceptados (el contrato de MapForge es uint16).
ACCEPTED_DTYPES = (np.uint16,)


def spline_z(
    dem_not_resized: np.ndarray,
    x: float,
    y: float,
    multiplier: float,
    height_scale: int,
) -> float:
    """Muestrea la altura Z para splines desde el DEM ``not_resized``.

    Réplica exacta de Maps4FS ``i3d.py.get_z_coordinate_from_dem``:
    sin interpolación, clamp a bordes, y
    ``z = dem[y, x] × (1/multiplier) × (height_scale/65535)``.

    Arguments:
        dem_not_resized: DEM uint16 ``map_size²`` (crop central, pre-resta).
        x: columna en píxeles (se trunca a int y se recorta a los bordes).
        y: fila en píxeles (ídem).
        multiplier: ``dem_settings.multiplier`` usado al generar el DEM.
        height_scale: height_scale calculado por el pipeline.

    Returns:
        Altura en metros (float).
    """
    dem_x_size, dem_y_size = dem_not_resized.shape[:2]
    xi = int(max(0, min(x, dem_x_size - 1)))
    yi = int(max(0, min(y, dem_y_size - 1)))
    z = float(dem_not_resized[yi, xi])
    return z * (1.0 / multiplier) * (height_scale / 65535.0)


class DemPipeline:
    """Genera ``map/data/dem.png`` y los DEM intermedios del background.

    Tras :meth:`run` expone:

    - ``height_scale`` (int): también se copia a ``project.height_scale``.
    - ``mesh_z_scaling_factor`` (float): ``65535 / height_scale``.
    - ``height_scale_multiplier`` (float): ``height_scale / 255``.
    - ``dem_full`` (uint16 ``background_size²``): estado final de FULL.png.
    - ``dem_not_resized`` (uint16 ``map_size²``): crop central pre-resta
      (la fuente de muestreo de splines).
    - ``dem_map`` (uint16 ``(map_size+1)²``): contenido de map/data/dem.png.
    - ``info`` (dict): estadísticas por etapa + height_scale (se vuelca a
      ``dem_info.json`` en el directorio de salida).

    Arguments:
        project: proyecto MapForge.
        custom_dem: si True, el heightmap ya es el DEM final uint16
            normalizado (espejo del ``custom_dem`` de Maps4FS) y se omite el
            procesado S2. Si None, se lee ``settings.dem.custom_dem`` cuando
            exista (default False).
    """

    def __init__(self, project: "Project", custom_dem: bool | None = None) -> None:
        self.project = project
        self.logger = logger
        if custom_dem is None:
            custom_dem = bool(getattr(project.settings.dem, "custom_dem", False))
        self.custom_dem = custom_dem

        self.height_scale: int | None = None
        self.mesh_z_scaling_factor: float | None = None
        self.height_scale_multiplier: float | None = None
        self.dem_full: np.ndarray | None = None
        self.dem_not_resized: np.ndarray | None = None
        self.dem_map: np.ndarray | None = None
        self.info: dict[str, Any] = {}

    # ------------------------------------------------------------ pipeline

    def run(self, water_mask: np.ndarray | None = None) -> int:
        """Ejecuta el pipeline completo y escribe los PNG de salida.

        Arguments:
            water_mask: máscara uint8 ``background_size²`` con 255 en agua
                (hook de la resta de ``water_depth``; None = sin resta, el
                default de la Fase 1 — la máscara la produce la Fase 3).

        Returns:
            El ``height_scale`` calculado.
        """
        project = self.project
        if project.map.rotation:
            raise ValueError(
                "DemPipeline: rotation != 0 no está soportado en la Fase 1 "
                f"(rotation={project.map.rotation})"
            )
        if project.map.output_size:
            self.logger.warning(
                "DemPipeline: output_size=%s se ignora (no soportado en Fase 1)",
                project.map.output_size,
            )

        data = self._load_heightmap()
        self.update_info("original", data)

        if self.custom_dem:
            full_raw = self._process_custom(data)
        else:
            full_raw = self._process_raw(data)
        self.update_info("full", full_raw)

        self._save_outputs(full_raw, water_mask)
        self._write_dem_info()

        assert self.height_scale is not None
        return self.height_scale

    # ------------------------------------------------------------- etapas

    def _load_heightmap(self) -> np.ndarray:
        """Carga y valida el heightmap uint16 de entrada."""
        path = self.project.paths.heightmap
        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise FileNotFoundError(f"DemPipeline: no se pudo leer el heightmap {path}")
        if data.ndim != 2:
            raise ValueError(
                f"DemPipeline: el heightmap debe ser de 1 canal, tiene shape {data.shape}"
            )
        if data.dtype not in ACCEPTED_DTYPES:
            raise ValueError(
                f"DemPipeline: el heightmap debe ser uint16, es {data.dtype}"
            )
        if not np.any(data):
            raise ValueError("DemPipeline: el heightmap está completamente a cero")
        self.logger.info(
            "heightmap %s: shape=%s dtype=%s min=%s max=%s",
            path.name,
            data.shape,
            data.dtype,
            data.min(),
            data.max(),
        )
        return data

    def _process_raw(self, data: np.ndarray) -> np.ndarray:
        """Modo raw: pipeline S2 completo sobre el heightmap de entrada."""
        settings = self.project.settings.dem
        background_size = self.project.map.background_size

        # 2. a metros (float64 para no perder precisión en el round-trip).
        meters = data.astype(np.float64) * self.project.settings.input_height_scale

        # 1./2b. resize a la resolución de salida (background_size²).
        if meters.shape != (background_size, background_size):
            self.logger.info(
                "redimensionando heightmap %s -> %s (INTER_LINEAR)",
                meters.shape,
                (background_size, background_size),
            )
            meters = cv2.resize(
                meters,
                (background_size, background_size),
                interpolation=cv2.INTER_LINEAR,
            )
        self.update_info("meters", meters)

        # 3. multiplier (Maps4FS lo omite si es exactamente 1).
        if settings.multiplier != 1:
            meters = meters * settings.multiplier
            self.update_info("multiplied", meters)

        # 4. shift para que min = plateau + water_depth.
        if settings.adjust_terrain_to_ground_level:
            desired_ground_level = settings.plateau + settings.water_depth
            meters = meters + (desired_ground_level - meters.min())
            self.update_info("raised_lowered", meters)

        # 5. height_scale con ceiling.
        height_scale = self._determine_height_scale(float(meters.max()))

        # 6. normalizar a uint16 (astype trunca, como Maps4FS).
        normalized = np.clip((meters / height_scale) * 65535, 0, 65535).astype(
            np.uint16
        )
        del meters
        self.update_info("normalized", normalized)

        # 7. blur gaussiano (después de normalizar).
        blur_radius = self._effective_blur_radius(settings.blur_radius)
        if blur_radius:
            normalized = cv2.GaussianBlur(
                normalized, (blur_radius, blur_radius), sigmaX=10, sigmaY=10
            )
            self.update_info("blurred", normalized)
        return normalized

    def _process_custom(self, data: np.ndarray) -> np.ndarray:
        """Modo custom_dem: el heightmap ya es el DEM final normalizado."""
        background_size = self.project.map.background_size
        if data.shape != (background_size, background_size):
            raise ValueError(
                "DemPipeline (custom_dem): el heightmap debe medir "
                f"{background_size}² (background_size), mide {data.shape}"
            )
        max_meters = float(data.max()) * self.project.settings.input_height_scale
        self._determine_height_scale(max_meters)
        self.logger.info(
            "modo custom_dem: DEM final de entrada usado tal cual (sin S2)"
        )
        return data

    def _determine_height_scale(self, max_meters: float) -> int:
        """``ceil(max(minimum_height_scale, max + ceiling))`` + derivados."""
        settings = self.project.settings.dem
        adjusted = math.ceil(
            max(settings.minimum_height_scale, max_meters + settings.ceiling)
        )
        self.height_scale = adjusted
        self.mesh_z_scaling_factor = 65535 / adjusted
        self.height_scale_multiplier = adjusted / 255
        self.project.height_scale = adjusted
        self.info["height_scale"] = {
            "height_scale_from_settings": settings.minimum_height_scale,
            "adjusted_height_scale": adjusted,
            "mesh_z_scaling_factor": self.mesh_z_scaling_factor,
            "height_scale_multiplier": self.height_scale_multiplier,
        }
        self.logger.info("height_scale calculado: %s", adjusted)
        return adjusted

    @staticmethod
    def _effective_blur_radius(blur_radius: int | None) -> int:
        """Radio efectivo del blur: 0/negativo → off; par → +1 (Maps4FS)."""
        if blur_radius is None or blur_radius <= 0:
            return 0
        if blur_radius % 2 == 0:
            return blur_radius + 1
        return blur_radius

    # ------------------------------------------------- background/outputs

    def _save_outputs(
        self, full_raw: np.ndarray, water_mask: np.ndarray | None
    ) -> None:
        """Réplica de ``background.py.process``: escribe los 4 PNG."""
        project = self.project
        map_size = project.map.size
        paths = project.paths

        paths.background_dir.mkdir(parents=True, exist_ok=True)
        paths.map_data_dir.mkdir(parents=True, exist_ok=True)

        # not_substracted = DEM completo antes de la resta de agua.
        not_substracted_path = paths.background_dir / "not_substracted.png"
        cv2.imwrite(str(not_substracted_path), full_raw)

        # not_resized = crop central map_size² (pre-resta, como Maps4FS).
        self.dem_not_resized = self._cut_out_center(full_raw, map_size // 2)
        self.update_info("not_resized", self.dem_not_resized)
        cv2.imwrite(str(paths.background_dir / "not_resized.png"), self.dem_not_resized)

        # Hook water_depth: resta bajo máscara (off por defecto en Fase 1).
        full = full_raw
        water_depth = project.settings.dem.water_depth
        if water_mask is not None and water_depth:
            full = self._subtract_water(full_raw.copy(), water_mask, water_depth)
            self.update_info("subtracted", full)
        self.dem_full = full
        cv2.imwrite(str(paths.background_dir / "FULL.png"), full)

        # dem.png = crop central de FULL (post-resta) → resize (map_size+1)².
        cutout = self._cut_out_center(full, map_size // 2)
        output_size = map_size + 1
        self.dem_map = cv2.resize(
            cutout, (output_size, output_size), interpolation=cv2.INTER_LINEAR
        )
        self.update_info("dem", self.dem_map)
        cv2.imwrite(str(paths.dem_png), self.dem_map)
        self.logger.info(
            "DEM guardado: %s (%s²), FULL/not_resized/not_substracted en %s",
            paths.dem_png,
            output_size,
            paths.background_dir,
        )

    @staticmethod
    def _cut_out_center(image: np.ndarray, half_size: int) -> np.ndarray:
        """Crop central ``2×half_size`` (réplica de ``cut_out_np``)."""
        center = (image.shape[0] // 2, image.shape[1] // 2)
        x1, x2 = center[0] - half_size, center[0] + half_size
        y1, y2 = center[1] - half_size, center[1] + half_size
        return image[x1:x2, y1:y2]

    def _subtract_water(
        self, image: np.ndarray, water_mask: np.ndarray, water_depth: float
    ) -> np.ndarray:
        """Resta ``water_depth × mesh_z_scaling_factor`` bajo la máscara.

        Réplica de ``subtract_by_mask`` (mask_by=255, erosión 3×3 ×1). La
        resta se hace en uint16 con wraparound, igual que Maps4FS.
        """
        assert self.mesh_z_scaling_factor is not None
        subtract_by = int(water_depth * self.mesh_z_scaling_factor)
        mask = water_mask == 255
        mask = cv2.erode(
            mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
        ).astype(bool)
        image[mask] = image[mask] - subtract_by
        self.logger.info(
            "water_depth restado: %s unidades uint16 bajo %s px de agua",
            subtract_by,
            int(mask.sum()),
        )
        return image

    # ------------------------------------------------------------ info/API

    def update_info(self, state: str, data: np.ndarray) -> None:
        """Registra estadísticas de una etapa (como ``DEM.update_info``)."""
        self.info[state] = {
            "min": float(data.min()),
            "max": float(data.max()),
            "deviation": float(data.max() - data.min()),
            "dtype": str(data.dtype),
            "shape": str(data.shape),
        }

    def _write_dem_info(self) -> None:
        """Vuelca ``dem_info.json`` al directorio de salida."""
        payload = {
            "mode": "custom_dem" if self.custom_dem else "raw",
            "input_height_scale": self.project.settings.input_height_scale,
            **self.info,
        }
        path = self.project.paths.output_dir / "dem_info.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        self.logger.info("dem_info.json escrito: %s", path)

    def spline_z(self, x: float, y: float) -> float:
        """Altura Z en metros para splines en ``(x, y)`` (ver :func:`spline_z`)."""
        if self.dem_not_resized is None or self.height_scale is None:
            raise RuntimeError("DemPipeline.spline_z: ejecuta run() primero")
        return spline_z(
            self.dem_not_resized,
            x,
            y,
            self.project.settings.dem.multiplier,
            self.height_scale,
        )
