"""Textura procedural del background (Fase 8 del plan).

Sustituye a la imagen satélite del artefacto. Dos modos:

- **Por relieve** (``height_texture: true``, default): la textura se calcula a
  partir del DEM del background, de modo que el color sigue la orografía —
  verde en las cotas bajas, gris de roca en las medias/pendientes fuertes y
  nieve por encima de ``snow_height``. Requiere que se le pase el DEM.
- **Ruido plano** (``height_texture: false``, o sin DEM): el modo original,
  fBm multi-octava mapeado sobre la rampa ``palette_low`` → ``palette_high``
  sin relación con la altura.

Ambos son deterministas por ``settings.seed``
(``numpy.random.default_rng([seed, 80])``, independiente del dissolve).

Alineación textura ↔ DEM (FACT, verificado sobre el golden)
-----------------------------------------------------------
El exportador emite UVs planares ``u = (x_mesh − x_min)/span_x`` y
``v = (y_mesh − y_min)/span_y``. En el mesh, ``x`` crece con la **columna**
del DEM e ``y`` decrece con la **fila** (rotación 180° en Y y Z de
:mod:`mapforge.background.mesh`), luego ``u=0`` ↔ columna 0 y ``v=1`` ↔ fila 0.
Con la convención OBJ/GL (``v=1`` = borde superior de la imagen = fila 0 del
array), la textura se escribe **en la misma orientación que el DEM**: un
simple ``cv2.resize``, sin volteo.

Comprobación sobre el artefacto ``FS25_Valle_Bonito``: sus UVs cumplen
``u = (x+6144)/12288`` y ``v = 1 − (z_i3d+6144)/12288`` (idénticas a las
nuestras), y su ``background_texture.jpg`` es píxel a píxel el
``satellite/satellite_background.png`` north-up **sin voltear**
(corr. 0.99 en identidad frente a 0.02 en volteo vertical).

Si aun así el editor GIANTS renderizara la textura invertida (convención
DirectX del DDS, ``v=0`` arriba), ``texture_flip_v: true`` invierte el eje V
sin tocar el resto del pipeline.

Bandas
------
La cota de cada banda se compara contra la altura **en metros de mundo**
``dem × z_scaling_factor`` (la misma que da su Y al mesh, ver
:func:`mapforge.background.mesh.background_z_scaling_factor`), y las
transiciones se suavizan con ``band_blend`` metros de mezcla más un jitter fBm
de ``band_noise`` metros que rompe la curva de nivel perfecta. La pendiente
añade roca en los taludes fuertes (``slope_rock_deg``) e impide que la nieve
se agarre a las paredes verticales (``snow_slope_limit_deg``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from mapforge.project import Project

#: Stream del RNG del proyecto reservado para la textura del background
#: (``project.fresh_rng(stream)``): independiente del dissolve de texturas.
TEXTURE_RNG_STREAM = 80

#: Resolución de la rejilla base de la octava 0 (lado de la rejilla de ruido).
BASE_GRID_RES = 8

#: Atenuación de amplitud por octava del fBm.
PERSISTENCE = 0.5


def _fbm(
    rng: np.random.Generator,
    size: int,
    octaves: int,
    base_res: int = BASE_GRID_RES,
    persistence: float = PERSISTENCE,
) -> np.ndarray:
    """fBm multi-octava por value noise (rejillas aleatorias reescaladas).

    Devuelve un array ``float64 size²`` normalizado a [0, 1].
    """
    if octaves < 1:
        raise ValueError(f"octaves debe ser >= 1, recibido {octaves}")

    accum = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    for octave in range(octaves):
        res = min(base_res * (2**octave), size)
        grid = rng.random((res, res))
        layer = cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)
        accum += amplitude * layer
        total_amplitude += amplitude
        amplitude *= persistence

    accum /= total_amplitude
    # Normalización robusta a [0, 1] (INTER_CUBIC puede sobrepasar levemente).
    lo, hi = float(accum.min()), float(accum.max())
    if hi > lo:
        accum = (accum - lo) / (hi - lo)
    else:  # ruido degenerado (p. ej. octaves=1 y res>=size): constante
        accum = np.zeros_like(accum)
    return accum


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Curva suave clásica ``3t² − 2t³`` sobre [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


def _ramp(value: np.ndarray, start: float, width: float) -> np.ndarray:
    """Rampa suave 0→1 entre ``start`` y ``start + width`` (smoothstep).

    ``width <= 0`` degenera en un escalón duro en ``start``.
    """
    if width <= 0:
        return (value >= start).astype(np.float64)
    return _smoothstep(np.clip((value - start) / width, 0.0, 1.0))


def _slope_degrees(height_m: np.ndarray, metres_per_px: float) -> np.ndarray:
    """Pendiente del terreno en grados a partir de un raster de alturas (m)."""
    if metres_per_px <= 0:
        raise ValueError(f"metres_per_px debe ser > 0, recibido {metres_per_px}")
    dy, dx = np.gradient(height_m, metres_per_px)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def _base_layer(
    rng: np.random.Generator, size: int, octaves: int, low, high
) -> tuple[np.ndarray, np.ndarray]:
    """Capa base tierra→verde (el modo original): devuelve (rgb, campo fBm)."""
    field = _smoothstep(_fbm(rng, size, octaves))
    low_rgb = np.asarray(low, dtype=np.float64).reshape(1, 1, 3)
    high_rgb = np.asarray(high, dtype=np.float64).reshape(1, 1, 3)
    return low_rgb + (high_rgb - low_rgb) * field[..., None], field


def _dem_to_metres(project: "Project", dem_full: np.ndarray, size: int) -> np.ndarray:
    """Remuestrea el DEM a ``size²`` y lo pasa a metros de mundo.

    Misma escala Z que el mesh, de forma que las cotas de las bandas coinciden
    con la altura real de la geometría. Orientación: idéntica a la del DEM (ver
    la nota de alineación del módulo).
    """
    from mapforge.background.mesh import background_z_scaling_factor

    if dem_full.ndim != 2:
        raise ValueError(f"el DEM del background debe ser 2D: {dem_full.shape}")

    dem = dem_full.astype(np.float64)
    if dem.shape != (size, size):
        # INTER_AREA promedia al reducir (evita el aliasing de las curvas de
        # nivel); INTER_LINEAR si por lo que sea hubiera que ampliar.
        interp = cv2.INTER_AREA if dem.shape[0] >= size else cv2.INTER_LINEAR
        dem = cv2.resize(dem, (size, size), interpolation=interp)
    return dem * background_z_scaling_factor(project)


def _warn_bands_out_of_range(project: "Project", height_m: np.ndarray) -> None:
    """Avisa si una banda queda por encima del techo real del terreno."""
    cfg = project.settings.background.procedural
    top = float(height_m.max())
    for label, threshold in (("rock_height", cfg.rock_height), ("snow_height", cfg.snow_height)):
        if threshold > top:
            project.logger.warning(
                "%s=%s m está por encima de la cota máxima del background "
                "(%.1f m): esa banda no se dibujará",
                label,
                threshold,
                top,
            )


def generate_background_texture(
    project: "Project", dem_full: np.ndarray | None = None
) -> np.ndarray:
    """Genera la textura procedural del background.

    Arguments:
        project: proyecto (``settings.background.procedural`` + ``seed``; el
            modo por relieve necesita además ``project.height_scale``).
        dem_full: DEM completo del background (``background_size²`` uint16, el
            ``FULL.png`` de la fase DEM). Si es ``None`` — o si
            ``height_texture`` está desactivado — se genera la textura de ruido
            plano original, sin relación con la altura.

    Returns:
        Array RGB uint8 de ``texture_size × texture_size × 3``, en la misma
        orientación que el DEM (fila 0 = fila 0 del DEM) salvo que
        ``texture_flip_v`` esté activo.
    """
    cfg = project.settings.background.procedural
    size = int(cfg.texture_size)
    if size < 2:
        raise ValueError(f"texture_size debe ser >= 2, recibido {size}")

    rng = project.fresh_rng(TEXTURE_RNG_STREAM)
    octaves = int(cfg.noise_octaves)

    # Capa base tierra→verde (común a los dos modos).
    rgb, _ = _base_layer(rng, size, octaves, cfg.palette_low, cfg.palette_high)

    snow = np.zeros((size, size), dtype=np.float64)
    if cfg.height_texture and dem_full is not None:
        height_m = _dem_to_metres(project, dem_full, size)
        _warn_bands_out_of_range(project, height_m)

        # Jitter de la cota: rompe la curva de nivel perfecta de las bandas.
        if cfg.band_noise > 0:
            jitter = _fbm(rng, size, max(octaves - 1, 1), base_res=BASE_GRID_RES * 4)
            height_eff = height_m + float(cfg.band_noise) * (2.0 * jitter - 1.0)
        else:
            height_eff = height_m

        # El DEM es 1 px/m por contrato, así que cada texel cubre
        # dem_side/texture_size metros (se usa el lado REAL del DEM, igual que
        # hace build_background_mesh si no coincide con background_size).
        metres_per_px = float(dem_full.shape[0]) / size
        slope = _slope_degrees(height_m, metres_per_px)

        # Roca: por cota y, además, allí donde la pendiente es fuerte.
        rock = _ramp(height_eff, float(cfg.rock_height), float(cfg.band_blend))
        rock = np.maximum(
            rock, _ramp(slope, float(cfg.slope_rock_deg), float(cfg.slope_rock_blend))
        )

        # Nieve: por cota, pero no se agarra a las paredes casi verticales.
        snow = _ramp(height_eff, float(cfg.snow_height), float(cfg.band_blend))
        snow *= 1.0 - _ramp(
            slope, float(cfg.snow_slope_limit_deg), float(cfg.slope_rock_blend)
        )

        rock_rgb = np.asarray(cfg.palette_rock, dtype=np.float64).reshape(1, 1, 3)
        snow_rgb = np.asarray(cfg.palette_snow, dtype=np.float64).reshape(1, 1, 3)
        rgb = rgb * (1.0 - rock[..., None]) + rock_rgb * rock[..., None]
        rgb = rgb * (1.0 - snow[..., None]) + snow_rgb * snow[..., None]

        project.logger.info(
            "textura por relieve: cota %.1f–%.1f m, roca desde %s m, nieve "
            "desde %s m (%.1f%% de la superficie con nieve)",
            float(height_m.min()),
            float(height_m.max()),
            cfg.rock_height,
            cfg.snow_height,
            100.0 * float((snow > 0.5).mean()),
        )

    # Variación fina de luminancia (fBm de alta frecuencia, ±8 %), atenuada
    # sobre la nieve para que no se ensucie.
    detail = _fbm(rng, size, octaves=3, base_res=max(BASE_GRID_RES * 8, 16))
    amplitude = 0.16 * (1.0 - 0.7 * snow)
    rgb *= 1.0 - amplitude[..., None] / 2.0 + amplitude[..., None] * detail[..., None]

    if cfg.texture_flip_v:
        rgb = rgb[::-1]

    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def write_background_texture(
    project: "Project", path: str | Path, dem_full: np.ndarray | None = None
) -> Path:
    """Genera la textura y la escribe como PNG en ``path`` (crea el dir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = generate_background_texture(project, dem_full)
    # cv2.imwrite espera BGR.
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"no se pudo escribir la textura en {path}")
    project.logger.info("textura de background escrita: %s", path)
    return path
