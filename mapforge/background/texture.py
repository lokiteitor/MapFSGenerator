"""Textura procedural del background (Fase 8 del plan).

Sustituye a la imagen satélite del artefacto: ruido fBm (fractal Brownian
motion) multi-octava generado con numpy y sembrado con la seed global del
proyecto (``numpy.random.default_rng``), mapeado sobre una paleta configurable
tierra→verde → ``background_texture.png`` (PNG; la conversión a DDS queda
fuera de alcance — el editor GIANTS la acepta o el usuario la convierte).

Algoritmo (determinista por seed):

1. fBm por octavas: para la octava ``k`` se genera una rejilla aleatoria
   uniforme de lado ``base_res × 2^k`` con el RNG del proyecto y se amplía a
   ``texture_size²`` con interpolación bicúbica (value noise); amplitud
   ``persistence^k`` (0.5).
2. La suma se normaliza a [0, 1] y se pasa por una curva suave (smoothstep)
   para acentuar la separación tierra/verde.
3. Cada píxel interpola linealmente entre ``palette_low`` (tierra) y
   ``palette_high`` (verde); una segunda capa de fBm de alta frecuencia y
   poca amplitud añade variación de luminancia (evita el aspecto plano).
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


def generate_background_texture(project: "Project") -> np.ndarray:
    """Genera la textura procedural del background.

    Determinista por ``settings.seed`` (RNG independiente por stream, no
    consume el RNG global del proyecto). Devuelve un array RGB uint8 de
    ``texture_size × texture_size × 3``.
    """
    cfg = project.settings.background.procedural
    size = int(cfg.texture_size)
    if size < 2:
        raise ValueError(f"texture_size debe ser >= 2, recibido {size}")

    rng = project.fresh_rng(TEXTURE_RNG_STREAM)

    # Campo principal tierra/verde.
    field = _fbm(rng, size, int(cfg.noise_octaves))
    field = _smoothstep(field)

    low = np.asarray(cfg.palette_low, dtype=np.float64).reshape(1, 1, 3)
    high = np.asarray(cfg.palette_high, dtype=np.float64).reshape(1, 1, 3)
    rgb = low + (high - low) * field[..., None]

    # Variación fina de luminancia (fBm de alta frecuencia, ±8 %).
    detail = _fbm(rng, size, octaves=3, base_res=max(BASE_GRID_RES * 8, 16))
    rgb *= 0.92 + 0.16 * detail[..., None]

    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def write_background_texture(project: "Project", path: str | Path) -> Path:
    """Genera la textura y la escribe como PNG en ``path`` (crea el dir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = generate_background_texture(project)
    # cv2.imwrite espera BGR.
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"no se pudo escribir la textura en {path}")
    project.logger.info("textura de background escrita: %s", path)
    return path
