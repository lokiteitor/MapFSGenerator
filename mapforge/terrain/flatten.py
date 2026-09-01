"""Aplanado del terreno bajo carreteras y farmyards.

Dos features que operan sobre el DEM completo del background
(``background_size²``, 1 px = 1 m) justo antes de escribir ``FULL.png``:

- ``background.flatten_roads`` — **réplica funcional** de la feature 3.x de
  Maps4FS. El algoritmo exacto de 3.x no está disponible (el código fuente del
  repo es 1.8.242, que no la trae) y la ingeniería inversa del golden dejó su
  regla de altura longitudinal sin resolver; ver
  ``docs/analisis_flatten_roads.md`` para la evidencia clasificada
  (FACT/HYPOTHESIS) y para las desviaciones deliberadas respecto del golden.
  Lo que sí quedó demostrado del golden y **se replica** aquí es la geometría:
  el efecto es estrictamente local a las vías, el radio del corredor escala con
  el ``width`` del schema (radio total ≈ ``2×width``), la sección transversal
  es constante y el aplanado se aplica sobre el DEM de 12288², no sobre el crop.

- ``dem.flatten_farmyard`` — **extensión propia de MapForge**, sin equivalente
  en Maps4FS. Sigue el patrón verificado de ``create_foundations``
  (``maps4fs-1.8.242/maps4fs/generator/component/background.py:104``,
  FACT-source): máscara ``fillPoly`` + altura objetivo = ``cv2.mean`` del área
  redondeada al dtype + asignación uniforme; se le añade un feather
  configurable para no dejar un escalón en el borde.

Principios de diseño (el criterio es la calidad del terreno resultante, no la
igualdad bit a bit con el golden):

1. **Sin cortes abruptos.** El peso de cada geometría es 1 en su núcleo y cae a
   0 con un ``smoothstep`` (``t²(3−2t)``) a lo largo de la banda de feather. La
   derivada es continua en los dos extremos, así que no aparece ni un escalón
   en el borde ni un pliegue en el arranque de la transición.
2. **Composición ponderada, no "el último gana".** Cada geometría acumula
   ``num += α·objetivo`` y ``den += α``; al final
   ``dem = w·(num/den) + (1−w)·dem`` con ``w = clip(den, 0, 1)``. Los cruces de
   vías y los farmyards solapados se promedian en vez de pisarse ⇒ ni escalones
   en las intersecciones ni dependencia del orden de iteración.
3. **Determinista.** Sin aleatoriedad y sin dependencia del orden de entrada
   (salvo el redondeo float32, acotado muy por debajo de 1 unidad uint16).

Orden de aplicación: **farmyards primero, carreteras después**, y el perfil de
las vías se muestrea sobre el DEM ya compuesto. Así un camino que cruza una era
hereda su altura plana dentro del recinto y sale suavemente al terreno, en vez
de que los dos aplanados compitan por los mismos píxeles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from mapforge.textures.schema import load_texture_schema

logger = logging.getLogger("mapforge.flatten")

#: Denominador mínimo al normalizar el acumulador (evita 0/0).
EPS = 1e-6

#: Máximo de muestras de eje por trozo de polilínea. Acota la ventana (y por
#: tanto la memoria) de las vías que cruzan el mapa entero; los trozos se
#: solapan lo suficiente para que la composición no note la partición.
MAX_CHUNK_SAMPLES = 4096

#: Paso de densificado del eje de una vía, en píxeles (= metros).
DENSIFY_STEP = 1.0

#: Ancho de la banda de transición de una vía cuando no se fija explícitamente,
#: en múltiplos del ``width`` (radio) de la capa. Con 2.0 la pendiente máxima
#: del talud queda en ~2.3× la del terreno original; con 1.0 sube a ~2.9× y con
#: 3.0 solo baja a ~2.1×, así que 2.0 es el punto donde deja de compensar
#: ensanchar más el corredor (medido sobre un DEM en rampa, ver
#: ``tests/test_flatten.py::test_feather_mas_ancho_suaviza_el_talud``).
DEFAULT_FEATHER_RATIO = 2.0


# --------------------------------------------------------------------- utils


def _falloff(dist: np.ndarray, core: float, feather: float) -> np.ndarray:
    """Peso α por distancia: 1 hasta ``core``, smoothstep hasta 0 en ``+feather``.

    ``smoothstep`` (``t²(3−2t)``) tiene derivada nula en t=0 y t=1, de modo que
    la superficie resultante no tiene ni escalón en el borde del núcleo ni
    pliegue donde la transición muere contra el terreno.

    Arguments:
        dist: distancia euclídea de cada píxel a la geometría, en píxeles.
        core: radio con peso 1 (puede ser 0: el núcleo es la propia geometría).
        feather: ancho de la banda de transición en píxeles. ``<= 0`` ⇒ corte
            duro en ``core`` (sin transición).

    Returns:
        Array float32 con valores en [0, 1] de la misma forma que ``dist``.
    """
    if feather <= 0:
        return (dist <= core).astype(np.float32)
    t = np.clip((dist - core) / feather, 0.0, 1.0)
    return (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)


class _Accumulator:
    """Acumulador ponderado ``(num, den)`` sobre una ventana del DEM.

    Solo reserva la ventana que ocupa la geometría (más el feather), no el DEM
    completo: para el mapa del golden son ~8232² en vez de 12288².
    """

    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        y0, y1, x0, x1 = bbox
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
        shape = (max(0, y1 - y0), max(0, x1 - x0))
        self.num = np.zeros(shape, dtype=np.float32)
        self.den = np.zeros(shape, dtype=np.float32)
        self.count = 0

    @property
    def empty(self) -> bool:
        return self.count == 0 or self.num.size == 0

    def add(
        self,
        window: tuple[int, int, int, int],
        alpha: np.ndarray,
        target: np.ndarray | float,
    ) -> None:
        """Suma la contribución de una geometría en coordenadas globales."""
        wy0, wy1, wx0, wx1 = window
        ys = slice(wy0 - self.y0, wy1 - self.y0)
        xs = slice(wx0 - self.x0, wx1 - self.x0)
        self.num[ys, xs] += alpha * target
        self.den[ys, xs] += alpha
        self.count += 1

    def compose(self, dem: np.ndarray) -> np.ndarray:
        """Devuelve una copia de ``dem`` con las contribuciones aplicadas."""
        out = dem.copy()
        if self.empty:
            return out
        weight = np.clip(self.den, 0.0, 1.0)
        value = self.num / np.maximum(self.den, EPS)
        window = dem[self.y0 : self.y1, self.x0 : self.x1].astype(np.float32)
        blended = weight * value + (1.0 - weight) * window
        info = np.iinfo(dem.dtype)
        out[self.y0 : self.y1, self.x0 : self.x1] = np.clip(
            np.rint(blended), info.min, info.max
        ).astype(dem.dtype)
        return out


def _clamp_window(
    y0: float, y1: float, x0: float, x1: float, shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Ventana entera semiabierta recortada al DEM, o None si queda vacía."""
    h, w = shape
    iy0 = max(0, int(np.floor(y0)))
    ix0 = max(0, int(np.floor(x0)))
    iy1 = min(h, int(np.ceil(y1)) + 1)
    ix1 = min(w, int(np.ceil(x1)) + 1)
    if iy1 <= iy0 or ix1 <= ix0:
        return None
    return iy0, iy1, ix0, ix1


def _sample_bilinear(dem: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Muestreo bilineal del DEM con clamp a bordes (float64).

    Se muestrea directamente del uint16 (sin convertir el DEM entero a float,
    que para 12288² serían 600 MB).
    """
    h, w = dem.shape
    x = np.clip(xs, 0.0, w - 1.0)
    y = np.clip(ys, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    v00 = dem[y0, x0].astype(np.float64)
    v01 = dem[y1, x0].astype(np.float64)
    v10 = dem[y0, x1].astype(np.float64)
    v11 = dem[y1, x1].astype(np.float64)
    return (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
    )


def _densify(points: np.ndarray, step: float = DENSIFY_STEP) -> np.ndarray:
    """Reparte muestras cada ``step`` px por longitud de arco de la polilínea.

    Arguments:
        points: array ``(N, 2)`` de vértices ``(x, y)``.

    Returns:
        Array ``(M, 2)`` float64 con el último vértice incluido. Devuelve el
        propio punto si la polilínea tiene longitud 0.
    """
    if len(points) == 1:
        return points.astype(np.float64).copy()
    deltas = np.diff(points, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    total = float(lengths.sum())
    if total <= 0:
        return points[:1].astype(np.float64).copy()

    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    n = max(1, int(np.ceil(total / step)))
    targets = np.linspace(0.0, total, n + 1)
    xs = np.interp(targets, cumulative, points[:, 0].astype(np.float64))
    ys = np.interp(targets, cumulative, points[:, 1].astype(np.float64))
    return np.column_stack((xs, ys))


def _smooth_profile(profile: np.ndarray, window_px: float) -> np.ndarray:
    """Media móvil centrada de ``window_px`` px con padding por réplica.

    Es el paso que quita baches y ondulaciones del eje sin borrar las pendientes
    reales del terreno: una rampa constante sobrevive intacta a una media móvil.
    """
    size = int(round(window_px))
    if size < 3 or profile.size < 2:
        return profile
    if size % 2 == 0:
        size += 1
    pad = size // 2
    padded = np.pad(profile, pad, mode="edge")
    kernel = np.ones(size, dtype=np.float64) / size
    return np.convolve(padded, kernel, mode="valid")


def _voronoi_propagate(
    mask: np.ndarray, *value_imgs: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Distancia al eje y valores del punto de eje MÁS CERCANO, por píxel.

    Es lo que produce la sección transversal constante observada en el golden:
    cada píxel del corredor toma la altura del punto de eje más próximo, no una
    media del entorno.

    Arguments:
        mask: uint8 con 255 en los píxeles del eje.
        value_imgs: arrays con un valor por píxel semilla (altura, índice de
            muestra…); se propagan todos con el mismo Voronoi.

    Returns:
        ``(dist, [valores propagados])``; ``dist`` en píxeles.
    """
    inverted = np.where(mask > 0, 0, 255).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        inverted, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    seed_y, seed_x = np.nonzero(mask)
    seed_labels = labels[seed_y, seed_x]
    # El id de label de cada píxel semilla se lee en el propio píxel, así no se
    # depende del orden interno con que OpenCV los numera.
    size = int(labels.max()) + 1
    propagated = []
    for values in value_imgs:
        lut = np.zeros(size, dtype=values.dtype)
        lut[seed_labels] = values[seed_y, seed_x]
        propagated.append(lut[labels])
    return dist, propagated


# ----------------------------------------------------------------- farmyards


def flatten_farmyards(
    dem: np.ndarray,
    polygons: Iterable[Sequence[Sequence[float]]],
    *,
    offset: int = 0,
    feather: float = 8.0,
    max_relief: float | None = 10.0,
    units_per_metre: float = 257.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aplana el terreno dentro de cada polígono ``landuse=farmyard``.

    Extensión propia de MapForge (no existe en Maps4FS). El interior queda
    completamente plano a la media del área — que es lo que necesita una era
    para que los edificios y las máquinas se asienten — y la transición ocurre
    **fuera** del polígono, a lo largo de ``feather`` metros de terreno
    circundante.

    ``max_relief`` es una salvaguarda necesaria: ``landuse=farmyard`` se usa en
    OSM con mucha manga ancha y no es raro encontrar polígonos de cientos de
    hectáreas (en el mapa de validación hay uno de 513 ha con 78 m de desnivel).
    Aplanar eso a su media no mejora el terreno, lo destruye: dejaría una meseta
    artificial rodeada de un talud de decenas de metros. Los recintos cuyo
    desnivel interior supere ``max_relief`` metros se descartan con warning.

    Arguments:
        dem: DEM uint16 (se devuelve una copia; el original no se toca).
        polygons: anillos en coordenadas de píxel del mapa, tal y como los deja
            ``info_layers/textures.json`` bajo la clave ``farmyards``.
        offset: desplazamiento a sumar a las coordenadas para pasar del marco
            del mapa al del DEM del background (``(background−map)//2``).
        feather: ancho de la transición en metros (1 px = 1 m). ``0`` = corte
            duro en el borde del polígono.
        max_relief: desnivel interior máximo en metros. ``None`` o ``0``
            desactiva la salvaguarda (se aplanan todos los recintos).
        units_per_metre: unidades uint16 por metro del DEM
            (``65535 / height_scale``; 257.0 con el height_scale del golden).

    Returns:
        ``(dem_aplanado, stats)``.
    """
    polygons = list(polygons)
    stats: dict[str, Any] = {
        "input": len(polygons),
        "applied": 0,
        "skipped": 0,
        "skipped_by_relief": [],
        "feather": float(feather),
        "max_relief": max_relief,
    }
    if not polygons:
        return dem.copy(), stats
    relief_limit = (
        float(max_relief) * float(units_per_metre) if max_relief else float("inf")
    )

    feather = max(0.0, float(feather))
    pad = feather + 2.0
    rings: list[tuple[int, np.ndarray]] = []
    for index, ring in enumerate(polygons):
        pts = np.asarray(ring, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            stats["skipped"] += 1
            continue
        rings.append((index, pts + float(offset)))

    bbox = _global_bbox([pts for _, pts in rings], pad, dem.shape)
    if bbox is None:
        stats["skipped"] += len(rings)
        return dem.copy(), stats
    acc = _Accumulator(bbox)

    for index, pts in rings:
        window = _clamp_window(
            pts[:, 1].min() - pad,
            pts[:, 1].max() + pad,
            pts[:, 0].min() - pad,
            pts[:, 0].max() + pad,
            dem.shape,
        )
        if window is None:
            stats["skipped"] += 1
            continue
        wy0, wy1, wx0, wx1 = window
        mask = np.zeros((wy1 - wy0, wx1 - wx0), dtype=np.uint8)
        local = np.round(pts - [wx0, wy0]).astype(np.int32)
        cv2.fillPoly(mask, [local], 255)
        if not mask.any():
            # Polígono degenerado o completamente fuera del DEM.
            stats["skipped"] += 1
            continue

        dem_window = dem[wy0:wy1, wx0:wx1]
        inside = mask > 0
        values = dem_window[inside]
        relief = float(values.max() - values.min())
        if relief > relief_limit:
            stats["skipped"] += 1
            stats["skipped_by_relief"].append(
                {
                    "index": index,
                    "area_px": int(inside.sum()),
                    "relief_m": round(relief / float(units_per_metre), 2),
                }
            )
            continue

        target = float(np.round(cv2.mean(dem_window, mask=mask)[0]))
        if feather > 0:
            dist = cv2.distanceTransform(
                np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5
            )
            alpha = _falloff(dist, 0.0, feather)
        else:
            alpha = (mask > 0).astype(np.float32)
        acc.add(window, alpha, target)
        stats["applied"] += 1

    if stats["skipped_by_relief"]:
        logger.warning(
            "flatten_farmyard: %d recinto(s) descartados por superar %.1f m de "
            "desnivel interior (probable landuse=farmyard sobredimensionado en "
            "el OSM): %s",
            len(stats["skipped_by_relief"]),
            float(max_relief or 0.0),
            ", ".join(
                f"#{item['index']} ({item['relief_m']} m, "
                f"{item['area_px'] / 10000:.0f} ha)"
                for item in stats["skipped_by_relief"]
            ),
        )
    logger.info(
        "flatten_farmyard: %d/%d farmyards aplanados (feather %.1f m)",
        stats["applied"],
        stats["input"],
        feather,
    )
    return acc.compose(dem), stats


# --------------------------------------------------------------------- roads


def road_widths_from_schema(layers: Iterable[Any]) -> dict[str, int]:
    """Mapa ``str(tags) -> width`` de las capas del schema con vías.

    La clave es el string que el motor de texturas guarda en cada entrada de
    ``roads_polylines`` (``str(layer.tags)``), así que sirve tal cual para
    resolver el ancho de cada polilínea. ``width`` es el RADIO del buffer
    (§S3 del informe forense).
    """
    widths: dict[str, int] = {}
    for layer in layers:
        if getattr(layer, "info_layer", None) != "roads":
            continue
        if not getattr(layer, "width", None) or layer.tags is None:
            continue
        widths[str(layer.tags)] = int(layer.width)
    return widths


def _global_bbox(
    geometries: Sequence[np.ndarray], pad: float, shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Ventana que cubre todas las geometrías más ``pad``, recortada al DEM."""
    if not geometries:
        return None
    y0 = min(float(g[:, 1].min()) for g in geometries) - pad
    y1 = max(float(g[:, 1].max()) for g in geometries) + pad
    x0 = min(float(g[:, 0].min()) for g in geometries) - pad
    x1 = max(float(g[:, 0].max()) for g in geometries) + pad
    return _clamp_window(y0, y1, x0, x1, shape)


def _iter_chunks(n: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Parte el eje densificado en trozos solapados con tramo propio.

    Acota la ventana (y por tanto la memoria) de las vías que cruzan el mapa
    entero. Cada trozo se calcula con ``overlap`` muestras de contexto a cada
    lado pero solo **aporta** los píxeles cuya muestra de eje más cercana cae en
    su tramo propio, y los tramos propios teselan ``[0, n)`` sin solaparse. Así
    la partición no altera el resultado: ni costuras ni doble conteo de α (que
    estrecharía el feather justo en las zonas comunes).

    Returns:
        Lista de ``(inicio, fin, propio_inicio, propio_fin)`` en índices de
        muestra; ``[inicio, fin)`` es el contexto y ``[propio_inicio,
        propio_fin)`` el tramo aportado.
    """
    if n <= MAX_CHUNK_SAMPLES:
        return [(0, n, 0, n)]
    step = max(1, MAX_CHUNK_SAMPLES - 2 * overlap)
    chunks: list[tuple[int, int, int, int]] = []
    start = 0
    while True:
        end = min(n, start + MAX_CHUNK_SAMPLES)
        last = end >= n
        own0 = 0 if not chunks else start + overlap
        own1 = n if last else start + step + overlap
        chunks.append((start, end, own0, min(own1, n)))
        if last:
            break
        start += step
    return chunks


def flatten_roads(
    dem: np.ndarray,
    polylines: Iterable[dict[str, Any]],
    widths: dict[str, int],
    *,
    offset: int = 0,
    feather: float | None = None,
    smooth: float = 25.0,
    default_width: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aplana el corredor de cada vía y lo integra con el terreno colindante.

    Para cada polilínea: se densifica el eje a 1 px, se muestrea el DEM
    (bilineal), se suaviza el perfil longitudinal, y cada píxel del corredor
    toma la altura del punto de eje más cercano (sección transversal plana). El
    peso vale 1 hasta ``width`` px del eje — el ancho real de la calzada — y cae
    con smoothstep hasta 0 en ``width + feather``.

    Arguments:
        dem: DEM uint16 (se devuelve una copia).
        polylines: entradas de ``roads_polylines`` de ``textures.json``:
            ``{"points": [[x, y], ...], "tags": "<str(tags) de la capa>"}``.
        widths: ``str(tags) -> width`` (radio en px), de
            :func:`road_widths_from_schema`.
        offset: desplazamiento del marco del mapa al del DEM del background.
        feather: ancho de la transición en metros. ``None`` = proporcional al
            ancho de la vía (``feather = DEFAULT_FEATHER_RATIO × width``), que
            es el default recomendado: prioriza que el talud se integre con el
            terreno sobre reproducir el corredor más estrecho del golden
            (``2×width`` de radio total).
        smooth: ventana de la media móvil longitudinal en metros. ``0`` = off.
        default_width: ancho para polilíneas con tags desconocidos. ``None`` =
            se descartan (con warning).

    Returns:
        ``(dem_aplanado, stats)``.
    """
    entries = list(polylines)
    stats: dict[str, Any] = {
        "input": len(entries),
        "applied": 0,
        "skipped": 0,
        "unknown_tags": [],
        "feather": feather,
        "smooth": float(smooth),
    }
    if not entries:
        return dem.copy(), stats

    smooth = max(0.0, float(smooth))

    # 1. Resolver geometría y ancho de cada vía.
    resolved: list[tuple[np.ndarray, float, float]] = []  # (puntos, core, feather)
    unknown: set[str] = set()
    for entry in entries:
        pts = np.asarray(entry.get("points", ()), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] != 2:
            stats["skipped"] += 1
            continue
        tags = str(entry.get("tags", ""))
        width = widths.get(tags, default_width)
        if not width:
            unknown.add(tags)
            stats["skipped"] += 1
            continue
        core = float(width)
        band = (
            core * DEFAULT_FEATHER_RATIO
            if feather is None
            else max(0.0, float(feather))
        )
        resolved.append((pts + float(offset), core, band))

    if unknown:
        stats["unknown_tags"] = sorted(unknown)
        logger.warning(
            "flatten_roads: %d polilíneas descartadas por tags sin width en el "
            "texture schema: %s",
            stats["skipped"],
            ", ".join(sorted(unknown)),
        )
    if not resolved:
        return dem.copy(), stats

    max_pad = max(core + band for _, core, band in resolved) + 2.0
    bbox = _global_bbox([pts for pts, _, _ in resolved], max_pad, dem.shape)
    if bbox is None:
        stats["skipped"] += len(resolved)
        return dem.copy(), stats
    acc = _Accumulator(bbox)

    for pts, core, band in resolved:
        radius = core + band
        pad = radius + 2.0
        samples = _densify(pts)
        # Perfil longitudinal: DEM bajo el eje, suavizado.
        profile = _sample_bilinear(dem, samples[:, 0], samples[:, 1])
        if smooth:
            profile = _smooth_profile(profile, smooth)

        overlap = int(np.ceil(radius)) + 2
        applied = False
        for start, end, own0, own1 in _iter_chunks(len(samples), overlap):
            chunk = samples[start:end]
            window = _clamp_window(
                chunk[:, 1].min() - pad,
                chunk[:, 1].max() + pad,
                chunk[:, 0].min() - pad,
                chunk[:, 0].max() + pad,
                dem.shape,
            )
            if window is None:
                continue
            wy0, wy1, wx0, wx1 = window
            shape = (wy1 - wy0, wx1 - wx0)
            mask = np.zeros(shape, dtype=np.uint8)
            height_img = np.zeros(shape, dtype=np.float32)
            index_img = np.zeros(shape, dtype=np.int32)

            ix = np.round(chunk[:, 0]).astype(np.int64) - wx0
            iy = np.round(chunk[:, 1]).astype(np.int64) - wy0
            inside = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
            if not inside.any():
                continue
            mask[iy[inside], ix[inside]] = 255
            height_img[iy[inside], ix[inside]] = profile[start:end][inside]
            indices = np.arange(start, end, dtype=np.int32)
            index_img[iy[inside], ix[inside]] = indices[inside]

            dist, (target, nearest) = _voronoi_propagate(mask, height_img, index_img)
            alpha = _falloff(dist, core, band)
            if own0 > 0 or own1 < len(samples):
                # Solo los píxeles cuya muestra más cercana es de este tramo.
                alpha *= ((nearest >= own0) & (nearest < own1)).astype(np.float32)
            if not alpha.any():
                continue
            acc.add(window, alpha, target)
            applied = True

        if applied:
            stats["applied"] += 1
        else:
            stats["skipped"] += 1

    logger.info(
        "flatten_roads: %d/%d vías aplanadas (feather %s, suavizado %.0f m)",
        stats["applied"],
        stats["input"],
        "proporcional al ancho" if feather is None else f"{feather:.1f} m",
        smooth,
    )
    return acc.compose(dem), stats


# ---------------------------------------------------------------- orquestador


def load_flatten_geometry(
    textures_json: Path, texture_schema: Path
) -> tuple[list[dict[str, Any]], dict[str, int], list[list[list[int]]]]:
    """Lee ``info_layers/textures.json`` + el texture schema.

    Returns:
        ``(roads_polylines, widths, farmyards)``. Si el JSON no existe se
        devuelven listas vacías (el aplanado se omite con warning aguas arriba).
    """
    if not Path(textures_json).is_file():
        return [], {}, []
    with open(textures_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    widths = road_widths_from_schema(load_texture_schema(texture_schema))
    return (
        list(data.get("roads_polylines", ())),
        widths,
        list(data.get("farmyards", ())),
    )


def flatten_terrain(
    dem: np.ndarray,
    *,
    offset: int,
    roads: Iterable[dict[str, Any]] | None = None,
    road_widths: dict[str, int] | None = None,
    farmyards: Iterable[Sequence[Sequence[float]]] | None = None,
    roads_feather: float | None = None,
    roads_smooth: float = 25.0,
    farmyard_feather: float = 8.0,
    farmyard_max_relief: float | None = 10.0,
    units_per_metre: float = 257.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aplica farmyards y luego carreteras sobre el mismo DEM.

    El orden importa y es deliberado: los farmyards se aplanan primero y el
    perfil de las vías se muestrea sobre el resultado, de modo que un camino que
    atraviesa una era hereda su altura plana dentro del recinto y sale con la
    transición suave hacia el terreno.

    Arguments:
        dem: DEM uint16 del background (``background_size²``).
        offset: ``(background_size − map_size) // 2``.
        roads / farmyards: geometrías; ``None`` o vacío desactiva esa parte.
        road_widths: ``str(tags) -> width``.

    Returns:
        ``(dem_aplanado, stats)`` con las estadísticas de ambas fases.
    """
    stats: dict[str, Any] = {}
    result = dem
    if farmyards:
        result, stats["farmyards"] = flatten_farmyards(
            result,
            farmyards,
            offset=offset,
            feather=farmyard_feather,
            max_relief=farmyard_max_relief,
            units_per_metre=units_per_metre,
        )
    if roads:
        result, stats["roads"] = flatten_roads(
            result,
            roads,
            road_widths or {},
            offset=offset,
            feather=roads_feather,
            smooth=roads_smooth,
        )
    if result is dem:
        result = dem.copy()
    return result, stats
