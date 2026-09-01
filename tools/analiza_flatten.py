"""Análisis del aplanado de terreno: evidencia forense + métricas de calidad.

Produce ``output/analisis_flatten_roads/`` con:

- ``evidencia_golden.json`` — lo que se puede medir del par antes/después del
  golden (``background/not_resized.png`` vs
  ``background/not_resized_with_flattened_roads.png``): píxeles afectados,
  radio del corredor por clase de vía y sección transversal de ejemplo. Es la
  base de los FACT de ``docs/analisis_flatten_roads.md``.
- ``comparacion_golden.json`` — el DEM aplanado por MapForge frente al del
  golden (métrica informativa: el objetivo no es la igualdad bit a bit).
- ``calidad.json`` — las métricas que sí son criterio: pendiente máxima del
  talud en el margen del corredor comparada con la del terreno original, más
  algunos perfiles transversales antes/después.
- ``suavidad_splines.json`` — el bacheo que heredarían los CVs de tráfico,
  muestreando el eje sobre el DEM crudo y sobre el aplanado.
- ``splines.json`` — CVs de ``map/splines.i3d`` que difieren del golden en Y.

Uso::

    ./venv/bin/python tools/analiza_flatten.py \
        [--golden FS25_Valle_Bonito] [--output output/valle_bonito] \
        [--dest output/analisis_flatten_roads]

Todas las secciones son opcionales: si falta un artefacto (p. ej. no se ha
generado aún la salida de MapForge) esa sección se omite con un aviso.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mapforge.terrain.flatten import _densify, road_widths_from_schema  # noqa: E402
from mapforge.textures.schema import load_texture_schema  # noqa: E402

#: Conversión a metros del artefacto golden: altura_m = uint16 / 257.
UINT16_PER_METRE = 257.0

#: Radios de sondeo (px) al medir hasta dónde llega el efecto de cada vía.
PROBE_RADII = range(0, 26)


def _imread(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _delta_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Estadísticas de la diferencia b − a en unidades uint16 y en metros."""
    delta = b.astype(np.int64) - a.astype(np.int64)
    nonzero = delta != 0
    count = int(nonzero.sum())
    stats: dict[str, Any] = {
        "pixels_diff": count,
        "pixels_total": int(delta.size),
        "pct_diff": round(100.0 * count / delta.size, 4),
    }
    if count:
        magnitudes = np.abs(delta[nonzero])
        stats.update(
            {
                "max_delta_u16": int(magnitudes.max()),
                "max_delta_m": round(float(magnitudes.max()) / UINT16_PER_METRE, 4),
                "mean_delta_u16": round(float(magnitudes.mean()), 4),
                "mean_delta_m": round(float(magnitudes.mean()) / UINT16_PER_METRE, 5),
                "p50_u16": int(np.percentile(magnitudes, 50)),
                "p99_u16": int(np.percentile(magnitudes, 99)),
                "pct_gt_1m": round(
                    100.0 * int((magnitudes > UINT16_PER_METRE).sum()) / delta.size, 4
                ),
            }
        )
    return stats


def _road_classes(
    polylines: list[dict[str, Any]], widths: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in polylines:
        tags = str(entry.get("tags", ""))
        if tags in widths:
            grouped.setdefault(tags, []).append(entry)
    return grouped


def _centreline_mask(
    entries: list[dict[str, Any]], shape: tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for entry in entries:
        points = np.asarray(entry["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [points], False, 255, thickness=1)
    return mask


# ------------------------------------------------------------ evidencia golden


def evidencia_golden(
    golden: Path, polylines: list[dict[str, Any]], widths: dict[str, int]
) -> dict[str, Any] | None:
    """Mide el par antes/después del golden: alcance y radio por clase."""
    crudo = _imread(golden / "background" / "not_resized.png")
    aplanado = _imread(golden / "background" / "not_resized_with_flattened_roads.png")
    if crudo is None or aplanado is None:
        return None

    report: dict[str, Any] = {
        "fuente": "FS25_Valle_Bonito/background/not_resized{,_with_flattened_roads}.png",
        "global": _delta_stats(crudo, aplanado),
    }

    changed = crudo != aplanado
    grouped = _road_classes(polylines, widths)
    masks = {tags: _centreline_mask(rs, crudo.shape) for tags, rs in grouped.items()}

    por_clase: dict[str, Any] = {}
    for tags, mask in masks.items():
        otras = np.zeros_like(mask)
        for other_tags, other in masks.items():
            if other_tags != tags:
                otras |= other
        # Solo se mide donde no interfieren corredores de otra clase.
        aislado = cv2.distanceTransform(
            np.where(otras > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5
        ) > 40
        dist = cv2.distanceTransform(
            np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5
        )
        perfil = []
        radio_max = None
        for r in PROBE_RADII:
            anillo = (dist >= r) & (dist < r + 1) & aislado
            total = int(anillo.sum())
            if not total:
                continue
            afectados = int((anillo & changed).sum())
            perfil.append(
                {"r": r, "pixeles": total, "afectados": afectados,
                 "fraccion": round(afectados / total, 4)}
            )
            if afectados:
                radio_max = r
        por_clase[tags] = {
            "vias": len(grouped[tags]),
            "width_schema": widths[tags],
            "radio_maximo_afectado": radio_max,
            "perfil_radial": perfil,
        }
    report["por_clase"] = por_clase

    # Sección transversal en el píxel de mayor cambio: documenta que el
    # corredor es constante a lo ancho.
    delta = np.abs(aplanado.astype(np.int64) - crudo.astype(np.int64))
    y, x = np.unravel_index(int(delta.argmax()), delta.shape)
    lo, hi = max(0, y - 22), min(crudo.shape[0], y + 23)
    report["seccion_ejemplo"] = {
        "y": int(y),
        "x": int(x),
        "original": crudo[lo:hi, x].tolist(),
        "aplanado": aplanado[lo:hi, x].tolist(),
    }

    # FACT: el aplanado se aplicó sobre el DEM completo, no sobre el crop.
    full = _imread(golden / "background" / "FULL.png")
    not_substracted = _imread(golden / "background" / "not_substracted.png")
    if full is not None and not_substracted is not None:
        half = crudo.shape[0] // 2
        centre = full.shape[0] // 2
        crop = full[centre - half : centre + half, centre - half : centre + half]
        report["aplanado_sobre_dem_completo"] = {
            "crop_full_igual_a_not_resized_flattened": bool(
                np.array_equal(crop, aplanado)
            ),
            "full_vs_not_substracted": _delta_stats(not_substracted, full),
        }
    return report


# --------------------------------------------------------- comparación golden


def comparacion_golden(golden: Path, salida: Path) -> dict[str, Any] | None:
    """El DEM aplanado por MapForge frente al del golden (informativo)."""
    resultado: dict[str, Any] = {}
    pares = {
        "not_resized_with_flattened_roads": (
            golden / "background" / "not_resized_with_flattened_roads.png",
            salida / "background" / "not_resized_with_flattened_roads.png",
        ),
        "FULL": (
            golden / "background" / "FULL.png",
            salida / "background" / "FULL.png",
        ),
        "dem": (golden / "map" / "data" / "dem.png", salida / "map" / "data" / "dem.png"),
    }
    for nombre, (ruta_golden, ruta_salida) in pares.items():
        a, b = _imread(ruta_golden), _imread(ruta_salida)
        if a is None or b is None or a.shape != b.shape:
            continue
        resultado[nombre] = _delta_stats(a, b)
    return resultado or None


# ------------------------------------------------------------------- calidad


#: Separación (px) entre cortes transversales al medir el talud de una vía.
CROSS_SECTION_STEP = 25

#: Pendiente transversal mínima del terreno original (u16/px) para que un corte
#: entre en la estadística: por debajo el terreno ya era llano y el aplanado no
#: tiene nada que integrar.
MIN_SLOPE_U16 = 5


def _cortes_transversales(
    entry: dict[str, Any], half: int, step: int = CROSS_SECTION_STEP
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Coordenadas ``(ys, xs)`` de cortes perpendiculares al eje de la vía."""
    points = np.asarray(entry["points"], dtype=np.float64)
    if len(points) < 2:
        return []
    cortes: list[tuple[np.ndarray, np.ndarray]] = []
    ks = np.arange(-half, half + 1)
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        direction = p1 - p0
        length = float(np.hypot(*direction))
        if length < 1:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        for t in np.arange(0.0, length, step) / length:
            centre = p0 + direction * t
            coords = centre + normal * ks.reshape(-1, 1)
            cortes.append((coords[:, 1], coords[:, 0]))
    return cortes


def calidad(
    salida: Path, polylines: list[dict[str, Any]], widths: dict[str, int]
) -> dict[str, Any] | None:
    """Cuánto se empina el margen del corredor respecto del terreno original.

    Es la métrica de aceptación: un aplanado con corte duro dejaría en el borde
    un escalón de ``width × pendiente_transversal`` en un solo píxel; con el
    feather ese desnivel se reparte y el talud queda en un múltiplo pequeño de
    la pendiente que ya tenía el terreno. Se muestrea cada
    ``CROSS_SECTION_STEP`` px a lo largo de cada vía y solo se estadística
    donde el terreno original tenía pendiente real.
    """
    crudo = _imread(salida / "background" / "not_resized.png")
    aplanado = _imread(salida / "background" / "not_resized_with_flattened_roads.png")
    if crudo is None or aplanado is None:
        return None

    grouped = _road_classes(polylines, widths)
    resultado: dict[str, Any] = {}
    for tags, entries in grouped.items():
        width = widths[tags]
        half = int(width * 4) + 4
        antes_list: list[int] = []
        despues_list: list[int] = []
        muestras: list[dict[str, Any]] = []
        cortes_totales = 0
        for entry in entries:
            for ys, xs in _cortes_transversales(entry, half):
                cortes_totales += 1
                yi = np.clip(np.round(ys).astype(int), 0, crudo.shape[0] - 1)
                xi = np.clip(np.round(xs).astype(int), 0, crudo.shape[1] - 1)
                antes = crudo[yi, xi].astype(np.int64)
                g_antes = int(np.abs(np.diff(antes)).max())
                if g_antes < MIN_SLOPE_U16:
                    continue
                despues = aplanado[yi, xi].astype(np.int64)
                g_despues = int(np.abs(np.diff(despues)).max())
                antes_list.append(g_antes)
                despues_list.append(g_despues)
                if len(muestras) < 4 and g_antes > 3 * MIN_SLOPE_U16:
                    muestras.append(
                        {
                            "centro_yx": [int(yi[half]), int(xi[half])],
                            "original": antes.tolist(),
                            "aplanado": despues.tolist(),
                        }
                    )
        if not antes_list:
            continue
        antes_arr = np.array(antes_list, dtype=np.float64)
        despues_arr = np.array(despues_list, dtype=np.float64)
        ratios = despues_arr / antes_arr
        resultado[tags] = {
            "vias": len(entries),
            "width_schema": width,
            "cortes_totales": cortes_totales,
            "cortes_con_pendiente": int(antes_arr.size),
            "gradiente_transversal_max_u16": {
                "terreno_mediana": round(float(np.median(antes_arr)), 2),
                "aplanado_mediana": round(float(np.median(despues_arr)), 2),
                "terreno_p99": round(float(np.percentile(antes_arr, 99)), 2),
                "aplanado_p99": round(float(np.percentile(despues_arr, 99)), 2),
            },
            "ratio_talud_vs_terreno": {
                "mediana": round(float(np.median(ratios)), 3),
                "p95": round(float(np.percentile(ratios, 95)), 3),
                "max": round(float(ratios.max()), 3),
            },
            "escalon_de_corte_duro_evitado_u16": round(
                float(np.median(antes_arr) * width), 1
            ),
            "perfiles": muestras,
        }
    return resultado or None


#: Separación (px≈m) a la que se muestrea el eje para medir el bacheo que
#: heredarían los CVs de tráfico; es del orden del espaciado real de los CVs
#: con ``spline_density: 2`` en los segmentos OSM cortos.
SPLINE_SAMPLE_STEP = 5


def suavidad_splines(
    salida: Path, polylines: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Cuánto botan los CVs de tráfico antes y después del aplanado.

    Las splines muestrean el DEM, así que sin aplanado la Y de los CVs sigue el
    bacheo del terreno bajo la calzada. Se compara el salto de altura entre CVs
    consecutivos muestreando el mismo recorrido sobre el DEM crudo y sobre el
    aplanado.
    """
    crudo = _imread(salida / "background" / "not_resized.png")
    aplanado = _imread(salida / "background" / "not_resized_with_flattened_roads.png")
    if crudo is None or aplanado is None:
        return None

    saltos_crudo: list[float] = []
    saltos_aplanado: list[float] = []
    for entry in polylines:
        points = np.asarray(entry["points"], dtype=np.float64)
        if len(points) < 2:
            continue
        muestras = _densify(points, SPLINE_SAMPLE_STEP)
        if len(muestras) < 2:
            continue
        xi = np.clip(np.round(muestras[:, 0]).astype(int), 0, crudo.shape[1] - 1)
        yi = np.clip(np.round(muestras[:, 1]).astype(int), 0, crudo.shape[0] - 1)
        for dem, destino in ((crudo, saltos_crudo), (aplanado, saltos_aplanado)):
            z = dem[yi, xi].astype(np.float64) / UINT16_PER_METRE
            if z.size > 1:
                destino.extend(np.abs(np.diff(z)).tolist())
    if not saltos_crudo:
        return None
    a = np.array(saltos_crudo)
    b = np.array(saltos_aplanado)
    return {
        "muestras": int(a.size),
        "salto_entre_vertices_m": {
            "crudo_mediana": round(float(np.median(a)), 4),
            "aplanado_mediana": round(float(np.median(b)), 4),
            "crudo_p95": round(float(np.percentile(a, 95)), 4),
            "aplanado_p95": round(float(np.percentile(b, 95)), 4),
            "crudo_max": round(float(a.max()), 4),
            "aplanado_max": round(float(b.max()), 4),
        },
    }


# ------------------------------------------------------------------- splines


def _cv_values(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    tree = ET.parse(path)
    curvas: dict[str, list[tuple[float, float, float]]] = {}
    for curve in tree.getroot().iter("NurbsCurve"):
        name = curve.get("name") or ""
        puntos = []
        for cv in curve.findall("cv"):
            parts = [float(v) for v in (cv.get("c") or "0,0,0").split(",")]
            puntos.append(tuple(parts))
        curvas[name] = puntos
    return curvas


def splines(golden: Path, salida: Path) -> dict[str, Any] | None:
    """CVs que difieren del golden, separando XZ de Y (la altura del DEM)."""
    ruta_golden = golden / "map" / "splines.i3d"
    ruta_salida = salida / "map" / "splines.i3d"
    if not ruta_golden.is_file() or not ruta_salida.is_file():
        return None

    a, b = _cv_values(ruta_golden), _cv_values(ruta_salida)
    comunes = sorted(set(a) & set(b))
    total = solo_y = xz = 0
    deltas: list[float] = []
    for name in comunes:
        for pa, pb in zip(a[name], b[name]):
            total += 1
            if pa == pb:
                continue
            if pa[0] != pb[0] or pa[2] != pb[2]:
                xz += 1
            else:
                solo_y += 1
                deltas.append(abs(pa[1] - pb[1]))
    return {
        "curvas_comunes": len(comunes),
        "curvas_solo_golden": sorted(set(a) - set(b)),
        "curvas_solo_salida": sorted(set(b) - set(a)),
        "cvs_totales": total,
        "cvs_distintos_en_xz": xz,
        "cvs_distintos_solo_en_y": solo_y,
        "pct_cvs_distintos_solo_en_y": round(100.0 * solo_y / total, 3) if total else 0,
        "delta_y_max_m": round(max(deltas), 4) if deltas else 0.0,
        "delta_y_medio_m": round(float(np.mean(deltas)), 5) if deltas else 0.0,
    }


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=REPO_ROOT / "FS25_Valle_Bonito")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output/valle_bonito")
    parser.add_argument(
        "--dest", type=Path, default=REPO_ROOT / "output/analisis_flatten_roads"
    )
    parser.add_argument(
        "--texture-schema", type=Path, default=REPO_ROOT / "config/texture_schema.json"
    )
    args = parser.parse_args(argv)

    textures_json = args.output / "info_layers" / "textures.json"
    if not textures_json.is_file():
        print(f"[ERROR] falta {textures_json}: genera primero la salida", file=sys.stderr)
        return 2
    with open(textures_json, "r", encoding="utf-8") as f:
        info_layers = json.load(f)
    polylines = info_layers.get("roads_polylines", [])
    widths = road_widths_from_schema(load_texture_schema(args.texture_schema))

    args.dest.mkdir(parents=True, exist_ok=True)
    secciones = {
        "evidencia_golden": lambda: evidencia_golden(args.golden, polylines, widths),
        "comparacion_golden": lambda: comparacion_golden(args.golden, args.output),
        "calidad": lambda: calidad(args.output, polylines, widths),
        "suavidad_splines": lambda: suavidad_splines(args.output, polylines),
        "splines": lambda: splines(args.golden, args.output),
    }
    for nombre, fn in secciones.items():
        print(f"  · {nombre} …", flush=True)
        data = fn()
        if data is None:
            print("    (omitida: faltan artefactos)")
            continue
        destino = args.dest / f"{nombre}.json"
        with open(destino, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"    → {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
