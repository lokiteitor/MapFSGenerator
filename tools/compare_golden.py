#!/usr/bin/env python
"""Harness de validación: compara un directorio de salida contra el golden.

Compara un mapa generado por MapForge con ``FS25_Valle_Bonito/`` (el artefacto
de Maps4FS 3.1.2 verificado en el informe forense):

- **PNG**: dimensiones, dtype/canales, % de píxeles distintos y delta máximo.
- **XML / i3d**: diff de atributos por elemento (emparejando por tag + atributo
  ``name`` cuando es único, por orden en caso contrario).
- **Resto de ficheros**: existencia y tamaño.

Uso::

    ./venv/bin/python tools/compare_golden.py output/valle_bonito \\
        [--golden FS25_Valle_Bonito] [--json output/reporte.json] \\
        [--max-diffs 20] [--show-missing]

Solo se comparan en detalle los ficheros presentes en AMBOS directorios; los
que faltan en la salida se cuentan (con ``--show-missing`` se listan). Sale
con código 0 si todo lo comparado es idéntico, 1 si hay diferencias y 2 ante
errores de uso.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

XML_SUFFIXES = {".xml", ".i3d"}
PNG_SUFFIXES = {".png"}

#: Ficheros del golden que son entradas/metadatos de Maps4FS, no salida a replicar.
DEFAULT_IGNORES = {
    "valle_bonito.png",
    "custom_osm.osm",
    "generation_settings.json",
    "generation_info.json",
    "generation_logs.json",
    "main_settings.json",
    "performance_report.json",
    "render_pda.py",
}


# --------------------------------------------------------------------- PNG


def compare_png(golden: Path, ours: Path) -> dict[str, Any]:
    """Compara dos PNG: dims, dtype, % píxeles distintos, delta máximo."""
    img_g = cv2.imread(str(golden), cv2.IMREAD_UNCHANGED)
    img_o = cv2.imread(str(ours), cv2.IMREAD_UNCHANGED)
    if img_g is None or img_o is None:
        return {
            "status": "error",
            "detail": f"no se pudo leer PNG ({'golden' if img_g is None else 'salida'})",
        }

    result: dict[str, Any] = {
        "golden_shape": list(img_g.shape),
        "output_shape": list(img_o.shape),
        "golden_dtype": str(img_g.dtype),
        "output_dtype": str(img_o.dtype),
    }
    if img_g.shape != img_o.shape or img_g.dtype != img_o.dtype:
        result["status"] = "differs"
        result["detail"] = "shape/dtype distintos"
        return result

    diff = img_g.astype(np.int64) - img_o.astype(np.int64)
    if diff.ndim == 3:
        per_pixel = np.abs(diff).max(axis=2)
    else:
        per_pixel = np.abs(diff)
    n_diff = int(np.count_nonzero(per_pixel))
    total = int(per_pixel.size)
    result["pixels_diff"] = n_diff
    result["pixels_total"] = total
    result["pct_diff"] = round(100.0 * n_diff / total, 4) if total else 0.0
    result["max_delta"] = int(per_pixel.max()) if total else 0
    result["status"] = "identical" if n_diff == 0 else "differs"
    return result


# --------------------------------------------------------------------- XML


def _element_key_path(elem: ET.Element, parent_path: str, index: int) -> str:
    name = elem.get("name")
    if name:
        return f"{parent_path}/{elem.tag}[name={name}]"
    return f"{parent_path}/{elem.tag}[{index}]"


def _pair_children(
    golden: ET.Element, ours: ET.Element
) -> tuple[list[tuple[ET.Element, ET.Element, str, int]], list[str], list[str]]:
    """Empareja hijos por (tag, name) cuando el name es único; si no, por orden.

    Returns:
        (parejas, solo_en_golden, solo_en_salida) — los "solo en" como claves
        legibles.
    """
    pairs: list[tuple[ET.Element, ET.Element, str, int]] = []
    only_golden: list[str] = []
    only_ours: list[str] = []

    tags = sorted(
        {child.tag for child in golden} | {child.tag for child in ours},
        key=lambda t: t,
    )
    for tag in tags:
        g_children = [c for c in golden if c.tag == tag]
        o_children = [c for c in ours if c.tag == tag]

        g_names = [c.get("name") for c in g_children]
        o_names = [c.get("name") for c in o_children]
        use_names = (
            all(g_names)
            and all(o_names)
            and len(set(g_names)) == len(g_names)
            and len(set(o_names)) == len(o_names)
        )
        if use_names:
            o_by_name = {c.get("name"): c for c in o_children}
            for i, g_child in enumerate(g_children):
                name = g_child.get("name")
                if name in o_by_name:
                    pairs.append((g_child, o_by_name.pop(name), tag, i))
                else:
                    only_golden.append(f"{tag}[name={name}]")
            only_ours.extend(f"{tag}[name={n}]" for n in o_by_name)
        else:
            for i, (g_child, o_child) in enumerate(zip(g_children, o_children)):
                pairs.append((g_child, o_child, tag, i))
            if len(g_children) > len(o_children):
                only_golden.extend(
                    f"{tag}[{i}]" for i in range(len(o_children), len(g_children))
                )
            elif len(o_children) > len(g_children):
                only_ours.extend(
                    f"{tag}[{i}]" for i in range(len(g_children), len(o_children))
                )
    return pairs, only_golden, only_ours


def _diff_elements(
    golden: ET.Element,
    ours: ET.Element,
    path: str,
    diffs: list[dict[str, Any]],
    max_diffs: int,
) -> None:
    if len(diffs) >= max_diffs:
        return

    keys = sorted(set(golden.attrib) | set(ours.attrib))
    for key in keys:
        gv = golden.get(key)
        ov = ours.get(key)
        if gv != ov:
            diffs.append(
                {"path": path, "attribute": key, "golden": gv, "output": ov}
            )
            if len(diffs) >= max_diffs:
                return

    g_text = (golden.text or "").strip()
    o_text = (ours.text or "").strip()
    if g_text != o_text:
        diffs.append(
            {
                "path": path,
                "attribute": "#text",
                "golden": g_text[:120] or None,
                "output": o_text[:120] or None,
            }
        )
        if len(diffs) >= max_diffs:
            return

    pairs, only_golden, only_ours = _pair_children(golden, ours)
    for key in only_golden:
        diffs.append({"path": f"{path}/{key}", "attribute": "#element", "golden": "presente", "output": None})
        if len(diffs) >= max_diffs:
            return
    for key in only_ours:
        diffs.append({"path": f"{path}/{key}", "attribute": "#element", "golden": None, "output": "presente"})
        if len(diffs) >= max_diffs:
            return
    for g_child, o_child, tag, index in pairs:
        _diff_elements(
            g_child,
            o_child,
            _element_key_path(g_child, path, index),
            diffs,
            max_diffs,
        )
        if len(diffs) >= max_diffs:
            return


def compare_xml(golden: Path, ours: Path, max_diffs: int) -> dict[str, Any]:
    """Diff de atributos por elemento entre dos XML/i3d."""
    try:
        g_root = ET.parse(golden).getroot()
        o_root = ET.parse(ours).getroot()
    except ET.ParseError as exc:
        return {"status": "error", "detail": f"XML no parseable: {exc}"}

    diffs: list[dict[str, Any]] = []
    _diff_elements(g_root, o_root, f"/{g_root.tag}", diffs, max_diffs)
    truncated = len(diffs) >= max_diffs
    return {
        "status": "identical" if not diffs else "differs",
        "n_diffs": len(diffs),
        "truncated": truncated,
        "diffs": diffs,
    }


# ------------------------------------------------------------------- otros


def compare_other(golden: Path, ours: Path) -> dict[str, Any]:
    """Comparación básica: existencia (ya garantizada) y tamaño."""
    g_size = golden.stat().st_size
    o_size = ours.stat().st_size
    delta_pct = round(100.0 * abs(o_size - g_size) / g_size, 2) if g_size else 0.0
    return {
        "status": "identical" if g_size == o_size else "differs",
        "golden_size": g_size,
        "output_size": o_size,
        "size_delta_pct": delta_pct,
        "note": "comparación solo de tamaño (binario)",
    }


# ----------------------------------------------------------------- driver


def collect_files(root: Path) -> set[str]:
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
    }


def run_comparison(
    output_dir: Path,
    golden_dir: Path,
    max_diffs: int,
    ignore_meta: bool = True,
) -> dict[str, Any]:
    golden_files = collect_files(golden_dir)
    output_files = collect_files(output_dir)
    if ignore_meta:
        golden_files = {f for f in golden_files if Path(f).name not in DEFAULT_IGNORES}

    common = sorted(golden_files & output_files)
    missing = sorted(golden_files - output_files)
    extra = sorted(output_files - golden_files)

    files: dict[str, Any] = {}
    status_counter: Counter[str] = Counter()
    for rel in common:
        golden = golden_dir / rel
        ours = output_dir / rel
        suffix = golden.suffix.lower()
        if suffix in PNG_SUFFIXES:
            result = compare_png(golden, ours)
            result["kind"] = "png"
        elif suffix in XML_SUFFIXES:
            result = compare_xml(golden, ours, max_diffs)
            result["kind"] = "xml"
        else:
            result = compare_other(golden, ours)
            result["kind"] = "binary"
        files[rel] = result
        status_counter[result["status"]] += 1

    return {
        "output_dir": str(output_dir),
        "golden_dir": str(golden_dir),
        "files": files,
        "missing_in_output": missing,
        "extra_in_output": extra,
        "summary": {
            "compared": len(common),
            "identical": status_counter.get("identical", 0),
            "differs": status_counter.get("differs", 0),
            "errors": status_counter.get("error", 0),
            "missing_in_output": len(missing),
            "extra_in_output": len(extra),
        },
    }


def print_report(report: dict[str, Any], show_missing: bool) -> None:
    print(f"Comparando salida: {report['output_dir']}")
    print(f"contra golden:     {report['golden_dir']}")
    print()

    for rel, result in report["files"].items():
        status = result["status"]
        kind = result["kind"]
        if status == "identical":
            print(f"  [OK]     {rel}")
            continue
        if status == "error":
            print(f"  [ERROR]  {rel}: {result.get('detail')}")
            continue
        if kind == "png":
            if "pct_diff" in result:
                print(
                    f"  [DIFF]   {rel}: {result['pct_diff']}% pixeles distintos "
                    f"({result['pixels_diff']}/{result['pixels_total']}), "
                    f"max delta {result['max_delta']}"
                )
            else:
                print(
                    f"  [DIFF]   {rel}: shape/dtype distintos "
                    f"golden={result['golden_shape']}/{result['golden_dtype']} "
                    f"salida={result['output_shape']}/{result['output_dtype']}"
                )
        elif kind == "xml":
            extra_note = " (truncado)" if result.get("truncated") else ""
            print(f"  [DIFF]   {rel}: {result['n_diffs']} diferencias{extra_note}")
            for diff in result["diffs"][:10]:
                print(
                    f"           {diff['path']} @{diff['attribute']}: "
                    f"golden={diff['golden']!r} salida={diff['output']!r}"
                )
            if result["n_diffs"] > 10:
                print(f"           ... y {result['n_diffs'] - 10} más")
        else:
            print(
                f"  [DIFF]   {rel}: tamaño golden={result['golden_size']} "
                f"salida={result['output_size']} ({result['size_delta_pct']}%)"
            )

    summary = report["summary"]
    print()
    print(
        f"Resumen: {summary['compared']} comparados, {summary['identical']} identicos, "
        f"{summary['differs']} con diferencias, {summary['errors']} errores, "
        f"{summary['missing_in_output']} faltan en la salida, "
        f"{summary['extra_in_output']} extra en la salida"
    )
    if show_missing and report["missing_in_output"]:
        print("\nFaltan en la salida:")
        for rel in report["missing_in_output"]:
            print(f"  - {rel}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compara un mapa generado por MapForge contra el golden FS25_Valle_Bonito."
    )
    parser.add_argument("output_dir", help="Directorio de salida generado por mapforge")
    parser.add_argument(
        "--golden",
        default=str(Path(__file__).resolve().parent.parent / "FS25_Valle_Bonito"),
        help="Directorio golden (default: FS25_Valle_Bonito del repo)",
    )
    parser.add_argument(
        "--json",
        metavar="RUTA",
        help="Escribe además el reporte completo como JSON en esta ruta",
    )
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=200,
        help="Máximo de diferencias XML recogidas por fichero (default 200)",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Lista los ficheros del golden que faltan en la salida",
    )
    parser.add_argument(
        "--no-ignore-meta",
        action="store_true",
        help="No excluir los metadatos de Maps4FS (generation_info.json, etc.)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    golden_dir = Path(args.golden)
    if not output_dir.is_dir():
        parser.error(f"directorio de salida no existe: {output_dir}")
    if not golden_dir.is_dir():
        parser.error(f"directorio golden no existe: {golden_dir}")

    report = run_comparison(
        output_dir,
        golden_dir,
        max_diffs=args.max_diffs,
        ignore_meta=not args.no_ignore_meta,
    )
    print_report(report, show_missing=args.show_missing)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReporte JSON: {json_path}")

    summary = report["summary"]
    return 0 if summary["differs"] == 0 and summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
