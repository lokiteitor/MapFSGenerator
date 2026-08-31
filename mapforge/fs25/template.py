"""Despliegue del template de mapa FS25.

El template (``fs25-map-template.zip`` de Maps4FS o un directorio ya
extraído) contiene la estructura base del mod: ``map/`` (map.i3d, map.xml,
config/*.xml, splines.i3d esqueleto), ``modDesc.xml``, iconos... El
despliegue copia todo eso al directorio de salida del proyecto; el resto de
componentes escriben encima.

FACT (informe §S1): ``map/data/`` del template está vacío — todos los rasters
se generan desde cero.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger("mapforge.fs25.template")

#: Entradas mínimas que debe contener un template FS25 válido.
REQUIRED_ENTRIES = ("map/map.i3d", "map/map.xml", "modDesc.xml")


def deploy_template(template: str | Path, output_dir: str | Path) -> Path:
    """Despliega el template (zip o directorio) dentro de ``output_dir``.

    - Si ``template`` es un ``.zip`` se extrae directamente en ``output_dir``.
    - Si es un directorio se copia su contenido (se admite tanto la raíz del
      template como un directorio que lo contenga ya extraído).

    El despliegue garantiza que existan ``map/data/`` y los subdirectorios de
    trabajo que el template trae vacíos (zip no preserva dirs vacíos siempre).

    Returns:
        La ruta de ``output_dir`` como :class:`Path`.

    Raises:
        FileNotFoundError: si el template no existe.
        ValueError: si el template no contiene la estructura mínima FS25.
    """
    template = Path(template)
    output_dir = Path(output_dir)
    if not template.exists():
        raise FileNotFoundError(f"template no encontrado: {template}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if template.is_file() and zipfile.is_zipfile(template):
        _deploy_zip(template, output_dir)
    elif template.is_dir():
        _deploy_dir(template, output_dir)
    else:
        raise ValueError(f"template no es ni zip ni directorio: {template}")

    _validate_deployment(output_dir)

    # Directorios de trabajo que las fases posteriores esperan encontrar.
    (output_dir / "map" / "data").mkdir(parents=True, exist_ok=True)

    logger.info("template desplegado en %s", output_dir)
    return output_dir


def _deploy_zip(template_zip: Path, output_dir: Path) -> None:
    """Extrae el zip del template en ``output_dir`` (rechaza rutas fuera)."""
    with zipfile.ZipFile(template_zip) as zf:
        for info in zf.infolist():
            target = output_dir / info.filename
            # Protección contra zip-slip: todo debe quedar bajo output_dir.
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"entrada de zip fuera del destino: {info.filename}")
        zf.extractall(output_dir)
        logger.debug(
            "extraído zip %s (%d entradas)", template_zip.name, len(zf.namelist())
        )


def _deploy_dir(template_dir: Path, output_dir: Path) -> None:
    """Copia el contenido del directorio del template en ``output_dir``."""
    root = template_dir
    if not (root / "map" / "map.i3d").exists():
        # Admite un directorio contenedor con un único subdirectorio-template.
        candidates = [
            child
            for child in root.iterdir()
            if child.is_dir() and (child / "map" / "map.i3d").exists()
        ]
        if len(candidates) == 1:
            root = candidates[0]
    shutil.copytree(root, output_dir, dirs_exist_ok=True)
    logger.debug("copiado directorio template %s", root)


def _validate_deployment(output_dir: Path) -> None:
    missing = [entry for entry in REQUIRED_ENTRIES if not (output_dir / entry).exists()]
    if missing:
        raise ValueError(
            "despliegue de template incompleto, faltan: " + ", ".join(missing)
        )
