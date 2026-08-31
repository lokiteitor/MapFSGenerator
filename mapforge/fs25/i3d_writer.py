"""Escritor del map.i3d y XML asociados (Fase 7 del plan; §D, §S1 y §S4).

Sobre el map.i3d del template desplegado (ElementTree, preservando la
estructura existente y el encoding ``iso-8859-1`` del template/artefacto):

- ``Terrain.heightScale`` := height_scale calculado por la Fase 1 (DEM);
  ``lodTextureSize`` := map_size (FACT §D: lodTextureSize = tamaño de mapa).
- ``DisplacementLayer.size`` := map_size × 8 (FACT-source §S1,
  ``I3d._update_parameters``); ``maxHeight`` := setting
  ``displacement_layer_max_height``; ``cellSize`` := 16384 / map_size (regla
  inferida del par template 2048→cellSize 8 / artefacto 8192→cellSize 2;
  producto size×cellSize constante = 131072; configurable vía
  ``i3d.displacement_layer_cell_size_base``).
- Luz ``sun``: ``lastShadowMapSplitBboxMin/Max`` = ``∓S/2,-128,∓S/2`` /
  ``S/2,148,S/2`` (formato exacto de 1.8: sin espacios).
- Integración de fields: reutiliza :func:`mapforge.fields.add_fields_to_i3d`
  sobre el mismo árbol parseado (idempotente: si ``gameplay/fields`` ya tiene
  hijos —p. ej. el generador corrió antes :class:`FieldsWriter`— no re-inserta).
- Background: un ``<File fileId … filename="../assets/background/…"/>`` por
  parte + ``<ReferenceNode name="background_terrain_part_0N" referenceId
  nodeId/>`` en la raíz de ``<Scene>`` (patrón del artefacto §I; los ids
  continúan tras el máximo existente en el documento). Nota documentada: en el
  golden los ReferenceNodes aparecen agrupados bajo un TransformGroup
  ``backgroundTerrain`` (edición del usuario/editor); el plan manda raíz de
  Scene.
- ``map.xml``: ``width``/``height`` := map_size (réplica de ``Config`` 1.8;
  se escribe con ``tree.write(encoding="utf-8", xml_declaration=True)``, la
  misma llamada que produce la declaración con comillas simples del golden).
- ``modDesc.xml``: nombre del mapa (config ``project.name``) en todos los
  ``<title>`` (raíz y ``maps/map``), como el artefacto ("Valle Bonito" en
  ambos). Desviación documentada: los ``<![CDATA[…]]>`` de la descripción se
  re-serializan como texto escapado (equivalente XML; ElementTree no preserva
  CDATA) y la declaración pierde ``standalone="no"``.

El namespace ``xsi`` se registra globalmente para que la re-serialización
conserve el prefijo del template (``xsi:noNamespaceSchemaLocation``).
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Sequence
from xml.etree import ElementTree as ET

from mapforge.fields.fields import (
    _append_pretty,
    _element_depth,
    add_fields_to_i3d,
    field_layer_border,
    load_fields_from_textures_json,
)

if TYPE_CHECKING:
    from mapforge.project import Project

logger = logging.getLogger("mapforge.fs25.i3d_writer")

# Preserva el prefijo xsi del template al re-serializar (registro global de ET).
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

#: Encoding del map.i3d (template y artefacto: iso-8859-1).
I3D_ENCODING = "iso-8859-1"

#: Declaración XML del map.i3d, con comillas dobles como el template/artefacto.
I3D_XML_DECLARATION = '<?xml version="1.0" encoding="iso-8859-1"?>\n'

#: Rutas de los nodos que edita el escritor (idénticas a i3d.py de 1.8).
TERRAIN_PATH = ".//Scene/TerrainTransformGroup"
DISPLACEMENT_LAYER_PATH = ".//Scene/TerrainTransformGroup/Layers/DisplacementLayer"
SUN_PATH = ".//Scene/Light[@name='sun']"


def _fmt_number(value: float | int) -> str:
    """``2.0 → "2"``, ``0.2 → "0.2"`` (formato de atributo sin ceros muertos)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _require(root: ET.Element, path: str, what: str) -> ET.Element:
    element = root.find(path)
    if element is None:
        raise ValueError(f"map.i3d sin {what} (path {path!r})")
    return element


# ------------------------------------------------------------ nodos del i3d


def update_terrain(
    root: ET.Element, map_size: int, height_scale: int | None
) -> dict[str, str]:
    """Actualiza el TerrainTransformGroup: ``heightScale`` y ``lodTextureSize``.

    ``height_scale`` es el calculado por la Fase 1 (``project.height_scale``);
    si es None (DEM no ejecutado) se conserva el del template y solo se
    actualiza ``lodTextureSize`` — el mismo contrato que 1.8, que solo toca
    heightScale cuando el DEM lo pidió (§S1).
    """
    terrain = _require(root, TERRAIN_PATH, "TerrainTransformGroup")
    data: dict[str, str] = {"lodTextureSize": str(map_size)}
    if height_scale is not None:
        data["heightScale"] = str(int(height_scale))
    else:
        logger.warning(
            "height_scale no disponible (¿DEM no ejecutado?); "
            "se conserva heightScale=%s del template",
            terrain.get("heightScale"),
        )
    for key, value in data.items():
        terrain.set(key, value)
    return data


def update_displacement_layer(
    root: ET.Element,
    map_size: int,
    max_height: float,
    size_factor: int = 8,
    cell_size_base: int = 16384,
) -> dict[str, str]:
    """Actualiza el DisplacementLayer: ``size``, ``maxHeight`` y ``cellSize``.

    - ``size = map_size × size_factor`` (FACT-source: ×8, ``str(int(...))``
      como 1.8).
    - ``maxHeight`` = setting (0.2 en el artefacto).
    - ``cellSize = cell_size_base / map_size`` (regla inferida: 16384/2048=8
      en el template, 16384/8192=2 en el artefacto; §S9 documenta que 1.8 no
      tiene mecanismo para cellSize — decisión técnica del plan, configurable).
    """
    layer = _require(root, DISPLACEMENT_LAYER_PATH, "DisplacementLayer")
    data = {
        "size": str(int(map_size * size_factor)),
        "maxHeight": _fmt_number(max_height),
        "cellSize": _fmt_number(cell_size_base / map_size),
    }
    for key, value in data.items():
        layer.set(key, value)
    return data


def update_sun(root: ET.Element, map_size: int) -> dict[str, str]:
    """Actualiza la luz ``sun``: bbox del último split del shadow map.

    Réplica exacta de ``I3d._update_parameters`` de 1.8 (formato sin
    espacios): ``Min = -S/2,-128,-S/2`` / ``Max = S/2,148,S/2``.
    """
    sun = _require(root, SUN_PATH, "Light 'sun'")
    distance = map_size // 2
    data = {
        "lastShadowMapSplitBboxMin": f"-{distance},-128,-{distance}",
        "lastShadowMapSplitBboxMax": f"{distance},148,{distance}",
    }
    for key, value in data.items():
        sun.set(key, value)
    return data


# ------------------------------------------------------------- background


def _max_int_attr(elements: Sequence[ET.Element] | Any, attr: str) -> int:
    """Máximo valor entero del atributo ``attr`` (0 si no hay ninguno)."""
    best = 0
    for element in elements:
        raw = element.get(attr)
        if raw is None:
            continue
        try:
            best = max(best, int(raw))
        except ValueError:
            continue
    return best


def add_background_to_i3d(
    root: ET.Element, parts: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Inserta Files + ReferenceNodes de las partes del background.

    Patrón del artefacto (§I): un ``<File fileId
    filename="../assets/background/background_terrain_part_0N.i3d"/>`` en
    ``<Files>`` por parte y un ``<ReferenceNode
    name="background_terrain_part_0N" referenceId={fileId} nodeId=…/>`` en la
    raíz de ``<Scene>``, sin translation propia (la translation Y va dentro
    del i3d de cada parte, Fase 8).

    ``parts``: dicts con ``name`` y ``i3d`` (ruta relativa a la raíz del mod,
    p. ej. ``assets/background/background_terrain_part_01.i3d`` — el formato
    que devuelve :meth:`BackgroundExporter.run`). Los fileId continúan tras el
    máximo de ``<Files>`` y los nodeId tras el máximo del documento (evita
    colisiones con template, fields y cualquier edición previa). Idempotente:
    una parte cuyo ReferenceNode ya existe en la escena se salta.

    Returns:
        Lista de ``{"name", "file_id", "node_id", "filename"}`` insertados.
    """
    if not parts:
        return []

    files_node = _require(root, "Files", "sección <Files>")
    scene_node = _require(root, "Scene", "sección <Scene>")

    existing_refs = {
        node.get("name")
        for node in scene_node.iter("ReferenceNode")
        if node.get("name")
    }

    next_file_id = _max_int_attr(root.iter("File"), "fileId") + 1
    next_node_id = _max_int_attr(root.iter(), "nodeId") + 1

    new_files: list[ET.Element] = []
    new_nodes: list[ET.Element] = []
    added: list[dict[str, Any]] = []

    for part in parts:
        name = part.get("name") or PurePosixPath(str(part["i3d"])).stem
        if name in existing_refs:
            logger.info("background: ReferenceNode %s ya existe; se salta", name)
            continue
        # Ruta relativa desde map/ (donde vive map.i3d) a la raíz del mod.
        filename = "../" + PurePosixPath(str(part["i3d"])).as_posix()

        file_element = ET.Element("File")
        file_element.set("fileId", str(next_file_id))
        file_element.set("filename", filename)
        new_files.append(file_element)

        ref_element = ET.Element("ReferenceNode")
        ref_element.set("name", name)
        ref_element.set("referenceId", str(next_file_id))
        ref_element.set("nodeId", str(next_node_id))
        new_nodes.append(ref_element)

        added.append(
            {
                "name": name,
                "file_id": next_file_id,
                "node_id": next_node_id,
                "filename": filename,
            }
        )
        next_file_id += 1
        next_node_id += 1

    _append_pretty(files_node, new_files, _element_depth(root, files_node))
    _append_pretty(scene_node, new_nodes, _element_depth(root, scene_node))

    logger.info("background: %s parte(s) integradas en map.i3d", len(added))
    return added


# ---------------------------------------------------------- map.xml/modDesc


def update_map_xml(path: str | Path, map_size: int) -> dict[str, str]:
    """``map.xml``: ``width``/``height`` := map_size (réplica de ``Config``
    1.8, incluida la escritura ``encoding="utf-8", xml_declaration=True`` que
    produce la declaración con comillas simples del golden)."""
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    map_element = root if root.tag == "map" else root.find(".//map")
    if map_element is None:
        raise ValueError(f"{path}: sin elemento <map>")
    data = {"width": str(map_size), "height": str(map_size)}
    for key, value in data.items():
        map_element.set(key, value)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return data


def update_mod_desc(path: str | Path, map_name: str) -> int:
    """``modDesc.xml``: pone ``map_name`` en todos los ``<title>`` (raíz del
    modDesc y ``maps/map``), como el artefacto golden ("Valle Bonito" en
    ambos). Devuelve el número de elementos de idioma actualizados."""
    path = Path(path)
    tree = ET.parse(path)
    updated = 0
    for title in tree.getroot().iter("title"):
        for lang in title:
            lang.text = map_name
            updated += 1
    tree.write(path, encoding="utf-8", xml_declaration=True)
    if not updated:
        logger.warning("%s: ningún <title> con idiomas que actualizar", path)
    return updated


# -------------------------------------------------------------- escritura


def write_map_i3d(tree: ET.ElementTree, path: str | Path) -> None:
    """Escribe el map.i3d preservando el encoding iso-8859-1 del template.

    La declaración se emite a mano con comillas dobles (como template y
    artefacto); los caracteres fuera de latin-1 se escapan como referencias
    numéricas (``xmlcharrefreplace``)."""
    payload = ET.tostring(tree.getroot(), encoding="unicode")
    data = (I3D_XML_DECLARATION + payload).encode(
        I3D_ENCODING, errors="xmlcharrefreplace"
    )
    Path(path).write_bytes(data)


# ----------------------------------------------------------------- writer


class I3dWriter:
    """Aplica todas las ediciones del map.i3d/map.xml/modDesc.xml del output.

    Un único ciclo parse→editar→escribir sobre ``map/map.i3d`` (terrain,
    DisplacementLayer, sun, fields, background) + ``map/map.xml`` +
    ``modDesc.xml``. Pensado para correr tras el DEM (necesita
    ``project.height_scale``) y, en el pipeline completo, tras texturas
    (``info_layers/textures.json`` para los fields) y el exportador de
    background (partes en ``assets/background/``).
    """

    def __init__(self, project: "Project") -> None:
        self.project = project
        self.logger = logging.getLogger("mapforge.fs25.i3d_writer")
        self.info: dict[str, Any] = {}

    # ------------------------------------------------------------ helpers

    def _integrate_fields(self, root: ET.Element) -> dict[str, Any]:
        """Inserta los fields de textures.json si aún no están en el i3d."""
        paths = self.project.paths
        if not paths.textures_json.exists():
            self.logger.info(
                "sin %s; fields no integrados en este paso", paths.textures_json
            )
            return {"status": "skipped", "reason": "textures.json no existe"}

        fields_node = root.find(".//TransformGroup[@name='fields']")
        if fields_node is not None and len(fields_node):
            self.logger.info(
                "gameplay/fields ya tiene %s hijo(s); no se re-insertan fields",
                len(fields_node),
            )
            return {"status": "skipped", "reason": "fields ya presentes"}

        fields = load_fields_from_textures_json(paths.textures_json)
        if not fields:
            return {"status": "skipped", "reason": "textures.json sin fields"}

        border = 0
        if paths.texture_schema.exists():
            border = field_layer_border(paths.texture_schema)

        stats = add_fields_to_i3d(
            root,
            fields,
            map_size=self.project.map.size,
            map_rotated_size=self.project.map.rotated_size,
            rotation=self.project.map.rotation,
            border=border,
        )
        return {"status": "added", "added": stats.added, "skipped": stats.skipped}

    def _discover_background_parts(self) -> list[dict[str, Any]]:
        """Partes del background en ``assets/background/`` del output."""
        assets_dir = self.project.paths.output_dir / "assets" / "background"
        if not assets_dir.is_dir():
            return []
        parts = []
        for i3d_path in sorted(assets_dir.glob("background_terrain_part_*.i3d")):
            parts.append(
                {
                    "name": i3d_path.stem,
                    "i3d": str(i3d_path.relative_to(self.project.paths.output_dir)),
                }
            )
        return parts

    # --------------------------------------------------------------- run

    def run(
        self, background_parts: Sequence[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Aplica todas las ediciones y escribe los tres XML.

        Arguments:
            background_parts: partes del background (dicts ``name``/``i3d``
                del :meth:`BackgroundExporter.run`); si es None se buscan en
                ``assets/background/`` del output.

        Returns:
            dict con lo aplicado (para logs y ``generation_info``).
        """
        project = self.project
        paths = project.paths
        map_size = project.map.size
        i3d_settings = project.settings.i3d

        tree = ET.parse(paths.map_i3d)
        root = tree.getroot()

        info: dict[str, Any] = {
            "terrain": update_terrain(root, map_size, project.height_scale),
            "displacement_layer": update_displacement_layer(
                root,
                map_size,
                max_height=i3d_settings.displacement_layer_max_height,
                size_factor=i3d_settings.displacement_layer_size_factor,
                cell_size_base=i3d_settings.displacement_layer_cell_size_base,
            ),
            "sun": update_sun(root, map_size),
        }

        info["fields"] = self._integrate_fields(root)

        if background_parts is None:
            background_parts = self._discover_background_parts()
        info["background"] = add_background_to_i3d(root, background_parts)

        write_map_i3d(tree, paths.map_i3d)
        self.logger.info("map.i3d actualizado: %s", paths.map_i3d)

        info["map_xml"] = update_map_xml(paths.map_xml, map_size)
        self.logger.info("map.xml: width/height = %s", map_size)

        info["mod_desc_titles"] = update_mod_desc(paths.mod_desc, project.name)
        self.logger.info("modDesc.xml: título '%s'", project.name)

        self.info = info
        return info
