"""Orquestador del pipeline de generación de MapForge (Fase 9 del plan).

Ejecuta los componentes en el orden del plan::

    template → DEM → OSM/texturas → grle_layers → farmlands → fields
    → splines → i3d writer → background

y escribe ``generation_info.json`` en el directorio de salida con la
telemetría de cada componente (estadísticas devueltas por cada fase, tiempos
por etapa, settings efectivos y etapas saltadas).

Notas de integración:

- El escritor de map.i3d (Fase 7) corre ANTES del background (orden del
  plan); las partes del background se integran en el map.i3d al final de la
  etapa de background con :func:`mapforge.fs25.i3d_writer.add_background_to_i3d`
  sobre el mismo fichero (la operación es idempotente por nombre de
  ReferenceNode).
- Los fields se insertan en el map.i3d por :class:`FieldsWriter` (Fase 4); el
  :class:`I3dWriter` detecta que ``gameplay/fields`` ya tiene hijos y no los
  re-inserta.
- El hook de resta de agua del DEM (``water_mask``) queda desactivado:
  ``generate_water`` está fuera del alcance del proyecto (el golden se generó
  con ``generate_water: false``; su ``FULL.png`` sí lleva la resta de agua de
  Maps4FS 3.x — diferencia documentada en ``docs/validacion_golden.md``).
- ``skip_stages`` permite saltar etapas por nombre (con warning y constancia
  en ``generation_info.json``); pensado para excluir componentes que no
  superaron su validación de fase. Las etapas posteriores usan sus fallbacks
  (p. ej. splines relee ``background/not_resized.png``) o fallan con un error
  claro si dependen de la etapa saltada.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Collection
from xml.etree import ElementTree as ET

import numpy as np

import cv2

from mapforge import __version__
from mapforge.farmlands.farmlands import FarmlandsWriter
from mapforge.fields.fields import FieldsWriter
from mapforge.fs25.grle_layers import create_empty_grle_layers
from mapforge.fs25.plants import PlantsWriter
from mapforge.fs25.i3d_writer import I3dWriter, add_background_to_i3d, write_map_i3d
from mapforge.fs25.template import deploy_template
from mapforge.osm.parser import OsmData, parse_osm
from mapforge.project import Project
from mapforge.splines.traffic import TrafficSplinesWriter
from mapforge.terrain.dem import DemPipeline
from mapforge.textures.engine import TextureEngine

logger = logging.getLogger("mapforge.generator")

#: Orden canónico de las etapas del pipeline (nombres aceptados en skip_stages).
#:
#: ``textures`` va antes que ``dem`` porque el aplanado de carreteras/farmyards
#: consume ``info_layers/textures.json`` (``roads_polylines`` y ``farmyards``),
#: igual que hace Maps4FS 3.x: en sus logs el componente Texture termina antes
#: de que el Background aplane el DEM. La etapa de texturas solo depende del
#: OSM y del proyecto, así que el intercambio no crea ninguna dependencia nueva.
STAGE_ORDER: tuple[str, ...] = (
    "template",
    "textures",
    "dem",
    "grle_layers",
    "farmlands",
    "plants",
    "fields",
    "splines",
    "i3d",
    "background",
)


def _json_safe(value: Any) -> Any:
    """Convierte tipos numpy/Path a nativos para ``json.dump`` (default=)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class Generator:
    """Ejecuta el pipeline de generación completo sobre un :class:`Project`.

    Arguments:
        project: proyecto MapForge (config cargada).
        skip_stages: nombres de etapas de :data:`STAGE_ORDER` a saltar (se
            registran con warning y en ``generation_info.json``).
    """

    def __init__(
        self, project: Project, skip_stages: Collection[str] = ()
    ) -> None:
        self.project = project
        self.logger = logger

        unknown = sorted(set(skip_stages) - set(STAGE_ORDER))
        if unknown:
            raise ValueError(
                f"skip_stages desconocidas: {', '.join(unknown)} "
                f"(válidas: {', '.join(STAGE_ORDER)})"
            )
        self.skip_stages = set(skip_stages)

        # Estado compartido entre etapas.
        self.dem_pipeline: DemPipeline | None = None
        self.osm_data: OsmData | None = None

        # Telemetría acumulada.
        self.components: dict[str, Any] = {}
        self.timings: dict[str, float] = {}
        self.skipped: list[str] = []

    # ------------------------------------------------------------------ run

    def run(self) -> dict[str, Any]:
        """Ejecuta el pipeline completo y escribe ``generation_info.json``.

        Returns:
            El contenido de ``generation_info.json`` como dict.
        """
        project = self.project
        self.logger.info(project.describe())

        problems = project.validate_inputs()
        if problems:
            raise FileNotFoundError(
                "entradas del proyecto no disponibles: " + "; ".join(problems)
            )

        stages: tuple[tuple[str, Callable[[], Any]], ...] = (
            ("template", self._stage_template),
            ("textures", self._stage_textures),
            ("dem", self._stage_dem),
            ("grle_layers", self._stage_grle_layers),
            ("farmlands", self._stage_farmlands),
            ("plants", self._stage_plants),
            ("fields", self._stage_fields),
            ("splines", self._stage_splines),
            ("i3d", self._stage_i3d),
            ("background", self._stage_background),
        )

        total_start = time.perf_counter()
        for name, stage_fn in stages:
            if name in self.skip_stages:
                self.logger.warning(
                    "etapa '%s' SALTADA por skip_stages (no superó su "
                    "validación de fase o fue excluida explícitamente)",
                    name,
                )
                self.components[name] = {
                    "status": "skipped",
                    "reason": "skip_stages",
                }
                self.skipped.append(name)
                continue

            self.logger.info("=== etapa %s ===", name)
            stage_start = time.perf_counter()
            result = stage_fn()
            elapsed = round(time.perf_counter() - stage_start, 3)
            self.timings[name] = elapsed
            self.components[name] = result
            self.logger.info("etapa %s completada en %.3f s", name, elapsed)

        self.timings["total"] = round(time.perf_counter() - total_start, 3)
        payload = self._write_generation_info()
        self.logger.info(
            "pipeline completado en %.1f s → %s",
            self.timings["total"],
            project.paths.output_dir,
        )
        return payload

    # --------------------------------------------------------------- etapas

    def _stage_template(self) -> dict[str, Any]:
        """Despliega el template FS25 en el directorio de salida."""
        paths = self.project.paths
        deploy_template(paths.template, paths.output_dir)
        return {
            "template": str(paths.template),
            "output_dir": str(paths.output_dir),
        }

    def _stage_dem(self) -> dict[str, Any]:
        """Pipeline DEM (Fase 1): dem.png + DEM intermedios del background."""
        pipeline = DemPipeline(self.project)
        pipeline.run(water_mask=None)
        self.dem_pipeline = pipeline
        return {
            "mode": "custom_dem" if pipeline.custom_dem else "raw",
            "height_scale": pipeline.height_scale,
            "stages": pipeline.info,
            "flatten": pipeline.flatten_stats,
        }

    def _stage_textures(self) -> dict[str, Any]:
        """Parseo OSM + motor de texturas (Fases 2-3)."""
        project = self.project
        osm_data = parse_osm(project.paths.osm)
        self.osm_data = osm_data
        self.logger.info(
            "OSM parseado: %d nodos, %d ways, %d relations, %d features",
            len(osm_data.nodes),
            len(osm_data.ways),
            len(osm_data.relations),
            len(osm_data.features),
        )

        engine = TextureEngine(project, osm_data)
        engine.run()

        info_layer_counts: dict[str, int] = {}
        if project.paths.textures_json.is_file():
            with open(project.paths.textures_json, "r", encoding="utf-8") as f:
                info_layer_counts = {
                    key: len(value) for key, value in json.load(f).items()
                }

        return {
            "osm": {
                "nodes": len(osm_data.nodes),
                "ways": len(osm_data.ways),
                "relations": len(osm_data.relations),
                "features": len(osm_data.features),
            },
            "layers": len(engine.layers),
            "info_layers": info_layer_counts,
        }

    def _stage_grle_layers(self) -> dict[str, Any]:
        """infoLayers/densityMaps en cero según el grle_schema (Fase 5)."""
        created = create_empty_grle_layers(self.project)
        return {"created": len(created), "layers": created}

    def _stage_farmlands(self) -> dict[str, Any]:
        """infoLayer_farmlands.png + farmlands.xml (Fase 5)."""
        return FarmlandsWriter(self.project).run()

    def _stage_plants(self) -> dict[str, Any]:
        """Hierba base en densityMap_fruits.png (Fase 5)."""
        return PlantsWriter(self.project).run()

    def _stage_fields(self) -> dict[str, Any]:
        """Fields jugables en el map.i3d (Fase 4)."""
        stats = FieldsWriter(self.project).run()
        return {"added": stats.added, "skipped": stats.skipped}

    def _stage_splines(self) -> dict[str, Any]:
        """Splines de tráfico en map/splines.i3d (Fase 6)."""
        dem_not_resized = (
            self.dem_pipeline.dem_not_resized if self.dem_pipeline else None
        )
        writer = TrafficSplinesWriter(
            self.project,
            dem_not_resized=dem_not_resized,
            height_scale=self.project.height_scale,
        )
        written = writer.run()
        return {"written_curves": written}

    def _stage_i3d(self) -> dict[str, Any]:
        """Escritor de map.i3d/map.xml/modDesc.xml (Fase 7).

        Las partes del background aún no existen (el background corre después,
        orden del plan): se pasa una lista vacía y la integración de las
        partes la hace la etapa de background sobre el mismo fichero.
        """
        return I3dWriter(self.project).run(background_parts=[])

    def _stage_background(self) -> dict[str, Any]:
        """Background procedural (Fase 8) + integración en map.i3d."""
        project = self.project
        if not project.settings.background.generate_background:
            self.logger.info("generate_background=false: background omitido")
            return {"status": "skipped", "reason": "generate_background=false"}

        # Import perezoso: trimesh solo se paga si hay background.
        from mapforge.background.exporter import BackgroundExporter
        from mapforge.background.mesh import build_background_mesh

        dem_full = self._resolve_dem_full()
        mesh = build_background_mesh(project, dem_full)
        export = BackgroundExporter(project).run(mesh, dem_full)

        # Integración de las partes en map.i3d (Files + ReferenceNodes).
        tree = ET.parse(project.paths.map_i3d)
        references = add_background_to_i3d(tree.getroot(), export["parts"])
        write_map_i3d(tree, project.paths.map_i3d)
        export["i3d_references"] = references
        return export

    # -------------------------------------------------------------- helpers

    def _resolve_dem_full(self) -> np.ndarray:
        """DEM FULL para el background: el de la etapa DEM o el del disco."""
        if self.dem_pipeline is not None and self.dem_pipeline.dem_full is not None:
            return self.dem_pipeline.dem_full
        path = self.project.paths.background_dir / "FULL.png"
        dem_full = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if dem_full is None:
            raise FileNotFoundError(
                f"DEM FULL no encontrado: {path} (la etapa 'dem' fue saltada "
                "y no hay un FULL.png previo en el directorio de salida)"
            )
        if self.project.height_scale is None:
            raise RuntimeError(
                "project.height_scale no disponible para el background "
                "(la etapa 'dem' fue saltada)"
            )
        return dem_full

    # ------------------------------------------------------ generation_info

    def _write_generation_info(self) -> dict[str, Any]:
        """Escribe ``generation_info.json`` con la telemetría del pipeline."""
        project = self.project
        payload: dict[str, Any] = {
            "software": f"mapforge {__version__}",
            "date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "project": {
                "name": project.name,
                "size": project.map.size,
                "rotation": project.map.rotation,
                "rotated_size": project.map.rotated_size,
                "background_size": project.map.background_size,
                "latitude": project.map.latitude,
                "longitude": project.map.longitude,
                "output_size": project.map.output_size,
            },
            "inputs": {
                "heightmap": str(project.paths.heightmap),
                "osm": str(project.paths.osm),
                "template": str(project.paths.template),
                "texture_schema": str(project.paths.texture_schema),
                "grle_schema": str(project.paths.grle_schema),
            },
            "settings": project.settings.to_dict(),
            "height_scale": project.height_scale,
            "pipeline_order": list(STAGE_ORDER),
            "skipped_stages": self.skipped,
            "timings_s": self.timings,
            "components": self.components,
        }

        path = project.paths.generation_info_json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4, default=_json_safe)
        self.logger.info("generation_info.json escrito: %s", path)
        return payload
