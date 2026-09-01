# Validación E2E contra el golden (FS25_Valle_Bonito)

Reporte de la Fase 9: regeneración completa del mapa Valle Bonito con el
pipeline de MapForge y comparación contra el artefacto real de Maps4FS 3.1.2
(`FS25_Valle_Bonito/`).

## Cómo se generó

```bash
./venv/bin/python -m mapforge generate -c config/valle_bonito.yaml
./venv/bin/python tools/compare_golden.py output/valle_bonito \
    --json output/reporte_golden_e2e.json --show-missing
```

- Inputs: `FS25_Valle_Bonito/valle_bonito.png` (DEM 12288² uint16, el mismo
  `custom_dem` del golden) + `FS25_Valle_Bonito/custom_osm.osm` + template
  `maps4fs-1.8.242/data/fs25-map-template.zip`.
- Settings: espejo de `FS25_Valle_Bonito/generation_settings.json`
  (`config/valle_bonito.yaml`), con `dem.custom_dem: true` como el
  `main_settings.json` del golden.
- Ejecución: **72 s** en total (texturas 17.4 s, dem 12.6 s incluido el
  aplanado de las 88 vías, grle 13.8 s, background 28.0 s), sin excepciones.
  Telemetría completa en `output/valle_bonito/generation_info.json`.
- Resumen de `compare_golden.py`: **148 ficheros comparados, 96 idénticos
  byte a byte (a nivel píxel/atributo), 52 con diferencias, 0 errores**;
  438 ficheros del golden no se generan (ver "No generado" abajo) y 10
  ficheros extra propios de MapForge.
- El fichero comparado que se suma respecto de la validación anterior es
  `background/not_resized_with_flattened_roads.png`, que MapForge ya genera
  desde que `flatten_roads` está implementado.

## Tabla por artefacto

Porcentaje de coincidencia = 100 − % de píxeles distintos (PNG) o resultado
del diff de atributos XML. Todas las desviaciones están explicadas en la
columna final y detalladas en la sección siguiente.

| Artefacto | Coincidencia | Desviación y causa |
|---|---|---|
| `background/not_substracted.png` | **100 %** (0 px distintos) | — |
| `background/not_resized.png` | **100 %** (0 px distintos) | — |
| `background/FULL.png` | 99.13 % (antes 99.02 %) | 0.87 % de píxeles, max delta 1.97 m (antes 3.82 m): resto del `flatten_roads` del golden que nuestro algoritmo no reproduce exactamente (regla de altura de 3.x desconocida, ver `docs/analisis_flatten_roads.md` §F8). **Corrección:** esta diferencia no es la resta de `water_depth` como se documentaba antes — el golden se generó con `generate_water: false` y el 0.98 % original era íntegramente flatten_roads (FACT §F5) |
| `background/not_resized_with_flattened_roads.png` | 98.05 % | Se genera desde que `flatten_roads` está implementado. Max delta 1.97 m, media 0.013 m, p99 0.19 m, **>1 m solo el 0.0035 %**: nuestro corredor es algo más ancho (`3×width` contra `2×width`) y el perfil objetivo se calcula distinto |
| `map/data/dem.png` | 67.22 % px iguales (antes 66.99 %); **>1 m solo 0.019 %** (antes 0.027 %; max 3.24 m, antes 4.01 m) | Dominado por el resize: 3.x usa NEAREST y MapForge INTER_LINEAR (FACT-source 1.8, no se cambia). Con el DEM aplanado del propio golden, NEAREST da 0.022 % y INTER_LINEAR 31.54 % (`output/validacion_dem/delta_stats.json`), así que el resize explica ~31.5 de los 32.8 puntos |
| `map/data/*_weight.png` (86 weights comparados) | 58 idénticos al 100 %; los 28 restantes ≥ 90.8 % (mediana de diferencia 0.33 %) | Preprocessor de fields 3.x (padding 4.0, split/merge → mudDark 0.58 %), capas `*_extended_*`/`_bgforest_` de 3.x que reasignan píxeles de roads/forest (asphalt* 0.06–0.63 %, forestGrass01 9.18 % peor caso), y bosques excluidos del proyecto |
| `map/data/infoLayer_farmlands.png` | **99.997 % bajo el mapeo biyectivo de IDs**; máscara 255 idéntica (IoU 1.0); 192+255 IDs en ambos | Maps4FS 3.1.2 enumeró farmyards/fields en otro orden ⇒ los IDs son una permutación (186/192 con correspondencia exacta); el diff píxel a píxel crudo da 7.2 % por el renumerado (`output/validacion_farmlands/reporte_farmlands.json`) |
| `map/config/farmlands.xml` | 192/192 entradas, `pricePerHa` idéntico | `npcName`/`priceScale` difieren (GRANDPA/0.6 en el golden): 3.x los cambió; 1.8 escribe FORESTER/1 (FACT-source replicado) |
| resto de `infoLayer_*.png` (14) y `densityMap_*` (5 de 6) | **100 %** | Creados en cero según `grle_schema` — idénticos al golden |
| `map/data/densityMap_fruits.png` | 70.74 % | `add_grass` (meadow=131 en canal B) no replicado: decisión del plan, capas GRLE en cero |
| `map/splines.i3d` | 176/176 curvas, nombres/atributos idénticos; **XY de los 1664 CVs 100 % idénticos**; 614 CVs (36.9 %) difieren SOLO en Z (media 0.018 m, max 1.58 m) | Ambos muestrean ya un DEM aplanado, pero con reglas de altura distintas (§F8). Antes de implementar `flatten_roads` eran 626 CVs contra un DEM sin aplanar |
| `map/map.i3d` — nodos del escritor | **100 %**: `heightScale=255`, `lodTextureSize=8192`, DisplacementLayer `size=65536 cellSize=2 maxHeight=0.2`, sun bbox `∓4096,-128/148` idénticos al golden | — |
| `map/map.i3d` — fields | **151/151 fields, centroides y nº de puntos exactos** (`output/validacion_fields/reporte_validacion_fields.json`) | El preprocessor 3.x no causó desviación medible en este mapa (los polígonos de textures.json coinciden) |
| `map/map.i3d` — background | 4/4 `<File>` + `ReferenceNode background_terrain_part_0N` presentes en ambos | En el golden están agrupados bajo un TransformGroup `backgroundTerrain` (edición del editor/usuario); MapForge los emite en la raíz de Scene (patrón del plan) |
| `map/map.i3d` — resto | difiere | Contenido añadido por el usuario al golden (pack 80s_USTraffic, casas de `map/import/`, luces), `Export version` 10.0.13 (editor) vs 10.0.2 (template) y `PerInstanceMaterialParameters` que añade el editor al guardar |
| `map/map.xml` | width/height=8192 idénticos | Diffs restantes: `precisionFarming`/`thPFConfig`/`sounds` añadidos a mano al golden |
| `modDesc.xml` | títulos "Valle Bonito" idénticos | El golden añade `<dependencies>` (edición del usuario) |
| `map/config/*` (aiSystem, fields, weed, … 12 ficheros) | **100 %** | — |
| `map/config/environment.xml`, `trafficSystem.xml`, `placeables.xml` | difieren | environment: 3.x escribe latitud/niebla (1.8 no toca este fichero — verificado en el source); traffic/placeables: contenido del usuario (vehículos americanos, placeables propios) |
| `map/data/masks/PG_acres.png`, `PG_farmyards`, `PG_grasslands`, `PG_meadows`, `PG_railways`, `PG_scrub` | **100 %** | — |
| `masks/PG_buildings/dirtpaths/roads/sideroads/forests` | shape distinto | 3.x genera esas máscaras a 9216²/12288² (resolución extendida para meshes de roads y forest del background); 1.8/MapForge las emite a map_size² |
| `masks/BLOCKMASK.png` | 99.99 % | 3.x pinta buildings en el blockmask (buildings excluidos) |
| `background/decimated_background.obj` | bbox idéntico (12288×12288, z −105.54…−0.03); 103 818 vs 167 556 vértices | Decimadores distintos (fast_simplification vs pymeshlab de 3.x) y `remove_center` por filtrado de caras vs resta booleana — desviación documentada en el plan; el tamaño de fichero difiere 39.7 % |
| `assets/background/` | 4 partes i3d XML + `background_texture.png` | El golden trae además `*_binary.i3d(.shapes)` y `.dds`: los compila el GIANTS Editor al guardar (FACT); la textura es procedural (sin satélite, decisión de proyecto) — `translation Y = 114.9416` idéntica a la regla del artefacto (max/257) |
| `icon.dds`, `preview.dds`, `map/overview.dds` | difieren (tamaño) | Los del golden se generaron del satélite/los sustituyó el usuario; MapForge conserva los del template |
| `map/map.i3d.shapes` | difiere | Binario compilado por el editor al guardar el golden; el motor lo regenera (solo-PNG/XML por diseño) |

## No generado (439 ficheros del golden)

Por categoría — todo fuera del alcance acordado del proyecto:

- **Contenido del usuario**: `map/config/80s_USTraffic/` (60+ ficheros),
  `map/import/` (casas, iglesia, bga, StreetLightPack…), `map/Spline 86.*`,
  `placeables_bak.xml`, `sounds.xml`.
- **Binarios que compila el motor/editor** (FACT §D: el i3d referencia PNG y
  el motor genera los binarios): `*.grle`, `*.gdm`, `*.i3d.shapes`,
  `*_binary.i3d`, `overview/preview` DDS.
- **Features 3.x no replicadas**: road meshes (`roads/`, `assets/roads/`),
  `unprocessedHeightMap.png`, `soilMap`, capas `BC_*`/`PS_*` (buildings,
  postes y luces), `*_extended_*`/`*_bgforest_*` weights, `assets/map_bounds/`,
  `background/textured_mesh/` y `FULL.obj`.
- **Satélite/agua** (excluidos por diseño): `satellite/`, `water/`,
  `previews/`.

## Extra en la salida (10 ficheros)

`generation_info.json` y `dem_info.json` (telemetría propia),
`info_layers/textures.json` (contrato interno fases 3-6), las 4 partes
`background_terrain_part_0N.i3d` en XML + `background_texture.png`
(equivalentes editables de los `_binary.i3d`+`.dds` del golden) y
`map/config/footballField.xml`/`pedestrianSystem.xml` (vienen del template;
el usuario los borró del golden).

## Conclusión

- El pipeline E2E termina sin excepción y produce la estructura completa del
  mod (misma jerarquía `map/`, `map/data/`, `map/config/`, `assets/`,
  `background/`).
- Los artefactos núcleo salen **idénticos o equivalentes**: DEM intermedios
  100 %, splines con XY exacto, fields 151/151 exactos, farmlands
  geométricamente idénticos (permutación de IDs), infoLayers en cero 100 %,
  nodos i3d del escritor 100 %.
- `flatten_roads` ya está implementado (`mapforge/terrain/flatten.py`), con un
  algoritmo propio orientado a la calidad del terreno más que a la igualdad bit
  a bit: el detalle, la evidencia forense y las métricas están en
  `docs/analisis_flatten_roads.md`. Junto a él va `flatten_farmyard`, una
  extensión propia apagada por defecto que no afecta a esta validación.
- Todas las diferencias restantes son atribuibles a: (1) features 3.x
  documentadas como no replicadas (preprocessor de fields, road meshes, agua,
  buildings/luces, add_grass) o replicadas con algoritmo propio (flatten_roads,
  cuya regla de altura exacta de 3.x no es recuperable), (2) contenido añadido
  a mano por el usuario al golden, o (3) binarios que compila el GIANTS
  Editor/motor.

Reporte JSON completo: `output/reporte_golden_e2e.json` (y texto en
`output/reporte_golden_e2e.txt`). Reportes de fase:
`output/validacion_dem/`, `output/fase2_osm_validation.json`,
`output/validacion_texturas_reporte.json`, `output/validacion_fields/`,
`output/validacion_farmlands/`, `output/validacion_splines/`,
`output/validacion_i3d/`, `output/validacion_background/`.
