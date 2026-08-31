# Análisis forense del mapa FS25_Valle_Bonito (Maps4FS 3.1.2)

**Fecha del análisis:** 2026-08-31
**Artefacto:** `FS25_Valle_Bonito/` (594 archivos, 1.4 GB)
**Generado por:** Maps4FS **3.1.2** (Windows, 2026-08-06), `custom_dem: true`, `custom_osm: true`
**Mapa:** 8192×8192 m, rotación 0, centro (43.145692, −95.145079), background 12288 m

---

## ⚠️ Caveat de contaminación de evidencia

El artefacto **no es salida pura de Maps4FS**. Cronología por timestamps:

| Fecha | Evento |
|---|---|
| 2026-08-06 21:09–22:38 | Generación Maps4FS (data/, previews/, background/, splines.i3d, overview) |
| 2026-08-07 00:14–00:19 | `map.i3d.bak` y `map_test.i3d` guardados |
| 2026-08-28 00:32 | `map.i3d`, `dem.png`, `*.gdm`, muchos `*_weight.png` re-guardados por **GIANTS Editor 10.0.13** |

El usuario editó el mapa en GIANTS Editor tras la generación (añadió placeables de `Documents/mods placebables/`, tráfico 80s_USTraffic, imports). Consecuencias:

- `map.i3d` actual y `map.i3d.bak` son **ambos post-Maps4FS**; no existe template limpio en el artefacto → el diff template→output (Fase 10) queda **limitado**.
- 20 referencias de archivo rotas en `map.i3d` apuntan a mods locales del usuario — **no** son un defecto de Maps4FS (las 721 restantes existen).
- Los archivos re-guardados el 28-ago pueden diferir en bytes de la salida original de Maps4FS, aunque sus estadísticas coinciden con los intermedios del 6-ago (p. ej. `dem.png` ≈ resize de `not_resized_with_flattened_roads.png`).
- La pureza de `map_test.i3d` es desconocida (difiere de `.bak` pero con la misma lista de Files).

---

## A. Árbol del artefacto (resumen)

```text
FS25_Valle_Bonito/
├── modDesc.xml, icon.dds, preview.dds, overview…
├── main_settings.json           ← parámetros de invocación (FACT)
├── generation_settings.json     ← TODOS los settings usados (FACT)
├── generation_info.json         ← telemetría del pipeline (FACT)
├── generation_logs.json, performance_report.json, render_pda.py
├── valle_bonito.png             ← DEM 12288×12288 uint16 (== background/not_substracted.png)
├── custom_osm.osm               ← OSM de entrada (JOSM, 29k nodos, 280 ways)
├── map/
│   ├── map.i3d                  ← escena principal (editada por el usuario el 28-ago)
│   ├── map.i3d.bak, map_test.i3d
│   ├── map.xml                  ← config de misión (farmlands.xml, fields.xml…)
│   ├── splines.i3d              ← 352 NurbsCurve de tráfico (salida Maps4FS pura)
│   ├── config/*.xml             ← farmlands.xml (192), fields.xml, aiSystem…
│   └── data/
│       ├── dem.png              ← 8193×8193 uint16
│       ├── *_weight.png         ← 86+ máscaras binarias 8192×8192 uint8
│       ├── infoLayer_*.png      ← fuente editable de los infoLayers
│       ├── infoLayer_*.grle     ← compilados por el motor (magic "GRLE")
│       ├── densityMap_*.png/.gdm
│       └── masks/PG_*.png       ← máscaras intermedias Maps4FS (roads, acres, water…)
├── background/
│   ├── FULL.png (12288² u16), not_substracted.png, not_resized.png (8192²),
│   │   not_resized_with_flattened_roads.png, FULL.obj (trimesh, 2.36M verts),
│   │   decimated_background.obj (167 556 verts)
│   └── textured_mesh/ (obj + mtl + dds/jpg)
├── assets/
│   ├── background/background_terrain_part_0{1..4}_binary.i3d (+.shapes, textura dds)
│   ├── map_bounds/map_bounds.i3d
│   └── roads/asphalt-white/ (road mesh binario)
├── roads/, water/, satellite/, previews/
```

Inventario completo (594 filas: path, ext, size, dims, mode, md5): `docs/inventario_completo.tsv`.

## B. Distribución por tipo

| ext | n | MB | | ext | n | MB |
|---|---|---|---|---|---|---|
| .dds | 263 | 292 | | .grle | 10 | 5.4 |
| .png | 179 | 359 | | .gdm | 6 | 20 |
| .i3d | 44 | 22.5 | | .obj | 5 | 260 |
| .shapes | 35 | 150 | | .cache | 3 | 282 |
| .xml | 36 | 1.7 | | .osm | 1 | 2.7 |

---

## C. DEM

### Cadena completa reconstruida (FACT — todos los intermedios están en el artefacto)

```text
valle_bonito.png                12288×12288 uint16, rango [2000, 29540]
        = background/not_substracted.png (md5 idéntico)
        ≈ background/FULL.png (estadísticas idénticas → DEM del background completo)
        │  crop central 8192×8192 (1 px = 1 m)
        ▼
background/not_resized.png      8192×8192, rango [2000, 25600]
        │  flatten_roads=true (cambios menores de píxel)
        ▼
background/not_resized_with_flattened_roads.png
        │  resize 8192 → 8193 (= tamaño_mapa + 1)
        ▼
map/data/dem.png                8193×8193 uint16, min=2000, max=25600,
                                mean=4640.6, median=3500
```

### Conversión altura física (FACT, verificado por 3 vías independientes)

```text
altura_m = valor_uint16 × heightScale / 65535 = valor_uint16 / 257.0
```

1. `generation_info.json` → `DEM.height_scale`: `adjusted_height_scale: 255`, **`mesh_z_scaling_factor: 257.0`** (65535/255 = 257.0).
2. Splines: la Y constante de spline_1 es `13.618677042801556` = 3500/257.0 exactamente (3500 = mediana/base del DEM).
3. Background: translation Y del shape = `114.94163424124514` = 29540/257.0 (máximo del DEM de entrada), y el mesh normalizado tiene rango z ≈ 107.14 = (29540−2000)/257.

- `heightScale="255"` en TerrainTransformGroup **=** setting `minimum_height_scale: 255` (con `adjust_terrain_to_ground_level: true` no se redujo).
- Resolución: `unitsPerPixel="1"`, muestras = tamaño+1 (8193 para 8192; consistente con 12289 para 12288 documentado).
- Settings usados: multiplier=1, blur_radius=3, plateau=15, ceiling=15, water_depth=15 (DOCUMENTED en settings; el efecto numérico exacto de plateau/ceiling sobre este DEM no es separable porque no tenemos el heightmap crudo pre-Maps4FS → UNKNOWN si `valle_bonito.png` es el input del usuario o ya está procesado).

---

## D. map.i3d — Terrain y DisplacementLayer

### TerrainTransformGroup (idéntico en map.i3d, .bak y map_test.i3d — FACT)

```xml
<TerrainTransformGroup name="terrain" static="true" collisionFilterGroup="0x100"
  collisionFilterMask="0xfffff9c3" heightMapId="1" patchSize="65" maxLODDistance="750"
  heightScale="255" unitsPerPixel="1" lodBlendStart="200" lodBlendEnd="300"
  lodTextureSize="8192" lodBlendStartDynamic="50" lodBlendEndDynamic="65"
  detailLodBlendDelta="5" materialId="…" castShadowMap="true"
  occNumLODs="1" occMaxLODDistance="300" occPatchSize="65" occLevel="2"
  occDistanceWeight="1" occMaxAdjacentFaces="10">
```

`heightMapId=1` → `<File fileId="1" filename="data/dem.png"/>`. `lodTextureSize` = tamaño de mapa.

### DisplacementLayer (FACT — idéntico en las 3 versiones del i3d)

| Atributo | Valor | Relación |
|---|---|---|
| name | terrainDisplacement | constante |
| size | 65536 | HYPOTHESIS: constante del template FS25 (no escala con mapa 8192) |
| tileSize | 16 | constante |
| numChannels | 6 | constante |
| cellSize | 2 | constante |
| viewDistance | 25 | constante |
| blendOutDistance | 5 | constante |
| **maxHeight** | **0.2** | **= setting `displacement_layer_max_height: 0.2`** (FACT de correspondencia; al ser el valor por defecto no discrimina si Maps4FS lo escribió o ya estaba) |
| densityMapShaderNames | terrainDisplacementMap | constante |

Contenido del bloque Terrain: 86 `<Layer>` de pintura, 2 `<DetailLayer>` (terrainDetail 11ch, terrainDetailHeight 12ch), 10 `<InfoLayer>`, 1 `<DisplacementLayer>`, `<FoliageSystem>` con 33 FoliageType, OccluderLods, ProceduralPlacementMasks.

---

## E. Layers / rasters

### InfoLayers declarados en map.i3d (FACT)

| InfoLayer | numChannels | PNG real | Contenido |
|---|---|---|---|
| environment | 4 | 2048² L, todo 0 | vacío |
| **farmlands** | **8** | **4096² L (2 m/px), IDs 1–192 + 255** | ver §G |
| soilMap | 2 | 1024² L, valores {0,1,2,3} | mapa de suelos |
| indoorMask | 1 | 16384² L, todo 0 | vacío |
| navigationCollision | 1 | — | — |
| tipCollision(Generated) | 1/2 | — | — |
| placementCollision(Generated) | 1/1 | — | — |
| **fieldType** | 1 | **16384² L, todo 0** | **los fields NO se pintan aquí** |

**FACT clave:** `map.i3d` referencia los infoLayers como **`data/infoLayer_*.png`**. Los `.grle` (magic `GRLE`, RLE de GIANTS) coexisten en disco pero son la caché compilada del motor. → **No hace falta implementar codec GRLE: basta escribir PNG con los canales correctos.** Lo mismo con densityMap_*.png (fuente, aquí en blanco/placeholder 16384²) vs .gdm (magic `"MDF`, compilado).

### Weight maps de texturas (FACT)

- 8192×8192 (1 px/m), uint8, **binarios {0, 255}** (dissolve=false → sin valores intermedios).
- Un PNG por Layer del terrain (86 layers = parejas `nombre01`/`nombre02`; ambos con contenido en este mapa).
- Los archivos `*_extended_*`, `*_bgforest_*`, `PS_*`, `BC_*` en data/ **no están referenciados por map.i3d** — son artefactos de trabajo de Maps4FS (INFERRED: máscaras auxiliares para poste/edificio/luz, funcionalidad que eliminamos).
- `data/masks/PG_*.png`: máscaras intermedias del pipeline OSM→raster (PG_acres 8192² binaria = fields; PG_roads, PG_water, PG_forests…). Evidencia directa de que Maps4FS rasteriza cada categoría OSM a máscara binaria antes de asignar texturas.

---

## F. Fields

**FACT — los fields viven en map.i3d como nodos, no como raster:**

```xml
<TransformGroup name="gameplay">
  <TransformGroup name="fields">
    <TransformGroup name="field1" translation="-1220 0 3955">   ← centroide (X Z mundo)
      <TransformGroup name="polygonPoints">
        <TransformGroup name="point1" translation="-189 0 -114"/> ← vértices RELATIVOS
        …                                                           al centroide, enteros
      </TransformGroup>
      <TransformGroup name="nameIndicator">
        <Note name="Note" text="field1&#xA;8.87 ha" color="4278190080" fixedSize="true"/>
      </TransformGroup>
      <TransformGroup name="teleportIndicator"/>
    </TransformGroup>
    … (151 fields; == generation_info "added_fields": 151)
```

- Los 151 fields provienen del preprocesado OSM: 154 polígonos `landuse=farmland|meadow` → split (88 líneas divisorias) → 151 con `padding: 4.0` y 41 agujeros (números exactos en `generation_info.json.osm_preprocessing` — FACT).
- `map/config/fields.xml` solo define estado de cultivo (1 entrada); la geometría vive en el i3d.
- fieldType infoLayer queda a 0: el juego deriva el field del polígono.

## G. Farmlands

**FACT (datos del PNG + farmlands.xml):**

- `infoLayer_farmlands.png`: **uint8 L, 4096×4096 = mitad de resolución del mapa (2 m/px)**, declarado `numChannels="8"` en el i3d.
- Codificación: **valor de píxel = ID de farmland**, sin canales compuestos.
- IDs presentes: **1..192 consecutivos, sin huecos** + **255** (1 213 264 px ≈ 7.2%).
- No hay fondo 0 (`fill_empty_farmlands: true` rellenó todo el mapa).
- `map/config/farmlands.xml` declara exactamente los IDs 1–192 con `priceScale`/`npcName`; **255 no está declarado** → INFERRED: 255 = zona no comprable (carreteras/farmyards excluidos o marcador "sin farmland").
- 151 fields + farmyards (`add_farmyards: true`) + relleno = 192 farmlands (INFERRED sobre el reparto exacto).
- `farmland_margin: 5` aplicado (DOCUMENTED; no verificable sin la geometría pre-margin).

## H. Traffic splines

**FACT — serialización real (map/splines.i3d, salida pura de Maps4FS):**

```xml
<i3D name="spline01" version="1.6" …>
  <Shapes>
    <NurbsCurve name="spline_1_original_{'highway': ['secondary', 'tertiary', 'road', 'service']}"
                shapeId="5000" degree="3" form="open">
      <cv c="3143.0, 13.618677042801556, -3114.0" />
      <cv c="3143, 13.618677042801556, -3109" />
      …
    </NurbsCurve>
    <NurbsCurve name="spline_1_reversed_{…}" shapeId="5001" …>
      <cv …/>  ← misma lista de puntos en orden inverso
    </NurbsCurve>
```

- 352 NurbsCurve = 176 pares original/reversed (add_reversed_splines=true).
- Nombre = `spline_{n}_{original|reversed}_{dict de tags OSM}`; shapeId consecutivos desde 5000.
- Puntos: coordenadas mundo `x, y, z`; **y = altura muestreada del DEM = uint16/257.0** (FACT: 13.618677… = 3500/257).
- **`spline_density: 2` = nº de puntos interpolados añadidos por segmento OSM** (FACT aritmético: segmento OSM de 250 m → CVs cada 83.3 m; segmento de 15 m → CVs cada 5 m).
- Reversed = reverse de la lista de CVs, sin flags adicionales.
- En `map.i3d` del artefacto solo hay 3 pares (spline_86/87/88, bajo `trafficSystem` y con translations ±3.5 laterales) — coincide con `total_fitted_roads: 3`, pero al estar el mapa editado a mano, **UNKNOWN** si Maps4FS los insertó en map.i3d o los importó el usuario desde splines.i3d (existe `map/Spline 86.i3d` suelto, típico de export manual).

## I. Background

**FACT — cadena completa presente:**

```text
FULL.png (12288² u16, DEM completo del background)
   │ malla grid trimesh: 1536×1536 vértices (paso 8.0052 m = 12288/1535)
   ▼
FULL.obj (2 359 296 verts, generado por trimesh)
   │ decimación ~14× (+ remove_center=true)
   ▼
decimated_background.obj (167 556 verts, XY plano / Z altura)
   │ swap de ejes a convención i3d (X, Y=altura, Z)
   ▼
background/textured_mesh/background_textured_mesh.obj (+ .mtl + background_texture.dds/jpg)
   │ split en 4 + conversión binaria (i3dConverter 1.0)
   ▼
assets/background/background_terrain_part_0{1..4}_binary.i3d (+ .i3d.shapes externos)
```

- Normalización de alturas del mesh: `z_mesh = (valor − max)/257` → rango [−107.16, 0]; el i3d de cada parte lo recoloca con `translation="0 114.94163424124514 0"` donde **114.9416 = max_input/257 = 29540/257** → altura absoluta = valor/257 (coherente con el terrain).
- Textura: satelital en este mapa (nosotros la sustituiremos por procedural).
- Integración en map.i3d (FACT): 5 `<File>` (4 partes + map_bounds) + `<ReferenceNode name="background_terrain_part_0N" referenceId=…/>` en la raíz de la escena, sin translation propia; `<ReferenceNode name="mapbounds" translation="0 1024 0"/>`.
- `generation_info.Background.Mesh`: x_size=y_size=12288, z_size=107.1401, centros ≈ ±6146 (mesh construido en coords de píxel y recentrado).

## J. Diff template → output

**No disponible en forma pura**: el artefacto no incluye el template limpio y tanto `map.i3d` como `map.i3d.bak` son post-generación (y el actual, post-edición). Lo observable:

- `.bak` → actual: renumeración de fileId/nodeId (667↔727…), placeables añadidos por el usuario, mismos 86 Layers, mismo Terrain/DisplacementLayer, mismos 151 fields.
- Huellas de Maps4FS sobre el template estándar de FS25 (INFERRED por nomenclatura): Files+ReferenceNodes de background/map_bounds/roads, TransformGroup `fields` poblado, splines, `farmlands.xml`/`fields.xml` regenerados, dem.png + weights + infoLayer_farmlands.png reemplazados, overview.dds.
- **Pendiente (experimento futuro):** conseguir el template base de Maps4FS (repo `maps4fs` templates) y diffear contra `map.i3d.bak`.

## K. Conclusiones clasificadas

### FACT
1. `dem.png` es uint16, **(tamaño+1)²** (8193² para mapa de 8192), 1 px/m, `heightScale=255`, `unitsPerPixel=1`.
2. **altura_m = uint16/257.0** (= ×255/65535); verificado en DEM↔splines↔background↔generation_info.
3. Pipeline DEM: input 12288² → crop central 8192² → flatten_roads → resize +1 px.
4. Proyección OSM→mundo **lineal en el bbox** (error < 1 m): `x = (lon−min_lon)/Δlon·S − S/2`, `z = (max_lat−lat)/Δlat·S − S/2`; +X este, +Z sur, origen centro. El bbox se dimensiona para que Δlon·cos(lat)·111320 ≈ Δlat·111132 ≈ S metros.
5. Splines: `NurbsCurve degree=3 form=open`, `<cv c="x,y,z"/>`, nombre `spline_N_{original|reversed}_{tags}`, reversed = puntos invertidos, y=altura DEM; `spline_density` = puntos interpolados añadidos por segmento.
6. Fields: 151 TransformGroups con polygonPoints relativos al centroide + Note de superficie; fieldType infoLayer a 0.
7. Farmlands: PNG uint8 4096² (S/2), píxel=ID, IDs 1..192 consecutivos + 255, sin fondo 0; sincronizado con farmlands.xml.
8. Weight maps: uint8 8192² binarios {0,255}, uno por Layer (86 layers en parejas 01/02).
9. map.i3d referencia PNG (no GRLE/GDM); GRLE/GDM son caché compilada del motor → **no necesitamos codec GRLE**.
10. DisplacementLayer: `size=65536 tileSize=16 numChannels=6 cellSize=2 viewDistance=25 blendOutDistance=5 maxHeight=0.2`.
11. Background: grid 1536² desde DEM 12288² → decimación → 4 partes i3d binarias + ReferenceNodes; translation Y = max_dem/257.
12. Configuración completa de la generación en `generation_settings.json` / telemetría en `generation_info.json` (números de polígonos, splits, holes, farmlands…).

### DOCUMENTED (settings usados, efecto no aislado en los datos)
- plateau=15, ceiling=15, water_depth=15, blur_radius=3, multiplier=1, farmland_margin=5, fields padding=4.0/fields_padding=3, smooth_radius=14.6, dissolve=false.

### INFERRED
- 255 en farmlands = "no comprable" (ausente de farmlands.xml).
- PG_*.png = máscaras intermedias por categoría OSM del pipeline de texturas.
- `*_extended_*`/`PS_*`/`BC_*` weights = artefactos auxiliares no referenciados (features que eliminamos).
- DisplacementLayer no se modifica salvo maxHeight (idéntico en 3 versiones; size=65536 no escala con el mapa de 8192).

### HYPOTHESIS
- El resto de atributos del DisplacementLayer son constantes del template FS25 para cualquier tamaño.
- densityMap_*.png en blanco a 16384² actúan solo como placeholder de dimensiones para que el motor genere los .gdm.

### UNKNOWN
- Si `valle_bonito.png` es el heightmap crudo del usuario o ya pasó por plateau/blur de Maps4FS.
- Algoritmo exacto de `dissolve=true` (este mapa usó false; los binarios {0,255} son la línea base del experimento §23 del documento madre).
- Semántica exacta de priority/exclude_weight en solapes (no separable sin experimento controlado).
- Quién insertó los 3 pares de splines en map.i3d (Maps4FS "fitted roads" vs import manual).
- Efecto binario exacto de farmland_margin=5 sobre la geometría.

---

## Implicaciones para la especificación de MapForge

Ya implementable con confianza (los formatos de salida están verificados):

1. **DEM processor**: aceptar uint16 (S_bg)² → crop (S)² → resize (S+1)² → escribir PNG uint16; exponer heightScale y documentar altura=v·hs/65535.
2. **I3D writer**: modificar solo `heightScale`, `lodTextureSize`, `DisplacementLayer@maxHeight`, Files de data/, y añadir Files+ReferenceNodes de background; poblar `gameplay/fields`.
3. **Splines writer**: template NurbsCurve exacto de §H (copiable literal).
4. **Farmlands**: raster uint8 (S/2)² píxel=ID + farmlands.xml sincronizado; IDs consecutivos desde 1; 255 reservado.
5. **Texturas**: rasterizar máscaras binarias uint8 (S)² {0,255} por layer, con parejas 01/02.
6. **Fields**: polígonos → TransformGroups con puntos enteros relativos al centroide + Note "fieldN\n{ha} ha".
7. **Background**: grid (S_bg/8)² desde DEM, decimación (~14×, target ~167k verts), remove_center, normalizar a max=0, translation Y=max/257, export OBJ→i3d en 4 partes.
8. **LayerEngine/Exporter**: PNG únicamente; sin GRLE.

Experimentos aún necesarios antes de implementar: ~~dissolve, priority/overlap, farmland_margin exacto, y diff contra template limpio de Maps4FS~~ → **resueltos por análisis del código fuente; ver Parte II.**

---

---

# PARTE II — Análisis del código fuente (maps4fs-1.8.242) y template limpio

**Fuente:** `maps4fs-1.8.242/` (código completo) y `data/fs25-map-template.zip` (template limpio), proporcionados el 2026-08-31.

**⚠ Caveat de versión:** el mapa se generó con Maps4FS **3.1.2**; este código es **1.8.242**. Cada conclusión de esta parte es *FACT-source-1.8*; cuando el artefacto 3.1.2 la confirma, pasa a FACT pleno. Las discrepancias observadas se listan en §S9.

## S1. Template limpio (FACT)

- `map/data/` está **vacío**: Maps4FS genera todos los rasters desde cero (los tamaños salen de `fs25-grle-schema.json`, ver S5).
- `map/splines.i3d` existe en el template como esqueleto vacío (`<Shapes/><Scene/><UserAttributes/>`) — las splines se insertan ahí, no en map.i3d (en 1.8).
- Template dimensionado para mapa base 2048: `lodTextureSize="2048"`, DisplacementLayer `size="16384"`.
- **Diff DisplacementLayer template→output**: | atributo | template | generado | mecanismo |
  |---|---|---|---|
  | size | 16384 | 65536 | **`size = map_size × 8`** (i3d.py `_update_parameters`) |
  | maxHeight | 0.2 | 0.2 | setting `displacement_layer_max_height` |
  | cellSize | 8 | 2 | **sin mecanismo en 1.8** → cambio de template o código 3.x (UNKNOWN) |
  | resto | idénticos | idénticos | constantes |
- Además 1.8 actualiza la luz `sun`: `lastShadowMapSplitBboxMin/Max = ∓map_size/2, [-128,148]`.
- `heightScale` del Terrain se actualiza solo si el DEM lo pidió (`shared_settings.change_height_scale`).

## S2. DEM — algoritmo exacto (dem.py + background.py, FACT-source)

```text
1. DTM provider → array float/int16 en METROS
2. resize a resolución de salida (INTER_LINEAR)
3. × multiplier
4. adjust_terrain_to_ground_level: shift para que min = plateau + water_depth
5. height_scale = ceil(max(minimum_height_scale, max + ceiling))   ← semántica de "ceiling"
   → shared: mesh_z_scaling_factor = 65535/height_scale
   → shared: height_scale_multiplier = height_scale/255
   → i3d Terrain.heightScale := height_scale
6. normalizar: clip(data/height_scale × 65535, 0, 65535) → uint16
7. blur: GaussianBlur kernel (r,r) sigmaX=sigmaY=10 (r forzado a impar; 0 = off)
   [el blur se aplica DESPUÉS de normalizar]
```

En background.py (el DEM "grande" se procesa una vez a background_size y de él sale todo):

```text
background_size = map_size + 2×2048 (Parameters.BACKGROUND_DISTANCE)
FULL.png (background_size²)
  → copia not_substracted.png
  → water_depth: subtract(water_depth × mesh_z_scaling_factor) SOLO bajo máscara de agua
     (por eso plateau/water no aparecían en la aritmética del artefacto: máscara vacía)
  → crop central map_size² → not_resized.png
  → resize a (map_size+1)² INTER_LINEAR → map/data/dem.png   ← el +1 confirmado en código
  → add_foundations (opcional): nivelar DEM bajo edificios a su media
```

## S3. Texturas — priority, width y dissolve RESUELTOS (texture.py, FACT-source)

**Orden de dibujo y prioridad:**
```text
capas ordenadas: priority None primero, luego priority DESCENDENTE
la capa base (priority == 0) se pospone al final
cumulative = None
por capa:
    mask = NOT cumulative
    dibujar polígonos de la capa (fillPoly 255) en su imagen
    output = imagen_capa AND mask        ← solo píxeles aún no reclamados
    cumulative |= output
base_layer = NOT cumulative              ← rellena todo lo restante
```
→ **El primer claim gana** (prioridad más alta primero). No hay mezcla ni pesos: máscaras binarias exclusivas. Empates: orden del schema.

**Width:** las líneas se convierten con `geometry.buffer(width)` → el `width` del schema es el **radio** del buffer (ancho total = 2×width). Responde a §20 del documento madre: NO es width/2.

**Dissolve (NO es blur):** cada píxel 255 de la máscara combinada se asigna aleatoriamente a UNA de las `count` sublayers (01/02): `sublayers[randint(0, count)][pixel] = 255`. La máscara original se guarda como `*_preview.png`. Sin dissolve solo se escribe la sublayer 01.

**Otros (FACT-source):**
- `fields_padding`: `polygon.buffer(-padding)` antes de rasterizar (si el resultado colapsa, se ignora).
- Capas `procedural`: sus máscaras se copian/fusionan a `masks/PG_*.png` (+ BLOCKMASK.png vacío).
- Los polígonos de fields/farmyards y las polilíneas de roads se guardan en `info_layers/textures.json` (coords de píxel) — es el puente hacia i3d/grle.
- Proyección: `bbox = ox.utils_geo.bbox_from_point(centro, dist=map_rotated_size/2)`; `latlon_to_pixel` = interpolación lineal en el bbox (confirma la transformación verificada en Parte I).
- Rotación: se rasteriza a `map_rotated_size` y se rota/recorta la imagen después (`rotate_textures`).
- Schema de capa (layer.py): `name, count, tags, width, color, exclude_weight, priority, info_layer, usage, background, invisible, procedural, border`.

## S4. Fields y splines en i3d (i3d.py, FACT-source)

- Fields: nodeId desde **2000**; polígonos de textures.json → `fit_object_into_bounds` → coords centro (`x−S/2`); centroide = `Polygon.centroid` (shapely, int); puntos relativos = `punto − centroide`; Note = `field{id}&#xA;0.00 ha` (1.8 no calcula ha; el artefacto 3.1.2 sí). Cada field lleva UserAttribute: `angle=0, missionAllowed=true, missionOnlyGrass=false, nameIndicatorIndex=1, polygonIndex=0, teleportIndicatorIndex=2`.
- Splines: se escriben en `map/splines.i3d` (no en map.i3d), nodeId/shapeId desde **5000**; `interpolate_points(num_points=spline_density)` = **puntos extra entre cada par** (docstring literal); reversed = `[::-1]`; `z = dem[y,x] × (1/multiplier) × (1/mesh_z_scaling_factor)` muestreado del `not_resized.png` (sin interpolación, clamp a bordes); cada spline lleva UserAttribute `maxSpeedScale=1, speedLimit=100`; en Scene se añade `<Shape name=… translation="0 0 0" shapeId nodeId/>`.
- Árboles (no lo implementaremos): ReferenceNodes con y del DEM, rotación aleatoria, schema de árboles JSON.

## S5. Farmlands e infoLayers (grle.py + fs25-grle-schema.json, FACT-source)

- Todos los infoLayer/densityMap PNG se crean como **ceros** con tamaño `map_size × multiplier` según el schema JSON: farmlands ×0.5, environment ×0.25, fieldType/indoor/tip/densityMaps ×2.0, navigation/placement ×1.0 — coincide 1:1 con el artefacto (4096/2048/16384/8192 ✓).
- Farmlands: lista = farmyards (si `add_farmyards`) + fields, de textures.json; por polígono: `fit_object_into_bounds(margin=farmland_margin)` → **margin = `buffer(margin, join_style='mitre')` en píxeles (1 px = 1 m)**; coords ÷2 (`polygon_points_to_np(divide=2)`); `fillPoly(image, id)`. IDs secuenciales desde 1, **límite 254** (aviso del Giants Editor). XML: `<farmland id priceScale npcName/>` + `pricePerHa` del setting `base_price`.
- `fill_empty_farmlands`: `image[image==0] = 255` — confirma el significado del 255 del artefacto.
- `add_grass`/plants: escribe el canal B de `densityMap_fruits.png` con valor de planta (meadow=131, smallDenseMix=33) bajo la máscara grass+forest (erosionada 3×3, bordes a 0); islas aleatorias de valores {65,97,129,161,193,225} si `random_plants`.

## S6. Background mesh (component_mesh.py, FACT-source)

```text
image = image.max() − image            ← inversión (por eso alturas negativas)
image = image[::resize_factor, ::resize_factor]      (resize_factor=8 → 1536²  ✓)
grid: 2 triángulos por celda
rotar 180° sobre Y, luego 180° sobre Z
decimación (opcional): simplify_quadric_decimation(percent=decimation_percent/100,
                                                   aggression=decimation_agression)
   defaults 1.8: apply_decimation=false, percent=25, aggression=3
escala [resize_factor, resize_factor, z_scaling_factor=1/mesh_z_scaling_factor]
reescala XY exacta al tamaño original (usando extents)
remove_center: diferencia booleana con caja centrada de lado map_size
```
El posicionamiento en el juego (translation Y = max/257) y el split en 4 partes + i3dConverter no están en 1.8 (son de 3.x) — pero la Parte I los documenta desde el artefacto.

## S7. Defaults 1.8 (settings.py) vs los usados en el artefacto

Los defaults difieren de los del mapa (p. ej. farmland_margin 0 vs 5, add_reversed_splines false vs true, fill_empty_farmlands false vs true). `SplineSettings.spline_density` docstring: *"the number of extra points that will be added between each two existing points"* — confirma la semántica.

## S8. Cambios de estado en el informe de la Parte I

| Ítem | Antes | Ahora |
|---|---|---|
| Algoritmo dissolve | UNKNOWN | **FACT-source**: asignación aleatoria por píxel a sublayers |
| Priority/overlap | UNKNOWN | **FACT-source**: prioridad descendente, primer claim gana, base=resto |
| farmland_margin | UNKNOWN | **FACT-source**: buffer mitre en px antes de ÷2 |
| width de líneas | HYPOTHESIS width/2 | **FACT-source**: buffer(width) = radio |
| DEM plateau/ceiling/blur | DOCUMENTED | **FACT-source**: semántica exacta (S2) |
| dem.png = S+1 | FACT artefacto | + confirmado en código |
| DisplacementLayer.size | HYPOTHESIS constante | **FACT-source**: = map_size × 8 |
| Diff template→output | pendiente | hecho (S1) |
| 255 farmlands | INFERRED | **FACT-source**: fill_empty_farmlands |
| Tamaños infoLayers | FACT observado | **FACT-source**: multiplicadores del grle-schema |

## S9. Discrepancias 1.8 ↔ artefacto 3.1.2 (pendientes menores)

- `cellSize` 8→2 del DisplacementLayer: sin mecanismo en 1.8 (¿template 3.x o código 3.x?).
- El artefacto tiene splines también DENTRO de map.i3d (3 pares "fitted") y road meshes (`roads/`), preprocessor de fields (split/merge/holes), decimated_background.obj separado, split del background en 4 partes, Note con hectáreas reales, npcName GRANDPA para farmlands de fields — todo funcionalidad 3.x que 1.8 no tiene. Para nuestra implementación, la Parte I (artefacto) manda en formatos de salida; la Parte II manda en algoritmos.
- `grass02_weight` y otras sublayers 02 con contenido pese a dissolve=false: en 1.8 solo se escribiría la 01 → comportamiento 3.x o efecto del re-guardado del editor (UNKNOWN menor).
