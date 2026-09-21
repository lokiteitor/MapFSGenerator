# MapForge (MapFSGenerator)

Generador **local** de mapas de Farming Simulator 25: a partir de un
heightmap uint16, un fichero OSM y el template oficial de mapa FS25 produce
un mod de mapa completo (DEM, texturas, fields, farmlands, hierba jugable en
densityMap_fruits, splines de tráfico y terreno de background procedural por
relieve). Sin descargas, sin satélite, **determinista por seed**.

Es una alternativa local a [Maps4FS](https://github.com/iwatkot/maps4fs):
los formatos de salida y los algoritmos replican los verificados en la
ingeniería inversa del artefacto real (`docs/analisis_forense_maps4fs.md`,
Maps4FS 3.1.2 + código fuente 1.8.242).

## Requisitos e instalación

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt   # numpy, opencv-python, shapely,
                                             # pyyaml, trimesh, fast_simplification, pytest
```

## Uso

```bash
./venv/bin/python -m mapforge generate -c config/config.example.yaml [-v]
```

La salida se escribe en el `output_dir` del config con la estructura de mod
FS25 lista para abrir en GIANTS Editor:

```text
output/mi_mapa/
├── modDesc.xml, icon.dds, preview.dds
├── map/
│   ├── map.i3d          # terrain, DisplacementLayer, sun, fields, background refs
│   ├── map.xml, splines.i3d, config/*.xml (farmlands.xml regenerado)
│   └── data/            # dem.png, *_weight.png, infoLayer_*.png, densityMap_*.png, masks/
├── assets/background/   # 4 partes i3d del terreno de fondo + textura procedural
├── background/          # DEMs intermedios (FULL, not_resized…) + OBJ del mesh
├── info_layers/textures.json   # contrato interno (fields/farmyards/roads en px)
├── generation_info.json # telemetría de cada componente + tiempos
└── dem_info.json        # estadísticas por etapa del pipeline DEM
```

Validación opcional contra el artefacto golden (si está disponible en el entorno):

```bash
./venv/bin/python -m mapforge generate -c config/valle_bonito.yaml
./venv/bin/python tools/compare_golden.py output/valle_bonito --show-missing
```

Resultados y desviaciones explicadas: `docs/validacion_golden.md`.

## Configuración (`config.yaml`)

Ejemplo completo comentado en `config/config.example.yaml`; configuración activa del proyecto en `config/config.yaml`.

```yaml
project:
  name: "Mi Mapa"            # título del mod (modDesc.xml)
  output_dir: "output/mi_mapa"

map:
  size: 8192                 # lado del mapa en metros/píxeles
  rotation: 0                # rotación (0 en el alcance validado)
  latitude: 43.1456          # centro geográfico (proyección del OSM)
  longitude: -95.1450

inputs:
  heightmap: "maps/dem_new_12k.png"            # uint16, idealmente (size+4096)² a 1 px/m
  osm: "maps/custom.osm"                       # OSM XML local
  template: "templates/fs25-map-template.zip"  # template base FS25 incluido en el repo
  texture_schema: "config/texture_schema.json" # opcional (defaults del repo)
  grle_schema: "config/grle_schema.json"

settings:
  seed: 42                   # dissolve + textura procedural + islas (determinista)
  input_height_scale: 0.003891050583657588   # metros = valor_uint16 × escala (255/65535)
  dem:                       # pipeline S2 (multiplier, plateau, ceiling, blur, flatten_farmyard…)
    custom_dem: false        # true: el heightmap YA es el DEM final normalizado
  background:                # flatten_roads, procedural.{resize_factor, decimation, height_texture...}
  grle:                      # farmland_margin, add_farmyards, add_grass, base_grass, random_plants...
  i3d:                       # spline_density, add_reversed_splines, DisplacementLayer
  texture:                   # dissolve, fields_padding, skip_drains
```

Notas de semántica:

- **Heightmap**: `metros = valor × input_height_scale` (default 255/65535,
  es decir ÷257 — la convención del artefacto). Si no mide
  `(size+4096)²` se reescala con INTER_LINEAR. Con `dem.custom_dem: true`
  el PNG se toma como DEM final normalizado (así se generó el golden).
- **Seed**: alimenta el `dissolve` de texturas, el ruido fBm de la textura
  del background y las islas de vegetación aleatorias; misma config + misma seed ⇒ misma salida.
- **Hierba base y plantas** (`mapforge/fs25/plants.py`):
  - `grle.add_grass` (**on** por defecto) pinta la vegetación base en el canal R
    de `densityMap_fruits.png` (resolución 2× del mapa, escalado `INTER_NEAREST`).
    Fusiona la capa con `usage == "grass"` (base) y `usage == "forest"`
    (`forestGrass`), recortando un marco perimetral de 1 px a cero.
  - `grle.base_grass` define la variedad: `"meadow"` (131, default) o
    `"smallDenseMix"` (33).
  - `grle.random_plants` (**off** por defecto) siembra islas poligonales
    distorsionadas y redondeadas con valores `{65, 97, 129, 161, 193, 225}` dentro
    de la hierba erosionada. Los tamaños y porcentajes son configurables y el
    RNG sembrado garantiza total determinismo.
- **Textura de background por relieve** (`mapforge/background/texture.py`):
  - `background.procedural.height_texture` (**on** por defecto) modula la textura
    del terreno de fondo (4096²) según las cotas y pendientes reales del DEM en
    vez de usar ruido plano.
  - Divide el terreno en tres zonas con mezcla suave (`band_blend`) y jitter
    fractal (`band_noise`) para contornos orgánicos:
    - Valles y cotas bajas: transición entre `palette_low` (tierra) y
      `palette_high` (verde).
    - Pendientes y cotas medias: aflora `palette_rock` (gris roca) por encima
      de `rock_height` metros o en pendientes superiores a `slope_rock_deg`.
    - Cumbres: `palette_snow` (nieve) por encima de `snow_height` metros,
      evitando laderas demasiado verticales (`snow_slope_limit_deg`).
  - `texture_flip_v` permite invertir el eje vertical si el visor de GIANTS Editor
    lo requiere.
- **Aplanado del terreno** (`mapforge/terrain/flatten.py`), sobre el DEM
  completo del background y antes del resize:
  - `background.flatten_roads` (**on** por defecto) aplana el corredor de
    cada vía: la calzada queda plana a lo ancho (radio = `width` del schema)
    siguiendo un perfil longitudinal suavizado
    `background.flatten_roads_smooth` metros (25 por defecto), y el talud se
    integra con el terreno a lo largo de `background.flatten_roads_feather`
    metros (`null` = el doble del ancho de la vía). La transición usa
    `smoothstep`, así que no deja escalón ni pliegue.
  - `dem.flatten_farmyard` (**off** por defecto, extensión propia de
    MapForge) aplana el interior de cada `landuse=farmyard` a la media del
    área, con feather `dem.flatten_farmyard_feather` metros hacia fuera.
    `dem.flatten_farmyard_max_relief` (10 m) descarta los recintos con
    demasiado desnivel interior: `landuse=farmyard` se usa en OSM con mucha
    manga ancha y hay polígonos de cientos de hectáreas que no tiene sentido
    aplanar. Los farmyards se aplanan **antes** que las vías, para que un
    camino que cruza una era herede su altura plana.
  - Con cualquiera de los dos activo se escribe además
    `background/not_resized_with_flattened_roads.png`, y las splines de
    tráfico muestrean el DEM aplanado. Detalle y métricas en
    `docs/analisis_flatten_roads.md`.
- Los settings aceptan también las claves estilo Maps4FS
  (`DEMSettings`, `BackgroundSettings`, …), por lo que un
  `generation_settings.json` de Maps4FS se puede volcar directamente.

## Arquitectura

Pipeline orquestado por `mapforge/generator.py` en este orden:

```text
template → OSM/texturas → DEM → grle_layers → farmlands → plants
         → fields → splines → escritor i3d → background
```

Las texturas van antes que el DEM porque el aplanado de carreteras y farmyards
consume `info_layers/textures.json` (`roads_polylines` y `farmyards`) — el mismo
orden que se observa en los logs de Maps4FS 3.x.

| Módulo | Responsabilidad |
|---|---|
| `mapforge/project.py`, `settings.py` | modelo de proyecto (rutas, params, seed) y dataclasses de settings |
| `mapforge/fs25/template.py` | despliegue del template FS25 (zip o directorio) |
| `mapforge/terrain/dem.py` | pipeline DEM S2: metros → multiplier → shift → height_scale → normalizar → blur → FULL/not_resized/dem.png |
| `mapforge/terrain/flatten.py` | aplanado del terreno bajo carreteras (`flatten_roads`) y farmyards (`flatten_farmyard`), con talud `smoothstep` y composición ponderada |
| `mapforge/osm/` | parser OSM XML propio (nodos/ways/relations → shapely), matching de tags estilo osmnx y proyección bbox+latlon→píxel |
| `mapforge/textures/` | weight maps por prioridad ("el primer claim gana"), base = NOT cumulative, dissolve seeded, máscaras `PG_*` y `info_layers/textures.json` |
| `mapforge/farmlands/` + `fs25/grle_layers.py` | infoLayers/densityMaps en cero + `infoLayer_farmlands.png` (IDs ÷2, tope 254, fill 255) + `farmlands.xml` |
| `mapforge/fs25/plants.py` | hierba jugable en `densityMap_fruits.png` (canal R: meadow=131 / smallDenseMix=33, suma hierba+bosque) + islas aleatorias deterministas (`random_plants`) |
| `mapforge/fields/` | polígonos → `TransformGroup field{n}` con polygonPoints/nameIndicator/teleportIndicator en `gameplay/fields` del map.i3d |
| `mapforge/splines/` | roads → NurbsCurve (density interpolada, Z del DEM, original+reversed) en `map/splines.i3d` |
| `mapforge/fs25/i3d_writer.py` | map.i3d (heightScale, DisplacementLayer, sun, refs de background), map.xml, modDesc.xml |
| `mapforge/background/` | mesh del terreno de fondo (subsample, decimación quadric, remove_center) + textura procedural por relieve (verde / roca / nieve según cota y pendiente) + export OBJ/i3d en 4 partes |
| `tools/compare_golden.py` | harness de validación contra `FS25_Valle_Bonito/` |
| `tools/analiza_flatten.py` | evidencia forense de `flatten_roads` + métricas de calidad del talud y de suavidad de las splines |

Cada componente deja su telemetría en `generation_info.json` (estadísticas +
tiempo por etapa). El generador admite `skip_stages` (API Python) para saltar
etapas con constancia en el log y en la telemetría.

Principios de diseño (ver informe forense):

- **Solo PNG/XML, nunca GRLE/GDM/binarios**: el i3d referencia `.png` y el
  motor/GIANTS Editor compila los binarios al guardar.
- Los i3d del background se emiten como XML 1.6 con `IndexedTriangleSet`
  inline (el editor los compila a `_binary.i3d.shapes`).
- Paridad de primitivas con Maps4FS 1.8.242 (cv2, shapely, mismos órdenes de
  operación) para reproducir sus salidas píxel a píxel donde es posible.

## Tests

```bash
./venv/bin/pytest          # unitarios por componente + mini-E2E del pipeline (177 tests)
```

## Limitaciones conocidas

Features de Maps4FS **3.x** presentes en el golden y no replicadas (el
código fuente disponible es 1.8.242; impacto cuantificado en
`docs/validacion_golden.md`):

- **Preprocessor de fields** (split/merge/holes con padding 4.0): los
  polígonos de fields se rasterizan tal cual vienen del OSM (con
  `fields_padding` propio). En el golden causa diferencias sub-porcentuales
  en los weights de fields.
- **Road meshes** (`roads/`, `assets/roads/`, máscaras PG a 9216²): no se
  generan mallas 3D de carreteras.
- **Regla de altura exacta de `flatten_roads`**: la feature está
  implementada, pero con algoritmo propio (ver arriba). La regla con la que
  3.x calcula el perfil longitudinal de la calzada no es recuperable — su
  código no está disponible y el par antes/después del golden no la
  determina —, así que el DEM aplanado no coincide bit a bit con el suyo:
  `docs/analisis_flatten_roads.md` recoge la evidencia y las métricas.
- **Resize final del DEM**: MapForge usa `INTER_LINEAR` (FACT-source 1.8) y
  3.x un resize tipo NEAREST. Es la causa dominante (~31.5 de 32.8 puntos)
  de la diferencia residual de `map/data/dem.png` contra el golden.

Exclusiones de alcance del proyecto (decisión, no limitación técnica):
bosques/árboles 3D, buildings, postes y luces (`BC_*`/`PS_*`), agua
procedural y resta de `water_depth` por máscara, imágenes de satélite (la textura del background es
procedural por relieve), `rotation != 0` y `output_size` (no validados). Los binarios
(`.grle`, `.gdm`, `.i3d.shapes`, `.dds` del background) los genera el
motor/editor al abrir/guardar el mapa.

## Documentación

- `docs/analisis_forense_maps4fs.md` — informe de ingeniería inversa
  (Parte I: formatos del artefacto; Parte II: algoritmos del source 1.8).
- `docs/validacion_golden.md` — validación E2E contra el golden, tabla por
  artefacto con % de coincidencia y desviaciones explicadas.
- `docs/analisis_flatten_roads.md` — ingeniería inversa de `flatten_roads`
  (evidencia FACT/HYPOTHESIS del golden), diseño del algoritmo propio y de
  `flatten_farmyard`, y métricas de calidad del terreno resultante.
- `FS25_Valle_Bonito/` — golden output de referencia (mapa real generado con
  Maps4FS 3.1.2) con sus inputs (`valle_bonito.png`, `custom_osm.osm`).
