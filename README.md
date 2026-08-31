# MapForge (MapFSGenerator)

Generador **local** de mapas de Farming Simulator 25: a partir de un
heightmap uint16, un fichero OSM y el template oficial de mapa FS25 produce
un mod de mapa completo (DEM, texturas, fields, farmlands, splines de
tráfico y terreno de background procedural). Sin descargas, sin satélite,
**determinista por seed**.

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

Validación contra el golden del repo:

```bash
./venv/bin/python -m mapforge generate -c config/valle_bonito.yaml
./venv/bin/python tools/compare_golden.py output/valle_bonito --show-missing
```

Resultados y desviaciones explicadas: `docs/validacion_golden.md`.

## Configuración (`config.yaml`)

Ejemplo completo comentado en `config/config.example.yaml`; espejo de los
settings del golden en `config/valle_bonito.yaml`.

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
  heightmap: "ruta/dem.png"  # uint16, idealmente (size+4096)² a 1 px/m
  osm: "ruta/mapa.osm"       # OSM XML local (custom_osm)
  template: "maps4fs-1.8.242/data/fs25-map-template.zip"
  texture_schema: "config/texture_schema.json"   # opcional (defaults del repo)
  grle_schema: "config/grle_schema.json"

settings:
  seed: 42                   # dissolve + textura procedural (determinista)
  input_height_scale: 0.003891050583657588   # metros = valor_uint16 × escala (255/65535)
  dem:                       # pipeline S2 (multiplier, plateau, ceiling, blur…)
    custom_dem: false        # true: el heightmap YA es el DEM final normalizado
  background:                # remove_center, procedural.{resize_factor, decimation, textura}
  grle:                      # farmland_margin, add_farmyards, base_price, fill_empty_farmlands
  i3d:                       # spline_density, add_reversed_splines, DisplacementLayer
  texture:                   # dissolve, fields_padding, skip_drains
```

Notas de semántica:

- **Heightmap**: `metros = valor × input_height_scale` (default 255/65535,
  es decir ÷257 — la convención del artefacto). Si no mide
  `(size+4096)²` se reescala con INTER_LINEAR. Con `dem.custom_dem: true`
  el PNG se toma como DEM final normalizado (así se generó el golden).
- **Seed**: alimenta el `dissolve` de texturas y el ruido fBm de la textura
  del background; misma config + misma seed ⇒ misma salida.
- Los settings aceptan también las claves estilo Maps4FS
  (`DEMSettings`, `BackgroundSettings`, …), por lo que un
  `generation_settings.json` de Maps4FS se puede volcar directamente.

## Arquitectura

Pipeline orquestado por `mapforge/generator.py` en este orden:

```text
template → DEM → OSM/texturas → grle_layers → farmlands → fields
         → splines → escritor i3d → background
```

| Módulo | Responsabilidad |
|---|---|
| `mapforge/project.py`, `settings.py` | modelo de proyecto (rutas, params, seed) y dataclasses de settings |
| `mapforge/fs25/template.py` | despliegue del template FS25 (zip o directorio) |
| `mapforge/terrain/dem.py` | pipeline DEM S2: metros → multiplier → shift → height_scale → normalizar → blur → FULL/not_resized/dem.png |
| `mapforge/osm/` | parser OSM XML propio (nodos/ways/relations → shapely), matching de tags estilo osmnx y proyección bbox+latlon→píxel |
| `mapforge/textures/` | weight maps por prioridad ("el primer claim gana"), base = NOT cumulative, dissolve seeded, máscaras `PG_*` y `info_layers/textures.json` |
| `mapforge/farmlands/` + `fs25/grle_layers.py` | infoLayers/densityMaps en cero + `infoLayer_farmlands.png` (IDs ÷2, tope 254, fill 255) + `farmlands.xml` |
| `mapforge/fields/` | polígonos → `TransformGroup field{n}` con polygonPoints/nameIndicator/teleportIndicator en `gameplay/fields` del map.i3d |
| `mapforge/splines/` | roads → NurbsCurve (density interpolada, Z del DEM, original+reversed) en `map/splines.i3d` |
| `mapforge/fs25/i3d_writer.py` | map.i3d (heightScale, DisplacementLayer, sun, refs de background), map.xml, modDesc.xml |
| `mapforge/background/` | mesh del terreno de fondo (subsample, decimación quadric, remove_center) + textura procedural fBm + export OBJ/i3d en 4 partes |
| `tools/compare_golden.py` | harness de validación contra `FS25_Valle_Bonito/` |

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
./venv/bin/pytest          # unitarios por componente + mini-E2E del pipeline
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
- **`flatten_roads`**: el DEM no aplana las carreteras; `map/data/dem.png`
  difiere del golden en las carreteras (≤ 4 m, > 1 m solo en el 0.03 % de
  los píxeles) y las splines heredan diferencias de Z ≤ 1.2 m.

Exclusiones de alcance del proyecto (decisión, no limitación técnica):
bosques/árboles, buildings, postes y luces (`BC_*`/`PS_*`), agua
procedural y resta de `water_depth` por máscara, `add_grass` (densityMap de
plantas en cero), imágenes de satélite (la textura del background es
procedural), `rotation != 0` y `output_size` (no validados). Los binarios
(`.grle`, `.gdm`, `.i3d.shapes`, `.dds` del background) los genera el
motor/editor al abrir/guardar el mapa.

## Documentación

- `docs/analisis_forense_maps4fs.md` — informe de ingeniería inversa
  (Parte I: formatos del artefacto; Parte II: algoritmos del source 1.8).
- `docs/validacion_golden.md` — validación E2E contra el golden, tabla por
  artefacto con % de coincidencia y desviaciones explicadas.
- `FS25_Valle_Bonito/` — golden output de referencia (mapa real generado con
  Maps4FS 3.1.2) con sus inputs (`valle_bonito.png`, `custom_osm.osm`).
