# flatten_roads y flatten_farmyard

Ingeniería inversa de `flatten_roads` (feature de Maps4FS 3.x) y diseño de
`flatten_farmyard` (extensión propia de MapForge). Implementación en
`mapforge/terrain/flatten.py`; métricas reproducibles con
`./venv/bin/python tools/analiza_flatten.py` → `output/analisis_flatten_roads/`.

Se usa la convención de clasificación de `docs/analisis_forense_maps4fs.md`:
**FACT** (medido directamente en el artefacto), **FACT-source-1.8** (leído en el
código 1.8.242), **INFERRED**, **HYPOTHESIS**, **UNKNOWN**.

> **Caveat de disponibilidad.** El código fuente del que disponemos es Maps4FS
> **1.8.242**, que **no contiene `flatten_roads`** (búsqueda exhaustiva en el
> árbol: la cadena solo aparece en artefactos 3.x). El mapa golden se generó con
> **3.1.2**. Todo lo que sigue sobre el algoritmo de 3.x sale de medir el par
> antes/después del artefacto y de sus logs, no de leer su código.

---

## Parte I — Lo que sí se demostró del golden

Fuente: `FS25_Valle_Bonito/background/not_resized.png` (sin aplanar) y
`not_resized_with_flattened_roads.png` (aplanado), ambos 8192² uint16, más
`FULL.png`, `not_substracted.png` y `generation_logs.json`.
Datos completos en `output/analisis_flatten_roads/evidencia_golden.json`.

### F1. El efecto es estrictamente local a las vías — **FACT**

Cambian **1 483 344 px** (2.2104 % del crop), con deltas de −715 a +983 unidades
uint16 (−2.78 a +3.82 m). Calculando la distancia euclídea de cada píxel
cambiado al eje más cercano de `roads_polylines`, **no hay ni un solo píxel
modificado a más de 20 px de una carretera**. No es un filtro global.

### F2. El radio del corredor escala con el `width` del schema — **FACT**

Midiendo por anillos de distancia, en zonas donde no interfieren corredores de
otra clase:

| capa | tags | `width` (radio del buffer) | último radio con píxeles afectados |
|---|---|---|---|
| `gravelSmall` | `secondary, tertiary, road, service` | 4 | **9** px |
| `asphaltDusty` | `motorway, trunk, primary` | 8 | **18** px |

Es decir, radio de influencia ≈ `2×width + 2`. Dentro del corredor solo cambia
~el 40-65 % de los píxeles: sobre terreno ya llano el aplanado es un no-op, lo
que es coherente con "poner el corredor a una altura objetivo".

### F3. La sección transversal queda constante — **FACT**

Corte perpendicular en el píxel de mayor delta (y=996, x=7583):

```
original  … 3457 3442 3422 3398 3368 … 2462 2396 2335 2281 2232 2190 2155 …
aplanado  … 3318 3318 3318 3318 3318 … 3318 3318 3318 3095 2862 2402 2155 …
```

Un único valor a lo ancho de todo el corredor, con una transición de ~3 px en
los bordes. Longitudinalmente el perfil es suave (mesetas y rampas), mucho más
que el terreno bajo la calzada.

### F4. El aplanado se aplica sobre el DEM completo del background — **FACT**

`crop(FULL.png, 8192² centrales)` es **byte a byte idéntico** a
`not_resized_with_flattened_roads.png`. El aplanado no opera sobre el crop del
mapa sino sobre los 12288² del background, y de ahí salen tanto `FULL.png` como
el intermedio y `map/data/dem.png`.

### F5. Corrección: el 0.98 % de `FULL.png` es flatten_roads, no agua — **FACT**

`docs/validacion_golden.md` atribuía la diferencia de `FULL.png` a la resta de
`water_depth` bajo la máscara de agua. Es incorrecto: el golden se generó con
`generate_water: false`, y `FULL.png` vs `not_substracted.png` difiere en
**1 486 903 px (0.9847 %)** con exactamente el mismo perfil de deltas que el par
aplanado/sin aplanar. De esos píxeles, 3 559 caen **fuera** del crop central
(las polilíneas tienen vértices en x=−3 y x=8195, y el corredor se sale del
mapa). La tabla de validación queda corregida.

### F6. Mecánica confirmada por los logs — **FACT**

`generation_logs.json`, en orden cronológico:

```
21:09:53  Extended context populated: 0 buildings, 88 roads, … (shift=512).
21:09:57  DEM cutout saved: …\map\data\dem.png
21:09:59  Not resized DEM saved: …\background\not_resized.png
21:10:01  Found 88 roads polylines in textures info layer.
21:10:01  Fitted the osm_object into the bounds: POLYGON ((8704 -512, … -512 -512 …))   ×88
21:12:08  Flattened roads saved to full DEM file: …\background\FULL.png
21:12:09  Not resized DEM with flattened roads saved to: …
21:12:10  Flattened roads saved to DEM file: …\map\data\dem.png
```

Lee `roads_polylines` de `info_layers/textures.json`, ajusta cada vía a un bbox
extendido (borde −512), tarda ~54 s en las 88 vías y **reescribe** `dem.png`,
que ya se había guardado sin aplanar dos minutos antes.

### F7. La malla de carreteras es consecuencia, no origen — **FACT**

`roads/asphalt-white/roads_asphalt-white.obj` lleva exactamente las alturas del
DEM aplanado (`z × (−257)` = valor uint16, verificado en varios vértices), lo
que hacía pensar que el aplanado se derivaba de ella. No: el componente `Road`
corre a las **21:14:43**, dos minutos *después* del aplanado. La malla se
construye sobre el DEM ya aplanado.

### F8. La regla de altura longitudinal de 3.x: no resuelta — **UNKNOWN**

Se descartaron con datos, midiendo contra el corredor real:

- media móvil del eje (error medio ~20 u16 para toda ventana de 11 a 101 px);
- media y mediana de la sección transversal del corredor;
- media del corredor por segmento OSM (patrón `create_foundations`);
- blur isótropo del DEM (da valores por debajo del objetivo donde el objetivo
  está por encima del terreno del eje).

Queda una pista sin agotar: la `y` real de la vía horizontal analizada es
**982.646** y `textures.json` la trunca a **982**. Ese muestreo sub-píxel explica
el desfase sistemático en los tramos llanos (el corredor queda 40 u16 por debajo
del terreno, que es exactamente interpolar entre las filas 982 y 983) pero no
cuadra en los tramos con pendiente. **HYPOTHESIS**: 3.x muestrea el DEM en las
coordenadas float de la polilínea, no en las truncadas que guarda
`textures.json`.

Por decisión de proyecto **no se persigue más**: el criterio pasó a ser la
calidad del terreno resultante, no la igualdad bit a bit con 3.1.2.

---

## Parte II — El algoritmo de MapForge

`mapforge/terrain/flatten.py`. Replica de F1-F4 la geometría demostrada y toma
sus propias decisiones donde el golden no es reproducible o no es deseable.

### Corredor y perfil

1. **Geometría**: `roads_polylines` de `info_layers/textures.json`, con el
   `width` de cada capa resuelto desde `config/texture_schema.json` por el
   string `tags` (que es literalmente `str(layer.tags)`). `width` es el RADIO
   del buffer (FACT-source-1.8, §S3 del informe forense).
2. **Perfil longitudinal**: el eje se densifica a 1 px por longitud de arco, se
   muestrea el DEM con interpolación bilineal y el perfil resultante se suaviza
   con una media móvil centrada de `flatten_roads_smooth` metros (default 25).
   Es lo que quita baches sin borrar las pendientes reales: una rampa constante
   sobrevive intacta a una media móvil (verificado en
   `test_suavizado_conserva_la_pendiente_real`).
3. **Sección transversal**: cada píxel toma la altura del punto de eje **más
   cercano** (Voronoi vía `cv2.distanceTransformWithLabels`), que es lo que
   reproduce F3 — sección constante, no una media del entorno.
4. **Talud**: peso `α = 1` hasta `width` px del eje (la calzada) y caída
   `smoothstep` (`t²(3−2t)`) hasta 0 en `width + feather`. `smoothstep` tiene
   derivada nula en los dos extremos: ni escalón en el borde de la calzada ni
   pliegue donde el talud muere contra el terreno.

### Decisiones que se apartan del golden (y por qué)

| Decisión | Golden 3.1.2 | MapForge | Motivo |
|---|---|---|---|
| Ancho del talud | radio total ≈ `2×width` | `width` de calzada + `2×width` de feather | Con feather = `width` la pendiente máxima del talud sale ~2.9× la del terreno; con `2×width` baja a ~2.3× y con `3×width` solo a ~2.1×. 2.0 es donde deja de compensar ensanchar el corredor (`DEFAULT_FEATHER_RATIO`). |
| Altura objetivo | regla desconocida (F8) | perfil del eje suavizado 25 m | Es la formulación que hace la calzada conducible, que era el objetivo. |
| Composición en cruces | desconocida | media ponderada por α | Sumar `α·objetivo` y `α` y normalizar hace el resultado independiente del orden de iteración y evita el escalón que deja un "el último gana" en las intersecciones. |
| Resize final a `(map_size+1)²` | tipo NEAREST | INTER_LINEAR | FACT-source-1.8; es la causa dominante de la diferencia residual de `dem.png` y se deja como está (ver `docs/validacion_golden.md`). |

### `flatten_farmyard` — extensión propia

Sin equivalente en Maps4FS. Sigue el patrón **FACT-source-1.8** de
`create_foundations` (`maps4fs-1.8.242/maps4fs/generator/component/background.py:104`):
máscara `cv2.fillPoly`, altura objetivo `= np.round(cv2.mean(dem, mask))`,
asignación uniforme. Añadidos propios:

- **Feather hacia fuera** (`flatten_farmyard_feather`, default 8 m): el interior
  queda completamente plano — que es lo que necesita una era para asentar
  edificios y maquinaria — y la transición ocurre en el terreno circundante, no
  dentro del recinto.
- **Salvaguarda de desnivel** (`flatten_farmyard_max_relief`, default 10 m).
  Necesaria: `landuse=farmyard` se usa en OSM con mucha manga ancha. En el mapa
  de validación hay **41 recintos**, y dos de ellos son
  `#32` (514 ha, **78.4 m** de desnivel interior) y `#36` (294 ha, **55.4 m**).
  Aplanarlos a su media no mejora el terreno: deja una meseta artificial de
  cientos de hectáreas rodeada de un talud de decenas de metros. Con la
  salvaguarda se descartan con warning y se aplanan los 39 restantes; el DEM
  modificado baja del 9.86 % al 4.38 %. `None` o `0` la desactiva.

### Orden de aplicación: farmyards → carreteras

Los recintos se aplanan primero y **el perfil de las vías se muestrea sobre el
DEM ya compuesto**. Así un camino de servicio que cruza una era hereda su altura
plana dentro del recinto y sale con transición suave al terreno, en vez de que
los dos aplanados compitan por los mismos píxeles
(`test_orden_farmyard_luego_carretera`).

### Dónde se engancha

En `DemPipeline._save_outputs`, sobre el DEM completo de `background_size²`
(F4), después de escribir `not_substracted.png` y `not_resized.png` — que se
mantienen **sin aplanar**, como en el golden — y antes de `FULL.png`:

```
not_substracted.png  ← full_raw
not_resized.png      ← crop(full_raw)                     sin aplanar
                       ↓ flatten_farmyards → flatten_roads
not_resized_with_flattened_roads.png ← crop(aplanado)
FULL.png             ← aplanado (+ hook de agua)
map/data/dem.png     ← resize(crop(aplanado), map_size+1, INTER_LINEAR)
```

`DemPipeline.dem_not_resized` pasa a ser el crop **aplanado** (el crudo queda en
`dem_not_resized_raw`), de modo que las splines de tráfico muestrean la
superficie de la calzada y no el terreno que quedó debajo.

Como el aplanado consume `info_layers/textures.json`, la etapa `textures` pasa a
correr **antes** que `dem` en `STAGE_ORDER` — el mismo orden que se ve en los
logs de 3.x (F6). Si el JSON no existe (etapa saltada) el aplanado se omite con
warning en vez de romper la generación.

---

## Parte III — Resultados medidos

Réplica del golden: `custom_osm.osm` + `valle_bonito.png` + los settings de
`generation_settings.json`, con `flatten_roads: true` y `flatten_farmyard: false`.

### Calidad del terreno (el criterio)

`output/analisis_flatten_roads/calidad.json`. Se muestrean cortes
perpendiculares cada 25 px a lo largo de cada vía y se estadística solo donde el
terreno original tenía pendiente transversal real (≥5 u16/px ≈ 2 cm/m); en este
mapa la mayoría de las vías van sobre la meseta llana, así que la muestra útil
es pequeña pero significativa.

| clase | cortes con pendiente | gradiente máx. del terreno (mediana) | del talud aplanado | ratio mediana / p95 / máx | escalón de corte duro evitado |
|---|---|---|---|---|---|
| `secondary…` (width 4) | 3 de 5 122 | 8 u16/px | 20 u16/px | 2.00 / 2.56 / 2.63 | 32 u16 (0.12 m) |
| `motorway…` (width 8) | 82 de 1 039 | 6 u16/px | 12 u16/px | 1.58 / 2.00 / 4.17 | 48 u16 (0.19 m) |

Lectura: el talud queda entre 1.6× y 2.6× la pendiente que el terreno ya tenía,
en vez del corte vertical de 0.12-0.19 m en un solo píxel que dejaría un
aplanado sin feather.

**Suavidad de las splines** (`suavidad_splines.json`, 30 367 muestras cada 5 m
sobre los ejes): el salto de altura entre puntos consecutivos baja de **1.486 m
a 0.786 m** en el peor caso; la mediana y el p95 (0.0156 m) no cambian porque el
terreno es llano en casi todo el recorrido.

### `flatten_farmyard` (`output/analisis_flatten_roads/farmyards.json`)

No hay golden con el que comparar — la feature no existe en Maps4FS —, así que
se validó con la misma entrada del replay pero con `dem.flatten_farmyard: true`
(además de los tests sintéticos de `tests/test_flatten.py`):

- **41 recintos** en el OSM del mapa. La salvaguarda descarta **2** por
  desnivel interior: `#32` (513.5 ha, 78.4 m) y `#36` (293.7 ha, 55.4 m).
- Los **39 restantes** se aplanan y **los 39 quedan con el interior
  perfectamente plano** al valor medio, una vez se excluyen los corredores de
  carretera.
- Esa exclusión es necesaria porque el orden es farmyards → carreteras: donde
  una vía cruza una era, la calzada labra su propio perfil sobre ella. Medido:
  el **100 %** de los píxeles no uniformes del interior caen dentro de un
  corredor de vía; ninguno queda sin explicar.
- Píxeles modificados del crop del mapa: 1 573 870 (2.35 %) solo con vías →
  6 517 976 (9.71 %) con vías y farmyards. Sin la salvaguarda serían 14.8 M
  (9.86 % del DEM del background contra el 4.38 % con ella): los dos recintos
  descartados solos pesaban más que todo lo demás junto.

### Comparación con el golden (informativa, no criterio)

`output/analisis_flatten_roads/comparacion_golden.json` y
`output/reporte_golden_e2e.json`.

| artefacto | antes de la feature | después | máx. delta antes → después |
|---|---|---|---|
| `background/not_substracted.png` | 0 % | **0 %** | 0 → 0 |
| `background/not_resized.png` | 0 % | **0 %** | 0 → 0 |
| `background/FULL.png` | 0.9847 % | **0.8690 %** | 983 → **507** u16 (3.82 → 1.97 m) |
| `background/not_resized_with_flattened_roads.png` | no se generaba | 1.9479 % | — → 507 u16; media 3.4 u16 (0.013 m), p99 49, >1 m solo el 0.0035 % |
| `map/data/dem.png` | 33.01 % | **32.78 %** | 1031 → **833** u16; >1 m del 0.0272 % al 0.0187 % |
| `map/splines.i3d` | 626 CVs (37.6 %) distintos solo en Y | 614 CVs (36.9 %) | XZ idénticos en ambos; delta Y medio 0.018 m |

Nuestro aplanado toca 1 579 009 px del DEM del background (1.046 %) frente a los
1 486 903 (0.985 %) del golden — algo más porque el corredor es más ancho
(`3×width` contra `2×width`).

El residuo de `dem.png` (32.78 %) **no** viene del aplanado: viene del resize
final. Con el DEM aplanado del propio golden, reescalar con NEAREST da 0.022 % y
con INTER_LINEAR 31.54 % (`output/validacion_dem/delta_stats.json`). MapForge
mantiene INTER_LINEAR porque es lo que dice el código 1.8.242; cambiarlo sería
una decisión aparte.

---

## Cobertura de tests

`tests/test_flatten.py` (38 tests). Núcleo plano y radio del corredor, exterior
intacto, transición monótona y acotada, feather proporcional y feather 0 como
contraejemplo, suavizado que quita baches pero conserva pendientes, cruces sin
discontinuidad, determinismo e independencia del orden, troceado interno sin
costuras, vías degeneradas/fuera de bounds/con tags desconocidos, farmyards
(interior a la media exacta, feather monótono, salvaguarda de desnivel,
solapes), orden farmyard→carretera, y la integración con `DemPipeline`
(incluido el FACT `crop(FULL) == not_resized_with_flattened_roads`).
