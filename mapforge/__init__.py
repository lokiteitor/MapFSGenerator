"""MapForge: generador local de mapas FS25 (alternativa a Maps4FS).

A partir de un heightmap uint16, un OSM local, el template FS25 y un
``config.yaml`` genera un mapa Farming Simulator 25 completo: DEM, texturas,
fields, farmlands, splines de tráfico y background procedural. Sin descargas,
determinista por seed.

Fuente de verdad de formatos y algoritmos: ``docs/analisis_forense_maps4fs.md``.
"""

__version__ = "0.1.0"
