"""Subpaquete OSM: parser XML propio, matching de tags y proyección (Fase 2)."""

from mapforge.osm.parser import (
    OsmData,
    OsmFeature,
    OsmRelation,
    OsmWay,
    features_matching,
    parse_osm,
)
from mapforge.osm.projection import (
    EARTH_RADIUS_M,
    MapProjection,
    bbox_for_map,
    bbox_from_point,
    latlon_to_pixel,
    project_geometry,
)
from mapforge.osm.tags import TagsDefinition, tags_match

__all__ = [
    "EARTH_RADIUS_M",
    "MapProjection",
    "OsmData",
    "OsmFeature",
    "OsmRelation",
    "OsmWay",
    "TagsDefinition",
    "bbox_for_map",
    "bbox_from_point",
    "features_matching",
    "latlon_to_pixel",
    "parse_osm",
    "project_geometry",
    "tags_match",
]
