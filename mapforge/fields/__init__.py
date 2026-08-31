"""Subpaquete fields: campos jugables en el map.i3d (Fase 4)."""

from mapforge.fields.fields import (
    FIELDS_ATTRIBUTES,
    NODE_ID_STARTING_VALUE,
    FieldsStats,
    FieldsWriter,
    add_fields_to_i3d,
    add_fields_to_i3d_file,
    build_field_node,
    field_layer_border,
    fit_polygon_into_bounds,
    get_user_attribute_node,
    load_fields_from_textures_json,
    polygon_area_ha,
    polygon_center,
    top_left_to_center,
    write_i3d,
)

__all__ = [
    "FIELDS_ATTRIBUTES",
    "NODE_ID_STARTING_VALUE",
    "FieldsStats",
    "FieldsWriter",
    "add_fields_to_i3d",
    "add_fields_to_i3d_file",
    "build_field_node",
    "field_layer_border",
    "fit_polygon_into_bounds",
    "get_user_attribute_node",
    "load_fields_from_textures_json",
    "polygon_area_ha",
    "polygon_center",
    "top_left_to_center",
    "write_i3d",
]
