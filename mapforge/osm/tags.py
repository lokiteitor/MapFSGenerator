"""Matching de tags OSM estilo osmnx (Fase 2 del plan).

Semántica del dict de tags (la misma que usa el texture schema de Maps4FS,
que a su vez replica ``osmnx.features_from_bbox``):

- ``{"landuse": "farmland"}`` → el tag debe valer exactamente ese valor.
- ``{"natural": ["wood", "tree_row"]}`` → el tag debe valer uno de la lista.
- ``{"building": True}`` → basta con que el tag exista (cualquier valor).

Un dict con varias claves casa si CUALQUIERA de ellas casa (unión, como
``osmnx.features_from_bbox``): p. ej. ``{"natural": ["wood"], "landuse":
"forest"}`` selecciona los elementos que sean wood O forest.
"""

from __future__ import annotations

from typing import Any

# str | list[str] | True, por clave OSM.
TagsDefinition = dict[str, Any]


def _single_key_match(value: str, wanted: Any) -> bool:
    """¿Casa el valor ``value`` de un tag con lo pedido para esa clave?"""
    if wanted is True:
        return True
    if isinstance(wanted, str):
        return value == wanted
    if isinstance(wanted, (list, tuple, set)):
        return value in wanted
    raise ValueError(
        f"Valor de tag no soportado en la definición: {wanted!r} "
        "(se admite str, list[str] o True, semántica osmnx)"
    )


def tags_match(element_tags: dict[str, str], wanted: TagsDefinition) -> bool:
    """¿Casan los tags de un elemento OSM con la definición ``wanted``?

    Unión sobre las claves de ``wanted`` (basta con que una clave case).
    Un elemento sin tags nunca casa; una definición vacía tampoco casa.
    """
    if not element_tags or not wanted:
        return False
    for key, wanted_value in wanted.items():
        value = element_tags.get(key)
        if value is None:
            continue
        if _single_key_match(value, wanted_value):
            return True
    return False
