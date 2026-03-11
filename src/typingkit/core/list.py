"""
TypedList
=======
"""
# src/typingkit/core/list.py

import copy
import numbers
from collections.abc import Iterable
from types import GenericAlias
from typing import Any, Callable, Self, TypeVar, cast, get_origin

from typingkit.core._config import TypedCollectionConfig
from typingkit.core._validators import LengthError, validate_length
from typingkit.core.generics import RuntimeGeneric, get_runtime_args, mapping_from_alias

__all__ = [
    "TypedList",
    "ItemError",
    "LengthError",
]


# ── Type parameters ───────────────────────────────────────────────────────────

Length = TypeVar("Length", bound=int, default=int)
Item = TypeVar("Item", default=Any)


# ── Exceptions ────────────────────────────────────────────────────────────────


class ItemError(Exception):
    """Raised when a list item's type doesn't match the expected item type."""


# ── Item assignability ────────────────────────────────────────────────────────

# Fast-path lookup for numeric tower
_NUMERIC_TOWER: dict[type, type] = {
    complex: numbers.Complex,
    float: numbers.Real,
    int: numbers.Integral,
}


def _is_assignable(value: Any, expected_type: type) -> bool:
    """Return True when *value* is an instance of *expected_type*, respecting Any and the numeric tower."""
    if expected_type is Any:
        return True
    abstract = _NUMERIC_TOWER.get(expected_type)
    if abstract is not None:
        return isinstance(value, abstract)
    origin = get_origin(expected_type)
    try:
        return isinstance(value, origin if origin is not None else expected_type)
    except TypeError:
        return False


def _validate_items(items: Iterable[Any], item_type: Any) -> None:
    if not TypedCollectionConfig.VALIDATE_ITEM or item_type is Any:
        return None
    for index, item in enumerate(items):
        if not _is_assignable(item, item_type):
            raise ItemError(
                f"Item type mismatch at index {index}: "
                f"expected {item_type.__name__!r}, got {type(item).__name__!r}"
            )
    return None


# ── TypedList ─────────────────────────────────────────────────────────────────


class TypedList(RuntimeGeneric[Length, Item], list[Item]):
    """
    A ``list`` subclass whose element type and optional length are enforced at
    runtime via the ``RuntimeGeneric`` machinery.

    Usage::

        TypedList[Literal[3]]([1, 2, 3])           # element type checked
        TypedList[Literal[3], int]([1, 2, 3])      # length + element type checked
    """

    # ── RuntimeGeneric hooks ──────────────────────────────────────────────────

    def __runtime_generic_iter_children__(
        self, mapping: dict[Any, Any]
    ) -> Iterable[tuple[Any, Any]]:
        """Yield each element paired with the resolved Item type."""
        item_type = mapping.get(Item, Any)  # type: ignore[misc]
        for elem in self:
            yield elem, item_type

    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
        args = get_runtime_args(alias)
        # Positional: [Length, Item] — both optional at runtime
        length: Any = args[0] if len(args) > 0 else Length.__default__  # type: ignore[misc]
        item_type: Any = args[1] if len(args) > 1 else Item.__default__  # type: ignore[misc]

        if TypedCollectionConfig.VALIDATE_LENGTH:
            validate_length(self, length)
        _validate_items(self, item_type)

        mapping = mapping_from_alias(alias, type(self))
        self.__runtime_generic_propagate_to_children__(mapping)

    # ── list API ──────────────────────────────────────────────────────────────

    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    def copy(self) -> Self:
        return type(self)(super().copy())

    @property
    def length(self) -> Length:
        return self.__len__()

    @classmethod
    def full(cls, length: Length, fill_value: Item | Callable[[int], Item]) -> Self:
        """Create a TypedList of `length` elements, filled via `fill_value` or a factory."""
        if callable(fill_value):
            data: list[Item] = [cast(Item, fill_value(i)) for i in range(length)]
        else:
            data = [copy.deepcopy(fill_value) for _ in range(length)]
        return cls(data)
