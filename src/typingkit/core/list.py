"""
TypedList
=======
A list subclass with runtime element-type and length enforcement.
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

Length = TypeVar("Length", bound=int, default=int)
Item = TypeVar("Item", default=Any)


class ItemError(Exception):
    """Raised when a list item's type doesn't match the expected element type."""


_NUMERIC_TOWER: dict[type, type] = {
    complex: numbers.Complex,
    float: numbers.Real,
    int: numbers.Integral,
}


def _is_assignable(value: Any, expected_type: type) -> bool:
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
        return
    for index, item in enumerate(items):
        if not _is_assignable(item, item_type):
            raise ItemError(
                f"Item at index {index}: expected {item_type.__name__!r},"
                f" got {type(item).__name__!r}"
            )


class TypedList(RuntimeGeneric[Length, Item], list[Item]):
    """
    A ``list`` subclass with runtime element-type and optional length enforcement.

    Usage::

        TypedList[Literal[3], int]([1, 2, 3])   # length + element type checked
        TypedList[int]([1, 2, 3])               # element type only
    """

    def __runtime_generic_iter_children__(
        self, mapping: dict[Any, Any]
    ) -> Iterable[tuple[Any, Any]]:
        item_type = mapping.get(Item, Any)  # type: ignore[misc]
        for elem in self:
            yield elem, item_type

    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
        # Skip re-validation for instances already validated at construction.
        # Set first to guard against re-entrant calls during child propagation.
        if getattr(self, "_runtime_validated", False):
            return
        self._runtime_validated = True

        args = get_runtime_args(alias)
        length: Any = args[0] if len(args) > 0 else Length.__default__  # type: ignore[misc]
        item_type: Any = args[1] if len(args) > 1 else Item.__default__  # type: ignore[misc]

        if TypedCollectionConfig.VALIDATE_LENGTH:
            validate_length(self, length)
        _validate_items(self, item_type)

        self.__runtime_generic_propagate_children__(
            mapping_from_alias(alias, type(self))
        )

    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    def copy(self) -> Self:
        return type(self)(super().copy())

    @property
    def length(self) -> Length:
        return self.__len__()

    @classmethod
    def full(cls, length: Length, fill_value: Item | Callable[[int], Item]) -> Self:
        """Create a TypedList of ``length`` elements filled by value or factory."""
        if callable(fill_value):
            data: list[Item] = [cast(Item, fill_value(i)) for i in range(length)]
        else:
            data = [copy.deepcopy(fill_value) for _ in range(length)]
        return cls(data)
