"""
TypedList
=======
A list subclass with runtime element-type and length enforcement.
"""
# src/typingkit/core/list.py

import copy
import numbers
from collections.abc import Iterable
from dataclasses import dataclass
from types import GenericAlias
from typing import Any, Callable, Self, TypeVar, cast, get_origin, override

from typingkit.core._options import RuntimeOptions
from typingkit.core._validators import LengthError, validate_length
from typingkit.core.generics import RuntimeGeneric, get_runtime_args, mapping_from_alias

__all__ = [
    "TypedList",
    "TypedListOptions",
    "ItemError",
    "LengthError",
]

Length = TypeVar("Length", bound=int, default=int)
Item = TypeVar("Item", default=Any)


@dataclass(frozen=True)
class TypedListOptions(RuntimeOptions):
    validate_length: bool = True
    validate_item_type: bool = True


_DEFAULT_LIST_OPTIONS = TypedListOptions()


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
    for index, item in enumerate(items):
        if not _is_assignable(item, item_type):
            raise ItemError(
                f"Item at index {index}: expected {item_type.__name__!r},"
                + f" got {type(item).__name__!r}"
            )


class TypedList(
    RuntimeGeneric[Length, Item],
    list[Item],
    options=_DEFAULT_LIST_OPTIONS,
):
    """
    A ``list`` subclass with runtime element-type and optional length enforcement.

    Usage::

        TypedList[Literal[3], int]([1, 2, 3])   # length + element type checked
        TypedList[int]([1, 2, 3])               # element type only
    """

    @override
    def __runtime_generic_iter_children__(
        self, mapping: dict[Any, Any]
    ) -> Iterable[tuple[Any, Any]]:
        item_type = mapping.get(Item, Any)  # type: ignore[misc]
        for elem in self:
            yield elem, item_type

    @override
    def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
        # get_runtime_args always returns a full-length tuple (defaults filled).
        # Safe to unpack directly against the class's parameter list.
        length, item_type = get_runtime_args(alias)

        opts = self.__class__._runtime_options_
        # Narrow to TypedListOptions for the domain-specific flag.
        # Falls back gracefully if a subclass switched to a plain RuntimeOptions.
        if isinstance(opts, TypedListOptions) and opts.validate_length:
            validate_length(self, length)
        if isinstance(opts, TypedListOptions) and opts.validate_item_type:
            _validate_items(self, item_type)

        # Propagate resolved types into child RuntimeGeneric instances.
        # We call this here (rather than relying on the base post_init) because
        # we've already resolved the mapping above — avoids a second build.
        self.__runtime_generic_propagate_children__(
            mapping_from_alias(alias, type(self))
        )

    @override
    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    @override
    def copy(self) -> Self:
        return type(self)(super().copy())

    @property
    def length(self) -> Length:
        return self.__len__()

    @classmethod
    def full(cls, length: Length, fill_value: Item | Callable[[int], Item]) -> Self:
        """
        Create a TypedList of ``length`` elements filled by value or factory.

        Args:
            length: number of items
            fill_value: value OR value factory (index -> value)
        """

        # Generate values
        if callable(fill_value):
            data = [cast(Item, fill_value(index)) for index in range(length)]
        else:
            data = [cast(Item, copy.deepcopy(fill_value)) for _ in range(length)]

        return cls(data)
