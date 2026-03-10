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

from typingkit.core._validators import validate_length
from typingkit.core.generics import RuntimeGeneric, get_runtime_args, propagate_runtime

## Typings

Length = TypeVar("Length", bound=int, default=int)
Item = TypeVar("Item", default=Any)


## Exceptions


class ItemError(Exception):
    """Raised when list item type doesn't match expected item type."""


## Runtime validation


class TypedListConfig:
    VALIDATE_LENGTH: bool = True
    VALIDATE_ITEM: bool = True

    @classmethod
    def enable_all(cls):
        cls.VALIDATE_LENGTH = True
        cls.VALIDATE_ITEM = True

    @classmethod
    def disable_all(cls):
        cls.VALIDATE_LENGTH = False
        cls.VALIDATE_ITEM = False


def _is_assignable(value: Any, expected_type: type) -> bool:
    if expected_type is Any:
        return True
    if expected_type is complex:
        return isinstance(value, numbers.Complex)
    if expected_type is float:
        return isinstance(value, numbers.Real)
    if expected_type is int:
        return isinstance(value, numbers.Integral)
    try:
        origin = get_origin(expected_type)
        if origin is not None:
            return isinstance(value, origin)
        return isinstance(value, expected_type)
    except Exception:
        return False


def _validate_item(object: Iterable[Item], item_type: Any) -> None:
    if not TypedListConfig.VALIDATE_ITEM:
        return

    for index, item in enumerate(object):
        if not _is_assignable(item, item_type):
            raise ItemError(
                f"Item type mismatch: expected {item_type.__name__}, got {type(item).__name__} at index {index}"
            )


## TypedList
class TypedList(RuntimeGeneric[Length, Item], list[Item]):
    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
        ## Runtime validations
        typeargs = get_runtime_args(alias)
        if len(typeargs) == 2:
            length, item_type = typeargs
        elif len(typeargs) == 1:
            (length,) = typeargs
            item_type = Item.__default__  # type: ignore[misc]
            # The `item_type` default here should match the default in `Item`.
        else:
            raise TypeError

        if TypedListConfig.VALIDATE_LENGTH:
            validate_length(self, length)
        _validate_item(self, item_type)

        # Propagate runtime type info into items that are themselves RuntimeGenerics
        origin = get_origin(item_type)
        if origin is not None and issubclass(origin, RuntimeGeneric):
            for item in self:
                propagate_runtime(item, item_type)
        elif isinstance(item_type, type) and issubclass(item_type, RuntimeGeneric):
            for item in self:
                propagate_runtime(item, item_type)

        return None

    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    def copy(self) -> Self:
        return type(self)(super().copy())

    @property
    def length(self) -> Length:
        return self.__len__()

    @classmethod
    def full(cls, length: Length, fill_value: Item | Callable[[int], Item]) -> Self:
        data: list[Item]
        if callable(fill_value):
            data = [cast(Item, fill_value(i)) for i in range(length)]
        else:
            data = [copy.deepcopy(fill_value) for _ in range(length)]
        return cls(data)
