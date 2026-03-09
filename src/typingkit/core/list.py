"""
TypedList
=======
"""
# src/typingkit/core/list.py

import copy
import numbers
from collections.abc import Iterable, Sequence
from types import GenericAlias, NoneType, UnionType
from typing import Any, Callable, Literal, Self, TypeVar, cast, get_args, get_origin

from typingkit.core.generics import RuntimeGeneric, get_runtime_args

## Typings

Length = TypeVar("Length", bound=int, default=int)
Item = TypeVar("Item", default=Any)


## Exceptions


class LengthError(Exception):
    """Raised when list length doesn't match expected length."""


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


def _resolve_length(length: Any) -> Any:
    # ~TypeVar
    if isinstance(length, TypeVar):
        # [TODO]: How should typing.NoDefault be handled?
        length = _resolve_length(length.__default__)

    origin = get_origin(length)

    # Literal[A, B, ...]
    if origin is Literal:
        length = set(get_args(length))

    # Union[A, B, ...]
    if origin is UnionType:
        resolved = (_resolve_length(arg) for arg in get_args(length))
        result = set[Any]()
        for r in resolved:
            if isinstance(r, set):
                result |= r
            else:
                result.add(r)
        return result

    return length


def _validate_length(object: Sequence[Item], length: Any) -> None:
    if not TypedListConfig.VALIDATE_LENGTH:
        return None

    length = _resolve_length(length)

    # type[Any]
    if length is Any:
        return None

    # type[int]; Including <subclass of int>
    if isinstance(length, type) and issubclass(length, int):
        return None

    # Should already be disallowed statically, here we just raise a runtime error
    if length is NoneType:
        raise LengthError("Invalid length")
    # [TODO]: Others

    # [NOTE]: From a statical perspective, being strict,
    # prefer Literal[~int] over ~int, although we allow it here, for now.

    actual = len(object)
    if isinstance(length, set):
        if len(length) > 1:  # pyright: ignore[reportUnknownArgumentType]
            for arg in length:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(arg, int):
                    if arg == actual:
                        break

                ## [TODO]: Similar to the outer validation, so prolly can refactor through a recursive function call
                # type[Any]
                if arg is Any:
                    break

                # type[int]; Including <subclass of int>
                if isinstance(arg, type) and issubclass(arg, int):
                    break
            else:
                raise LengthError(
                    f"Length mismatch: expected one of {length}, got {actual}"
                )
        elif len(length) == 1:  # pyright: ignore[reportUnknownArgumentType]
            # Defer to the single case below
            length = length.pop()  # pyright: ignore[reportUnknownVariableType]
        else:  # len(length) == 0
            # This case should prolly never arise? Strictly speaking, statically.
            return None
    # (Concrete) int
    if isinstance(length, int):
        # This case is just for a minimal error message, we could already handle
        # it in the `set` case above, provided `_resolve_length` resolves it accordingly.
        if actual != length:
            raise LengthError(f"Length mismatch: expected {length}, got {actual}")

    return None  # Fallback


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
        _validate_length(self, length)
        _validate_item(self, item_type)
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
