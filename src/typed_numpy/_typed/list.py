"""
TypedList
=======
"""
# src/typed_numpy/_typed/list.py

import copy
import numbers
from types import GenericAlias
from typing import (
    Any,
    Generic,
    Literal,
    Self,
    Sequence,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

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
    return isinstance(value, expected_type)


def _validate_length(object: Sequence[Item], length: int | TypeVar | None) -> None:
    if not TypedListConfig.VALIDATE_LENGTH:
        return None

    origin = get_origin(length)

    # Literal[N]
    if origin is Literal:
        # [TODO] How should we handle multiple args case?
        length = get_args(length)[0]

    # ~TypeVar
    if isinstance(length, TypeVar):
        length = None

    if length is None:
        return None

    actual = len(object)
    if actual != length:
        raise LengthError(f"Length mismatch: expected {length}, got {actual}")


def _validate_item(object: Sequence[Item], item_type: Any) -> None:
    if not TypedListConfig.VALIDATE_ITEM:
        return

    for index, item in enumerate(object):
        if not _is_assignable(item, item_type):
            raise ItemError(
                f"Item type mismatch: expected {item_type.__name__}, got {type(item).__name__} at index {index}"
            )


## TypedList
class TypedList(Generic[Length, Item], list[Item]):
    @classmethod
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        ga = super().__class_getitem__(item)
        return _TypedListGenericAlias.from_generic_alias(ga)

    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    @property
    def length(self) -> Length:
        return self.__len__()

    @classmethod
    def full(
        cls: "type[TypedList[Length, Item]]", length: Length, fill_value: Item
    ) -> "TypedList[Length, Item]":
        return cls([copy.deepcopy(fill_value) for _ in range(length)])


class _TypedListGenericAlias(GenericAlias):
    """
    Deferred TypedNDArray constructor for shapes with TypeVars.
    Enables progressive type specialization, behaving like a type-level curry.
    """

    @classmethod
    def from_generic_alias(cls, alias: GenericAlias) -> Self:
        return cls(alias.__origin__, alias.__args__)  # pyright: ignore[reportArgumentType]

    def __getitem__(self, typeargs: Any) -> Self:
        ga = super().__getitem__(typeargs)
        return type(self).from_generic_alias(ga)

    def __call__(self, object: Sequence[Item], /) -> TypedList:
        # [NOTE] Should mimick `TypedList.__new__` signature

        base = cast(type[TypedList], get_origin(self))
        args = get_args(self)

        if len(args) == 2:
            length, item_type = args
        elif len(args) == 1:
            (length,) = args
            item_type = Item.__default__  # type: ignore[misc]
            # The `item_type` default here should match the default in `Item`.
        else:
            raise TypeError

        # Create `list` object
        data = base(object)

        # Runtime validations
        _validate_length(data, length)
        _validate_item(data, item_type)

        return data
