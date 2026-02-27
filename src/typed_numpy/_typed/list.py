"""
TypedList
=======
"""
# src/typed_numpy/_typed/list.py

import copy
import numbers
from types import GenericAlias, NoneType, UnionType
from typing import (
    Any,
    Callable,
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
    try:
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

    def copy(self) -> Self:
        return type(self)(super().copy())

    @property
    def length(self) -> Length:
        return self.__len__()

    @classmethod
    def full(
        cls: "type[TypedList[Length, Item]]",
        length: Length,
        fill_value: Item | Callable[[int], Item],
    ) -> "TypedList[Length, Item]":
        data: list[Item]
        if callable(fill_value):
            data = [cast(Item, fill_value(i)) for i in range(length)]
        else:
            data = [copy.deepcopy(fill_value) for _ in range(length)]
        return cls(data)


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

    def __call__(self, object: Sequence[Item] | None = None, /) -> TypedList:
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
        data = base(object) if object is not None else base()

        # Runtime validations
        _validate_length(data, length)
        _validate_item(data, item_type)

        return data
