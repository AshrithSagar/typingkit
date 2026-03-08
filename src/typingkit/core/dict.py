"""
TypedDict
=======
"""
# src/typingkit/core/dict.py

from collections.abc import Mapping
from types import GenericAlias, NoneType, UnionType
from typing import Any, Literal, Self, TypeVar, cast, get_args, get_origin

from typingkit.core.generics import RuntimeGeneric, get_runtime_args

## Typings

Length = TypeVar("Length", bound=int, default=int)
Key = TypeVar("Key", default=Any)
Value = TypeVar("Value", default=Any)


## Exceptions


class LengthError(Exception):
    """Raised when dict length doesn't match expected length."""


## Runtime validation


class TypedDictConfig:
    VALIDATE_LENGTH: bool = True

    @classmethod
    def enable_all(cls):
        cls.VALIDATE_LENGTH = True

    @classmethod
    def disable_all(cls):
        cls.VALIDATE_LENGTH = False


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


def _validate_length(object: Mapping[Key, Value], length: Any) -> None:
    if not TypedDictConfig.VALIDATE_LENGTH:
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


## TypedDict
class TypedDict(RuntimeGeneric[Length, Key, Value], dict[Key, Value]):
    @classmethod
    def __pre_new__(cls, alias: GenericAlias, *args: Any, **kwargs: Any) -> Self:
        # Create `dict` object
        obj = super().__pre_new__(alias, *args, **kwargs)

        ## Runtime validations
        typeargs = get_runtime_args(alias)
        if len(typeargs) == 3:
            (length, _, _) = typeargs
        elif len(typeargs) == 2:
            (length, _) = typeargs
        elif len(typeargs) == 1:
            (length,) = typeargs
        else:
            raise TypeError
        _validate_length(obj, length)

        return obj

    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    @property
    def length(self) -> Length:
        return self.__len__()
