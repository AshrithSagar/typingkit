"""
TypedDict
=======
A dict subclass with runtime length enforcement.
"""
# src/typingkit/core/dict.py

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import GenericAlias
from typing import Any, Self, TypeVar, cast, overload, override

from typingkit.core._options import RuntimeOptions
from typingkit.core._validators import LengthError, validate_length
from typingkit.core.generics import RuntimeGeneric, get_runtime_args, mapping_from_alias

__all__ = [
    "TypedDict",
    "TypedDictOptions",
    "LengthError",
]

Length = TypeVar("Length", bound=int, default=int)
Key = TypeVar("Key", default=Any)
Value = TypeVar("Value", default=Any)

IMMUTABLES = (int, float, str, bool, tuple, frozenset, bytes, type(None))


@dataclass(frozen=True)
class TypedDictOptions(RuntimeOptions):
    validate_length: bool = True


_DEFAULT_DICT_OPTIONS = TypedDictOptions()


class TypedDict(
    RuntimeGeneric[Length, Key, Value],
    dict[Key, Value],
    options=_DEFAULT_DICT_OPTIONS,
):
    """
    A ``dict`` subclass with optional runtime length enforcement.

    Usage::

        TypedDict[Literal[2], str, int]({"a": 1, "b": 2})  # length checked
        TypedDict[int, str, int]({"a": 1})                 # unconstrained length
    """

    @override
    def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
        # get_runtime_args always returns a full-length tuple (defaults filled).
        # Safe to unpack directly against the class's parameter list.
        length, _key, _value = get_runtime_args(alias)

        opts = self.__class__._runtime_options_
        # Narrow to TypedDictOptions for the domain-specific flag.
        # Falls back gracefully if a subclass switched to a plain RuntimeOptions.
        if isinstance(opts, TypedDictOptions) and opts.validate_length:
            validate_length(self, length)

        # Propagate resolved types into child RuntimeGeneric instances.
        # We call this here (rather than relying on the base post_init) because
        # we've already resolved the mapping above — avoids a second build.
        self.__runtime_generic_propagate_children__(
            mapping_from_alias(alias, type(self))
        )

    @override
    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    @property
    def length(self) -> Length:
        return self.__len__()

    @overload
    @classmethod
    def full(
        cls,
        keys: Iterable[Key],
        fill_value: Value | Callable[[Key], Value],
        *,
        length: None = ...,
    ) -> Self: ...
    @overload
    @classmethod
    def full(
        cls,
        keys: Iterable[Key],
        fill_value: Value | Callable[[Key], Value],
        *,
        length: Length,
    ) -> Self: ...
    @overload
    @classmethod
    def full(
        cls,
        keys: Callable[[int], Key],
        fill_value: Value | Callable[[Key], Value],
        *,
        length: Length,
    ) -> Self: ...
    #
    @classmethod
    def full(
        cls,
        keys: Iterable[Key] | Callable[[int], Key],
        fill_value: Value | Callable[[Key], Value],
        *,
        length: Length | None = None,
    ) -> Self:
        """
        Create a TypedDict of ``length`` items.

        Args:
            keys: iterable of keys OR key factory (index -> key)
            fill_value: value OR value factory (key -> value)
            length: number of items (optional if keys is an iterable)
        """

        if length is not None and length < 0:
            raise ValueError(f"Length must be non-negative, got {length}")

        # Generate keys
        if callable(keys):
            if length is None:
                raise TypeError(
                    "Length must be provided when keys is a callable (index -> key)"
                )
            key_list = [keys(index) for index in range(length)]
        else:
            key_list = list(keys)
            if length is None:
                length = cast(Length, len(key_list))
            elif len(key_list) != length:
                raise LengthError(
                    f"Expected {length} keys, got {len(key_list)}: {key_list!r}"
                )

        # Generate values
        if callable(fill_value):
            data = {key: cast(Value, fill_value(key)) for key in key_list}
        else:
            if isinstance(fill_value, IMMUTABLES):
                data = {key: cast(Value, fill_value) for key in key_list}
            else:
                data = {key: cast(Value, copy.deepcopy(fill_value)) for key in key_list}

        return cls(data)
