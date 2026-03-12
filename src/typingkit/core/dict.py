"""
TypedDict
=======
A dict subclass with runtime length enforcement.
"""
# src/typingkit/core/dict.py

from dataclasses import dataclass
from types import GenericAlias
from typing import Any, TypeVar, cast

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

    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    @property
    def length(self) -> Length:
        return self.__len__()
