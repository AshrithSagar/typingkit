"""
TypedDict
=======
A dict subclass with runtime length enforcement.
"""
# src/typingkit/core/dict.py

from types import GenericAlias
from typing import Any, TypeVar, cast

from typingkit.core._config import TypedCollectionConfig
from typingkit.core._validators import LengthError, validate_length
from typingkit.core.generics import RuntimeGeneric, get_runtime_args, mapping_from_alias

__all__ = [
    "TypedDict",
    "LengthError",
]

Length = TypeVar("Length", bound=int, default=int)
Key = TypeVar("Key", default=Any)
Value = TypeVar("Value", default=Any)


class TypedDict(RuntimeGeneric[Length, Key, Value], dict[Key, Value]):
    """
    A ``dict`` subclass with optional runtime length enforcement.

    Usage::

        TypedDict[Literal[2], str, int]({"a": 1, "b": 2})  # length checked
        TypedDict[int, str, int]({"a": 1})                 # unconstrained length
    """

    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
        if getattr(self, "_runtime_validated", False):
            return
        self._runtime_validated = True

        args = get_runtime_args(alias)
        length: Any = args[0] if args else Length.__default__  # type: ignore[misc]

        if TypedCollectionConfig.VALIDATE_LENGTH:
            validate_length(self, length)

        self.__runtime_generic_propagate_children__(
            mapping_from_alias(alias, type(self))
        )

    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    @property
    def length(self) -> Length:
        return self.__len__()
