"""
TypedDict
=======
"""
# src/typingkit/core/dict.py

from types import GenericAlias
from typing import Any, TypeVar, cast

from typingkit.core._validators import validate_length
from typingkit.core.generics import RuntimeGeneric, get_runtime_args

## Typings

Length = TypeVar("Length", bound=int, default=int)
Key = TypeVar("Key", default=Any)
Value = TypeVar("Value", default=Any)


## Runtime validation


class TypedDictConfig:
    VALIDATE_LENGTH: bool = True

    @classmethod
    def enable_all(cls):
        cls.VALIDATE_LENGTH = True

    @classmethod
    def disable_all(cls):
        cls.VALIDATE_LENGTH = False


## TypedDict
class TypedDict(RuntimeGeneric[Length, Key, Value], dict[Key, Value]):
    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
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

        if TypedDictConfig.VALIDATE_LENGTH:
            validate_length(self, length)
        return None

    def __len__(self) -> Length:
        return cast(Length, super().__len__())

    @property
    def length(self) -> Length:
        return self.__len__()
