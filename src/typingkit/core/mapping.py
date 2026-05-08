"""
TypedMapping
=======
A Protocol for length-tracked mappings.
"""
# src/typingkit/core/mapping.py

from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

__all__ = [
    "TypedMapping",
]

Length = TypeVar("Length", bound=int, covariant=True, default=int)
Key = TypeVar("Key", default=Any)
Value = TypeVar("Value", covariant=True, default=Any)


class TypedMapping(Protocol[Length, Key, Value]):
    """
    Read-only `Mapping` protocol with a tracked `Length` type parameter.
    """

    def __getitem__(self, key: Key, /) -> Value: ...
    def __iter__(self) -> Iterator[Key]: ...
    def __len__(self) -> Length: ...

    @property
    def length(self) -> Length: ...
