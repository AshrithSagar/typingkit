"""
TypedSequence
=======
A Protocol for length-tracked sequences.
"""
# src/typingkit/core/sequence.py

from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

__all__ = [
    "TypedSequence",
]

Length = TypeVar("Length", bound=int, covariant=True, default=int)
Item = TypeVar("Item", covariant=True, default=Any)


class TypedSequence(Protocol[Length, Item]):
    """
    `Sequence` protocol with a tracked `Length` type parameter.
    """

    def __getitem__(self, index: int, /) -> Item: ...
    def __len__(self) -> Length: ...
    def __iter__(self) -> Iterator[Item]: ...
    def __contains__(self, value: object, /) -> bool: ...
    def __reversed__(self) -> Iterator[Item]: ...
    def index(self, value: Any, /) -> int: ...
    def count(self, value: Any, /) -> int: ...

    @property
    def length(self) -> Length: ...
