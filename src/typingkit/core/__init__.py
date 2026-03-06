"""
TypingKit core
=======
"""
# src/typingkit/core/__init__.py

from typingkit.core.dict import TypedDict, TypedDictConfig
from typingkit.core.generics import RuntimeGeneric
from typingkit.core.list import TypedList, TypedListConfig

__all__ = [
    "RuntimeGeneric",
    "TypedList",
    "TypedListConfig",
    "TypedDict",
    "TypedDictConfig",
]
