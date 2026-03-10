"""
Runtime validators
=======
"""
# src/typingkit/core/_validators.py

from collections.abc import Sized
from types import NoneType, UnionType
from typing import Any, Literal, TypeVar, get_args, get_origin

__all__ = [
    "LengthError",
    "validate_length",
]


class LengthError(Exception):
    """Raised when an object's length doesn't match the expected length."""


# Sentinel returned by _coerce_length when a length constraint is trivially satisfied.
_UNCONSTRAINED = object()


def _coerce_length(length: Any) -> Any:
    """
    Normalise a length annotation into one of:
      - ``_UNCONSTRAINED``  — no check needed (TypeVar with no default, Any, bare int type)
      - ``int``             — exact required length
      - ``frozenset[int]``  — one of several allowed lengths (from Literal / Union)

    Raises ``LengthError`` for explicitly invalid annotations (e.g. ``NoneType``).
    """
    # Unbound TypeVar — fall back to its default, or treat as unconstrained
    if isinstance(length, TypeVar):
        default = getattr(length, "__default__", None)
        if default is None or default is length:
            return _UNCONSTRAINED
        return _coerce_length(default)

    # Any / bare int supertype — unconstrained
    if length is Any:
        return _UNCONSTRAINED
    if isinstance(length, type):
        if length is NoneType:
            raise LengthError(f"Invalid length annotation: {length!r}")
        if issubclass(length, int):
            return _UNCONSTRAINED  # e.g. Length=int means "any integer length"

    origin = get_origin(length)

    # Literal[a, b, ...] -> frozenset of ints
    if origin is Literal:
        return _coerce_union_args(get_args(length))

    # X | Y | Z  ->  union of coerced results
    if origin is UnionType:
        return _coerce_union_args(get_args(length))

    # Concrete int (e.g. Literal[3] already resolved, or bare 3)
    if isinstance(length, int):
        return length

    return _UNCONSTRAINED  # Unknown annotation shape — Skip


def _coerce_union_args(args: tuple[Any, ...]) -> Any:
    """Collapse a sequence of length annotations into a single ``frozenset`` or scalar."""
    collected = set[int]()
    for arg in args:
        coerced = _coerce_length(arg)
        if coerced is _UNCONSTRAINED:
            return _UNCONSTRAINED  # Any branch being unconstrained -> whole union is
        if isinstance(coerced, frozenset):
            collected |= coerced
        else:
            collected.add(coerced)
    return frozenset(collected) if len(collected) != 1 else next(iter(collected))  # type: ignore[arg-type]


def validate_length(obj: Sized, length: Any) -> None:
    """
    Assert that ``len(obj)`` satisfies the ``length`` annotation.

    ``length`` may be a concrete ``int``, a ``TypeVar``, ``Literal[...]``,
    a ``Union`` of the above, or ``Any``.  Unconstrained annotations (bare
    ``int`` type, unbound ``TypeVar``, ``Any``) are accepted without a check.
    """
    constraint = _coerce_length(length)
    if constraint is _UNCONSTRAINED:
        return None

    actual = len(obj)

    if isinstance(constraint, frozenset):
        if actual not in constraint:
            allowed = ", ".join(str(val) for val in sorted(constraint))  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            raise LengthError(
                f"Length mismatch: expected one of {{{allowed}}}, got {actual}"
            )
    else:
        if actual != constraint:
            raise LengthError(f"Length mismatch: expected {constraint}, got {actual}")

    return None
