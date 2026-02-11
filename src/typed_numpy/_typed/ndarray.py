"""
NDArray
=======
"""
# src/typed_numpy/_typed/ndarray.py

from types import GenericAlias
from typing import (
    Any,
    Iterator,
    Literal,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    get_args,
    get_origin,
    overload,
)

import numpy as np
import numpy.typing as npt

## Typed aliases


_AcceptedDim: TypeAlias = int | TypeVar | None
_AcceptedShape: TypeAlias = tuple[_AcceptedDim, ...]
_RuntimeDim: TypeAlias = int | None
_RuntimeShape: TypeAlias = tuple[_RuntimeDim, ...]
_ShapeRest = TypeVarTuple("_ShapeRest")

# `numpy` privates
_Shape: TypeAlias = tuple[Any, ...]  # Weakened type reduction
_AnyShape: TypeAlias = tuple[_AcceptedDim, ...]  # Stronger type promotion

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, default=np.dtype, covariant=True)

_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, default=Any, covariant=True)


## Exceptions


class ShapeError(Exception):
    """Raised when array shape doesn't match expected shape."""


class RankError(ShapeError):
    """Raised when array rank doesn't match expected dimensions."""


class DimensionError(ShapeError):
    """Raised when some dimension doesn't match expected."""


## Shape resolution


def _resolve_dim(dim: _AcceptedDim | type[int]) -> _RuntimeDim:
    """Resolve a dimension specifier into runtime-validatable form."""
    if dim is None or dim is int or type(dim) is TypeVar:
        return None
    elif isinstance(dim, int):
        return dim
    elif get_origin(dim) is Literal:
        if (lit := get_args(dim)) and len(lit) == 1 and isinstance(lit[0], int):
            return lit[0]
    return None  # Fallback


def _resolve_shape(shape: _Shape) -> _RuntimeShape:
    """Resolve each dimension in a shape specifier."""
    return tuple(_resolve_dim(dim) for dim in shape)


def _normalise_shape(item: _AcceptedDim | _AcceptedShape) -> _AcceptedShape:
    """Ensure shape is a tuple."""
    return item if isinstance(item, tuple) else (item,)


## Validation


def _validate_shape(expected: _RuntimeShape, actual: tuple[int, ...]) -> None:
    """Validate concrete shapes at runtime."""
    # Rank enforcement
    if len(expected) != len(actual):
        raise RankError(f"Rank mismatch: expected {len(expected)}, got {len(actual)}")

    # Shape enforcement
    for exp, act in zip(expected, actual):
        if exp is not None and exp != act:
            raise ShapeError(f"Shape mismatch: expected {expected}, got {actual}")


def _validate_shape_against_contexts(
    shape_spec: _Shape, actual: tuple[int, ...]
) -> None:
    """Validate shape against active TypeVar contexts (class-level and method-level)."""
    from typed_numpy._typed.context import (
        _active_class_context,
        _method_typevar_context,
    )

    method_context = _method_typevar_context.get()
    class_context = _active_class_context.get()
    typevar_bindings = dict[TypeVar, int]()
    for idx, dim in enumerate(shape_spec):
        if not isinstance(dim, TypeVar):
            continue
        if idx >= len(actual):
            continue
        actual_dim = actual[idx]
        # Check method_context first before class_context
        expected_dim = method_context.get(dim) or class_context.get(dim)
        if expected_dim is not None and actual_dim != expected_dim:
            raise ShapeError(
                f"TypeVar {dim} mismatch at dimension {idx}: "
                f"expected {expected_dim}, got {actual_dim}"
            )
        # Consistency check
        if dim in typevar_bindings:
            if actual_dim != typevar_bindings[dim]:
                raise ShapeError(
                    f"TypeVar {dim} inconsistent: dimension {idx} is {actual_dim}, "
                    f"but previous occurrence required {typevar_bindings[dim]}"
                )
        else:
            typevar_bindings[dim] = actual_dim


## Typed NDArray
class TypedNDArray(np.ndarray[_ShapeT_co, _DTypeT_co]):
    """Generic `numpy.ndarray` subclass with static shape typing and runtime shape validation."""

    # Metadata slots
    __shape_spec__: _Shape | None = None
    """The original shape specification (may contain TypeVars)."""

    __bound_shape__: _RuntimeShape | None = None
    """Runtime-resolved shape metadata."""

    @classmethod
    def __class_getitem__(
        cls,
        item:
        # Stronger type promotion
        GenericAlias | tuple[GenericAlias, GenericAlias],
        /,
    ) -> Any:  # Overrides base
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        _dtype: Any
        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError(f"{cls.__name__}[...] expects (shape, dtype) or shape")
            _shape, _dtype = item
        else:
            _shape, _dtype = item, Any

        shape_spec: _Shape
        if isinstance(_shape, GenericAlias):
            shape_spec = get_args(_shape)
        elif isinstance(_shape, tuple):
            shape_spec = _shape
        else:
            shape_spec = (_shape,)
        return _NDShape(cls, shape_spec, dtype_spec=_dtype)

    def __new__(
        cls,
        arr: npt.ArrayLike,
        /,
        *,
        dtype: npt.DTypeLike | None = None,
        shape: _ShapeT_co | None = None,
        shape_spec: _Shape | None = None,
    ) -> "TypedNDArray[_ShapeT_co, _DTypeT_co]":
        _arr: np.ndarray[tuple[int, ...]]
        _arr = np.asarray(arr, dtype=dtype)
        obj = _arr.view(cls)

        # Set metadata
        if shape_spec is not None:
            obj.__shape_spec__ = shape_spec
            obj.__bound_shape__ = _resolve_shape(shape_spec)
        elif shape is not None:
            obj.__shape_spec__ = shape if isinstance(shape, tuple) else (shape,)
            obj.__bound_shape__ = _resolve_shape(obj.__shape_spec__)
        else:
            obj.__shape_spec__ = None
            obj.__bound_shape__ = None

        # Runtime validation
        if obj.__bound_shape__ is not None:
            _validate_shape(expected=obj.__bound_shape__, actual=_arr.shape)

        # Validate against active TypeVar contexts if shape_spec exists
        if obj.__shape_spec__ is not None:
            _validate_shape_against_contexts(obj.__shape_spec__, _arr.shape)

        # [NOTE] numpy.ndarray.view should suffice for the return type;
        # Explicit casting would prolly have a redunant call to TypedNDArray.__class_getitem__;
        # So we just use a type: ignore comment, for strict type checkers [mypy --strict];
        return obj  # type: ignore
        # return cast(TypedNDArray[_ShapeT_co, _DTypeT_co], obj)

    def __array_finalize__(self, obj: npt.NDArray[Any] | None, /) -> None:
        if obj is None:
            return

        # Propagate metadata
        # [FIXME] May have downstream side effects
        self.__shape_spec__ = getattr(obj, "__shape_spec__", None)
        self.__bound_shape__ = getattr(obj, "__bound_shape__", None)

    def __repr__(self) -> str:
        return str(np.asarray(self).__repr__())

    @overload  # == 1D
    def __iter__(
        self: "TypedNDArray[tuple[int], np.dtype[_ScalarT_co]]",
    ) -> Iterator[_ScalarT_co]: ...
    @overload  # >= 1D
    def __iter__(
        self: "TypedNDArray[tuple[int, *_ShapeRest], _DTypeT_co]",
    ) -> Iterator["TypedNDArray[tuple[*_ShapeRest], _DTypeT_co]"]: ...
    @overload  # ?-d
    def __iter__(self, /) -> Iterator[Any]: ...
    #
    def __iter__(self, /) -> Iterator[Any]:
        return super().__iter__()


## Deferred shape binding


class _NDShape:
    """
    Deferred TypedNDArray constructor for shapes with TypeVars.
    Enables progressive type specialization, behaving like a type-level curry.
    """

    __slots__ = ("base", "shape_spec", "dtype_spec", "validate")

    def __init__(
        self,
        cls: type[TypedNDArray],
        shape_spec: _AcceptedShape,
        dtype_spec: Any = Any,
        validate: bool = True,
    ):
        self.base = cls
        self.shape_spec = shape_spec
        self.dtype_spec = dtype_spec
        self.validate = validate

    def __getitem__(self, item: _AcceptedDim | _AcceptedShape) -> "_NDShape":
        """Bind dimensions to unbound TypeVars by position, using defaults for missing ones."""
        _item = _normalise_shape(item)
        unbound_typevars = [
            (i, dim)
            for i, dim in enumerate(self.shape_spec)
            if isinstance(dim, TypeVar)
        ]
        if len(_item) > len(unbound_typevars):
            # For a runtime error; Statically should already be caught;
            raise DimensionError(
                f"Too many type arguments: expected at most {len(unbound_typevars)}, got {len(_item)}"
            )

        shape = list(self.shape_spec)
        for (pos, _), dim in zip(unbound_typevars[: len(_item)], _item):
            shape[pos] = dim
        for pos, typevar in unbound_typevars[len(_item) :]:
            default = getattr(typevar, "__default__", None)
            if default is not None:
                shape[pos] = default
        return _NDShape(
            cls=self.base,
            shape_spec=tuple(shape),
            dtype_spec=self.dtype_spec,
            validate=True,
        )

    def __call__(
        self,
        arr: npt.ArrayLike,
        /,
        *,
        dtype: npt.DTypeLike | None = None,
        shape: _ShapeT_co | None = None,
    ) -> TypedNDArray[_AnyShape, np.dtype[Any]]:
        # [NOTE] Should mimick TypedNDArray.__new__ signature
        # [TODO] Resolve any potential side-effects through the provided shape kwarg?
        should_validate = self.validate or not any(
            isinstance(dim, TypeVar) for dim in self.shape_spec
        )
        dtype_spec = (
            get_args(self.dtype_spec)[0] if self.dtype_spec is not Any else None
        )
        return self.base(
            arr,
            dtype=dtype or dtype_spec,
            shape=shape,
            shape_spec=self.shape_spec if should_validate else None,
        )

    def __repr__(self) -> str:
        dims = ", ".join(str(dim) for dim in self.shape_spec)
        return f"{self.base.__name__}[tuple({dims})]"
