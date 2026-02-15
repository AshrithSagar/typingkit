"""
NDArray
=======
"""
# src/typed_numpy/_typed/ndarray.py

# pyright: reportPrivateUsage = false

from types import GenericAlias
from typing import (
    Any,
    Iterator,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    get_args,
    get_origin,
    overload,
)

import numpy as np
import numpy.typing as npt

## Typings

_ShapeRest = TypeVarTuple("_ShapeRest")

# `numpy` privates
_Shape: TypeAlias = tuple[Any, ...]  # Weakened type reduction
_AnyShape: TypeAlias = tuple[Any, ...]

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


## Resolution & Validation


def _resolve_dtype_runtime(
    dtype_spec: GenericAlias | TypeVar | np.generic,
) -> npt.DTypeLike | None:
    match dtype_spec:
        # Case 1: np.dtype[T]
        case GenericAlias() as ga if get_origin(ga) is np.dtype:
            args = get_args(ga)
            if not args:
                return None
            inner = args[0]
            match inner:
                # np.dtype[np.generic]
                case np.generic() as t:
                    return t
                # np.dtype[TypeVar]
                case TypeVar():
                    return None
                case _:
                    return None
        # Case 2: bare TypeVar
        case TypeVar():
            return None
        # Case 3: direct scalar type
        case np.generic() as t:
            return t
        case _:
            return None


def _validate_shape_runtime(arr: np.ndarray[_Shape], shape_spec: GenericAlias) -> None:
    shape_args = list(get_args(shape_spec))
    actual_shape = arr.shape

    ## Rank enforcement
    if len(shape_args) != arr.ndim:
        raise RankError(f"Expected {len(shape_args)} dimensions, got {arr.ndim}")

    ## Shape enforcement

    bindings = dict[TypeVar, tuple[int, int]]()
    # store: TypeVar -> (first_index, first_value)

    for idx, (expected, actual) in enumerate(zip(shape_args, actual_shape)):
        origin = get_origin(expected)

        # Literal[N]
        if origin is Literal:
            literal_value = get_args(expected)[0]
            if actual != literal_value:
                raise DimensionError(
                    f"Dimension {idx}: expected {literal_value}, got {actual}"
                )

        # TypeVar
        elif isinstance(expected, TypeVar):
            if expected in bindings:
                prev_index, prev_value = bindings[expected]
                if prev_value != actual:
                    raise ShapeError(
                        f"Inconsistent dimensions.\n"
                        f"Found Dimension {prev_index} to be {prev_value} and Dimension {idx} to be {actual}"
                        f" but both were constrained to the same TypeVar {expected}."
                    )
            else:
                bindings[expected] = (idx, actual)

        # Relaxed dimension
        elif expected is int or expected is Any:
            continue


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

    @classmethod
    def __class_getitem__(
        cls,
        item:
        # Stronger type promotion; numpy.ndarray types it as `Any`;
        # The typing doesn't matter here really, since `GenericAlias` is intended as a runtime construct.
        # We try to include the possible runtime types here.
        GenericAlias | tuple[GenericAlias, GenericAlias],
        /,
    ) -> Any:  # Overrides base; This should actually return a `GenericAlias`, strictly speaking;
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        # This method is called when using `TypedNDArray` with generics as in `TypedNDArray[...]`.
        # The arguments can be just a Shape GenericAlias such as `tuple[...]`,
        # which is allowed since `_DTypeT_co` TypeVar defines a default.
        # OR it is a Shape GenericAlias and a DType GenericAlias as `tuple[...], np.dtype[...]`
        # which is passed in to `__class_getitem__` as a `tuple[GenericAlias, GenericAlias]`.
        # Any other case would result in a static error already.

        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError(f"{cls.__name__}[...] expects (shape, dtype) or shape")
            _shape, _dtype = item
        else:
            _shape, _dtype = item, GenericAlias(np.dtype, (Any,))
            # The `_dtype` default here should match the default in `_DTypeT_co`.

        # We defer the arguments to `_NDShape`. It has two roles:
        # 1. Support handling further partial binding just as the type system expects.
        #   This is done through `_NDShape.__getitem__`.
        # 2. Transfer the control back to `TypedNDArray` when its `_NDShape.__call__` method is called,
        # which should then invoke `TypedNDArray.__new__`.
        # Additionally we may use the bindings to perform runtime validation here.
        return _NDShape(cls, _shape, _dtype)

    # [FIXME]: Can we skip this method? It just uses
    #   `np.asarray(object, dtype=dtype).view(cls)`
    #   all of which are regular `numpy`.
    def __new__(
        cls,
        object: npt.ArrayLike,
        dtype: npt.DTypeLike | None = None,
        # *,
        # [TODO]: `numpy.array(...)` has other optional kwargs as well which are skipped for the time being.
        #   ::{copy, order subok, ndmin, ndmax, like}
    ) -> Self:
        # Overrides base; This doesn't follow `numpy.ndarray.__new__`,
        # but rather tries to mimick `numpy.array(...)`;

        arr = np.asarray(object, dtype=dtype)
        # The regular `numpy.ndarray` machinery is put to use here,
        # basically making `TypedNDArray` just a type wrapper around it, just as intended.

        # The `.view(...)` method should be used when subclassing `numpy.ndarray`.
        obj = arr.view(cls)
        return obj

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

    __slots__ = ("base", "shape_spec", "dtype_spec")

    def __init__(
        self,
        base: type[TypedNDArray],
        shape_spec: GenericAlias,
        dtype_spec: GenericAlias | TypeVar = GenericAlias(np.dtype, Any),
    ):
        self.base = base
        self.shape_spec = shape_spec
        self.dtype_spec = dtype_spec

    def __getitem__(
        self,
        # The typing doesn't matter here really. We try to include the possible runtime types here.
        item: int
        # We type it as `int` here, but in reality it is a `Literal` since
        # the type system only allows that when specifying in (full/partial) `TypedNDArray`.
        | type[int]
        | TypeVar
        | np.dtype
        | np.generic
        | GenericAlias
        | tuple[int | type[int] | TypeVar | np.dtype | np.generic | GenericAlias, ...],
    ) -> "_NDShape":
        """Bind dimensions to unbound TypeVars by position, using defaults for missing ones."""
        if not isinstance(item, tuple):
            item = (item,)

        # Extract current shape and dtype args
        shape_args = list(get_args(self.shape_spec))
        if isinstance(self.dtype_spec, GenericAlias):
            dtype_args = list(get_args(self.dtype_spec))
            dtype_inner = dtype_args[0] if dtype_args else Any
        else:
            dtype_inner = self.dtype_spec

        # Identify free TypeVars in shape and dtype
        free_shape_tvars = [
            i for i, dim in enumerate(shape_args) if isinstance(dim, TypeVar)
        ]
        dtype_free = isinstance(dtype_inner, TypeVar)
        expected_arity = len(free_shape_tvars) + (1 if dtype_free else 0)
        if len(item) != expected_arity:
            raise TypeError(
                f"Expected {expected_arity} type arguments, got {len(item)}"
            )

        # Prepare new shape/dtype bindings
        new_shape = shape_args[:]
        new_dtype = dtype_inner
        dim_index = 0
        for arg in item:
            origin = get_origin(arg)

            # Handle dtype binding
            if (
                (isinstance(arg, type) and issubclass(arg, np.generic))
                or origin is np.dtype
                or (isinstance(arg, TypeVar) and dtype_free)
            ):
                new_dtype = arg
                dtype_free = False
                continue

            # Handle dimension binding
            if dim_index >= len(free_shape_tvars):
                raise TypeError("Too many dimension arguments")
            target_pos = free_shape_tvars[dim_index]
            new_shape[target_pos] = arg
            dim_index += 1

        # Return a new `_NDShape` representing the partially bound type
        return _NDShape(
            self.base,
            GenericAlias(tuple, tuple(new_shape)),
            GenericAlias(np.dtype, (new_dtype,)),
        )

    def __call__(
        self,
        object: npt.ArrayLike,
        dtype: npt.DTypeLike | None = None,
    ) -> TypedNDArray[_AnyShape, np.dtype[Any]]:
        # [NOTE] Should mimick `TypedNDArray.__new__` signature

        if dtype is None:
            dtype = _resolve_dtype_runtime(self.dtype_spec)

        # Create `numpy.ndarray` object
        arr = self.base(object, dtype=dtype)

        # Runtime shape validation
        _validate_shape_runtime(arr, self.shape_spec)
        _validate_shape_against_contexts(get_args(self.shape_spec), arr.shape)

        return arr

    def __repr__(self) -> str:
        return f"{self.base.__name__}[{self.shape_spec}, {self.dtype_spec}]"
