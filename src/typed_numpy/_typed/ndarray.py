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
    NoReturn,
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
_Shape: TypeAlias = tuple[int, ...]
_AnyShape: TypeAlias = tuple[Any, ...]

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, default=np.dtype, covariant=True)

_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, default=Any, covariant=True)
_NonObjectScalarT = TypeVar(
    "_NonObjectScalarT",
    bound=np.bool | np.number | np.flexible | np.datetime64 | np.timedelta64,
)


## Exceptions


class ShapeError(Exception):
    """Raised when array shape doesn't match expected shape."""


class RankError(ShapeError):
    """Raised when array rank doesn't match expected dimensions."""


class DimensionError(ShapeError):
    """Raised when some dimension doesn't match expected."""


## Resolution & Validation


def _resolve_dtype(
    dtype_spec: GenericAlias | TypeVar | np.generic,
) -> npt.DTypeLike | None:
    """Resolve dtype at runtime."""

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


def _validate_shape(expected: _Shape, actual: tuple[int, ...]) -> None:
    """Validate shapes at runtime."""

    ## Rank enforcement
    if len(expected) != len(actual):
        raise RankError(f"Expected {len(expected)} dimensions, got {len(actual)}")

    ## Shape enforcement

    bindings = dict[TypeVar, tuple[int, int]]()
    # store: TypeVar -> (first_index, first_value)

    for idx, (exp, act) in enumerate(zip(expected, actual)):
        origin = get_origin(exp)

        # Literal
        if origin is Literal:
            literal_value = get_args(exp)[0]
            if act != literal_value:
                raise DimensionError(
                    f"Dimension {idx}: expected {literal_value}, got {act}"
                )

        # TypeVar
        elif isinstance(exp, TypeVar):
            if exp in bindings:
                prev_index, prev_value = bindings[exp]
                if prev_value != act:
                    raise ShapeError(
                        f"Inconsistent dimensions.\n"
                        f"Found Dimension {prev_index} to be {prev_value} and Dimension {idx} to be {act}"
                        f" but both were constrained to the same TypeVar {exp}."
                    )
            else:
                bindings[exp] = (idx, act)

        # int
        elif isinstance(exp, int):
            if exp != act:
                raise ShapeError(f"Shape mismatch: expected {expected}, got {actual}")

        # Relaxed dimension
        elif exp is int or exp is Any:
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

    @overload  # == 0D --> Not iterable
    def __iter__(self: "TypedNDArray[tuple[()], _DTypeT_co]", /) -> NoReturn: ...
    @overload  # == 1-d & dtype[T \ object_]
    def __iter__(
        self: "TypedNDArray[tuple[int], np.dtype[_NonObjectScalarT]]", /
    ) -> Iterator[_NonObjectScalarT]: ...
    @overload  # == 1-d & StringDType
    def __iter__(
        self: "TypedNDArray[tuple[int], np.dtypes.StringDType]", /
    ) -> Iterator[str]: ...
    @overload  # >= 1D
    # Currently, TypeVarTuple doesn't support bounds/constraints,
    # which we'd want to bound/constrain to `_ShapeT_co` := `tuple[int, ...]`.
    # We can relax `_ShapeT_co` to `tuple[Any, ...]` which would make the following overload type fine,
    # but would relax bounds for each dimension, which is a trade off.
    # So we just allow the following ~unsafely typed overload, which is just complained here
    # and shouldn't affect when using `__iter__` in downstream code, hopefully.
    def __iter__(
        self: "TypedNDArray[tuple[int, *_ShapeRest], _DTypeT_co]",  # type: ignore
        /,
    ) -> Iterator["TypedNDArray[tuple[*_ShapeRest], _DTypeT_co]"]: ...  # type: ignore
    @overload  # ?-d
    # Not required, but can keep as a fallback
    def __iter__(self, /) -> Iterator[Any]: ...  # type: ignore
    #
    def __iter__(self, /) -> Iterator[Any]:  # type: ignore[override]
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
        required_shape_tvars = [
            i
            for i in free_shape_tvars
            if getattr(shape_args[i], "__default__", None) is None
        ]
        dtype_free = isinstance(dtype_inner, TypeVar)
        dtype_required = (
            dtype_free and getattr(dtype_inner, "__default__", None) is None
        )
        min_arity = len(required_shape_tvars) + (1 if dtype_required else 0)
        max_arity = len(free_shape_tvars) + (1 if dtype_free else 0)
        if not (min_arity <= len(item) <= max_arity):
            raise TypeError(
                f"Expected between {min_arity} and {max_arity} type arguments, got {len(item)}"
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

        # Fill remaining unbound shape TypeVars with defaults
        for i in free_shape_tvars[dim_index:]:
            tvar = shape_args[i]
            default = getattr(tvar, "__default__", None)
            if default is not None:
                new_shape[i] = default

        # Fill dtype default
        if dtype_free and getattr(dtype_inner, "__default__", None) is not None:
            new_dtype = getattr(dtype_inner, "__default__")

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
            dtype = _resolve_dtype(self.dtype_spec)

        # Create `numpy.ndarray` object
        arr = self.base(object, dtype=dtype)

        # Runtime shape validation
        _validate_shape(get_args(self.shape_spec), arr.shape)
        _validate_shape_against_contexts(get_args(self.shape_spec), arr.shape)

        return arr

    def __repr__(self) -> str:
        return f"{self.base.__name__}[{self.shape_spec}, {self.dtype_spec}]"
