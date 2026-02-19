"""
NDArray
=======
"""
# src/typed_numpy/_typed/ndarray.py

# pyright: reportPrivateUsage = false

import builtins
from types import GenericAlias
from typing import (
    Any,
    Iterator,
    Literal,
    NoReturn,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    cast,
    get_args,
    get_origin,
    overload,
)
from typing import Literal as L

import numpy as np
import numpy._typing as npt_
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

_ArrayT_co = TypeVar("_ArrayT_co", bound=np.ndarray, covariant=True)


class _SupportsArray(Protocol[_ArrayT_co]):
    def __array__(self, /) -> _ArrayT_co: ...


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
                # np.dtype[<subclass of np.generic>]
                case t if isinstance(t, type) and issubclass(t, np.generic):
                    return np.dtype(t)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
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


def _validate_shape(expected: _AnyShape, actual: _Shape) -> None:
    """Validate shapes at runtime."""

    # tuple[T, ...]  (variadic shape)
    is_variadic = len(expected) == 2 and expected[1] is Ellipsis

    ## Rank enforcement
    # In the variadic case, rank is not enforced.
    if not is_variadic:
        if len(expected) != len(actual):
            raise RankError(f"Expected {len(expected)} dimensions, got {len(actual)}")

    ## Shape enforcement

    if is_variadic:
        dim_specs = tuple([expected[0]] * len(actual))
    else:
        dim_specs = expected

    bindings = dict[TypeVar, tuple[int, int]]()
    # store: TypeVar -> (first_index, first_value)

    for idx, (exp, act) in enumerate(zip(dim_specs, actual)):
        origin = get_origin(exp)

        # Literal
        if origin is Literal:
            args = get_args(exp)
            if act not in args:
                expected_str = f"{args[0]}" if len(args) == 1 else f"one of {set(args)}"
                raise DimensionError(
                    f"Dimension {idx}: expected {expected_str}, got {act}"
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

        # (Concrete) int
        elif isinstance(exp, int):
            if exp != act:
                raise ShapeError(f"Shape mismatch: expected {expected}, got {actual}")

        # Relaxed dimension
        elif exp is int or exp is Any:
            continue


def _validate_shape_against_contexts(shape_spec: _AnyShape, actual: _Shape) -> None:
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
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        # This method is called when using `TypedNDArray` with generics as in `TypedNDArray[...]`.
        # The arguments can be just a Shape GenericAlias such as `tuple[...]`,
        # which is allowed since `_DTypeT_co` TypeVar defines a default.
        # OR it is a Shape GenericAlias and a DType GenericAlias as `tuple[...], np.dtype[...]`
        # which is passed in to `__class_getitem__` as a `tuple[GenericAlias, GenericAlias]`.
        # Any other case would result in a static error already.

        # We defer the arguments to `_TypedNDArrayGenericAlias` which is a subclass of `GenericAlias`. It has two roles:
        # 1. Support handling further partial binding just as the type system expects.
        #   This is done through `_TypedNDArrayGenericAlias.__getitem__` which just ensures that
        #   `_TypedNDArrayGenericAlias` wraps the `GenericAlias` when partial binding.
        # 2. Transfer the control back to `TypedNDArray` when its `_TypedNDArrayGenericAlias.__call__` method is called,
        #   which should then invoke `TypedNDArray.__new__`.
        #   Additionally we can use the bindings here to perform runtime validation.

        ga = super().__class_getitem__(item)
        return _TypedNDArrayGenericAlias.from_generic_alias(ga)

    # [FIXME]: Can we skip this method? It just uses
    #   `np.asarray(object, dtype=dtype).view(cls)`
    #   all of which are regular `numpy`.
    @overload
    def __new__(  # type: ignore[misc]
        cls,
        object: np._ArrayT,
        dtype: None = None,
        *,
        copy: bool | np._CopyMode | None = True,
        order: np._OrderKACF = "K",
        subok: L[True],
        ndmin: int = 0,
        like: npt_._SupportsArrayFunc | None = None,
    ) -> np._ArrayT: ...
    @overload
    def __new__(  # type: ignore[misc]
        cls,
        object: _SupportsArray[np._ArrayT],
        dtype: None = None,
        *,
        copy: bool | np._CopyMode | None = True,
        order: np._OrderKACF = "K",
        subok: L[True],
        ndmin: L[0] = 0,
        like: npt_._SupportsArrayFunc | None = None,
    ) -> np._ArrayT: ...
    @overload
    def __new__(
        cls,
        object: npt_._ArrayLike[np._ScalarT],
        dtype: None = None,
        *,
        copy: bool | np._CopyMode | None = True,
        order: np._OrderKACF = "K",
        subok: bool = False,
        ndmin: int = 0,
        like: npt_._SupportsArrayFunc | None = None,
    ) -> "TypedNDArray[_ShapeT_co, np.dtype[np._ScalarT]]": ...
    @overload
    def __new__(
        cls,
        object: Any,
        dtype: npt_._DTypeLike[np._ScalarT],
        *,
        copy: bool | np._CopyMode | None = True,
        order: np._OrderKACF = "K",
        subok: bool = False,
        ndmin: int = 0,
        like: npt_._SupportsArrayFunc | None = None,
    ) -> "TypedNDArray[_ShapeT_co, np.dtype[np._ScalarT]]": ...
    @overload
    def __new__(
        cls,
        object: Any,
        dtype: npt.DTypeLike | None = None,
        *,
        copy: bool | np._CopyMode | None = True,
        order: np._OrderKACF = "K",
        subok: bool = False,
        ndmin: int = 0,
        like: npt_._SupportsArrayFunc | None = None,
    ) -> "TypedNDArray[_ShapeT_co, np.dtype[Any]]": ...
    #
    def __new__(  # type: ignore[misc]
        cls,
        object: npt.ArrayLike,
        dtype: npt.DTypeLike | None = None,
        *,
        copy: bool | np._CopyMode | None = True,
        order: np._OrderKACF = "K",
        subok: bool = False,
        ndmin: int = 0,
        like: npt_._SupportsArrayFunc | None = None,
    ) -> Self:
        # Overrides base; This doesn't follow `numpy.ndarray.__new__`,
        # but rather tries to mimick `numpy.array(...)`;

        arr = np.array(
            object, dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like
        )
        # The regular `numpy.ndarray` machinery is put to use here,
        # basically making `TypedNDArray` just a type wrapper around it, just as intended.

        # The `.view(...)` method should be used when subclassing `numpy.ndarray`.
        obj = arr.view(cls)
        obj = cast(Self, obj)  # pyright: ignore[reportUnnecessaryCast]  # pyrefly: ignore [redundant-cast]
        return obj

    def __array_finalize__(self, obj: npt.NDArray[Any] | None, /) -> None:
        if obj is None:
            return

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
        self: "TypedNDArray[tuple[int, *_ShapeRest], _DTypeT_co]",  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
        /,
    ) -> Iterator["TypedNDArray[tuple[*_ShapeRest], _DTypeT_co]"]: ...  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
    @overload  # ?-d
    # Not required, but can keep as a fallback
    def __iter__(self, /) -> Iterator[Any]: ...  # pyright: ignore[reportOverlappingOverload]
    #
    def __iter__(self, /) -> Iterator[Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().__iter__()

    @overload
    def astype(
        self,
        dtype: npt_._DTypeLike[np._ScalarT],
        order: np._OrderKACF = ...,
        casting: np._CastingKind = ...,
        subok: builtins.bool = ...,
        copy: builtins.bool | np._CopyMode = ...,
    ) -> "TypedNDArray[_ShapeT_co, np.dtype[np._ScalarT]]": ...
    @overload
    def astype(
        self,
        dtype: npt_.DTypeLike | None,
        order: np._OrderKACF = ...,
        casting: np._CastingKind = ...,
        subok: builtins.bool = ...,
        copy: builtins.bool | np._CopyMode = ...,
    ) -> "TypedNDArray[_ShapeT_co, np.dtype]": ...
    #
    def astype(
        self,
        dtype: npt_.DTypeLike | None,
        order: np._OrderKACF = "K",
        casting: np._CastingKind = "unsafe",
        subok: builtins.bool = True,
        copy: builtins.bool | np._CopyMode = True,
    ):
        return super().astype(
            dtype=dtype, order=order, casting=casting, subok=subok, copy=copy
        )

    def flatten(
        self, /, order: np._OrderKACF = "C"
    ) -> "TypedNDArray[tuple[int], _DTypeT_co]":
        return super().flatten(order=order)  # type: ignore[return-value]


## Deferred shape binding


class _TypedNDArrayGenericAlias(GenericAlias):
    """
    Deferred TypedNDArray constructor for shapes with TypeVars.
    Enables progressive type specialization, behaving like a type-level curry.
    """

    @classmethod
    def from_generic_alias(cls, alias: GenericAlias) -> Self:
        return cls(alias.__origin__, alias.__args__)  # pyright: ignore[reportArgumentType]

    def __getitem__(self, typeargs: Any) -> Self:
        ga = super().__getitem__(typeargs)
        return type(self).from_generic_alias(ga)

    def __call__(
        self,
        object: npt.ArrayLike,
        dtype: npt.DTypeLike | None = None,
        *,
        copy: bool | np._CopyMode | None = True,
        order: np._OrderKACF = "K",
        subok: bool = False,
        ndmin: int = 0,
        like: npt_._SupportsArrayFunc | None = None,
    ) -> TypedNDArray:
        # [NOTE] Should mimick `TypedNDArray.__new__` signature

        base = cast(type[TypedNDArray], get_origin(self))
        args = get_args(self)

        if len(args) == 2:
            shape_spec, dtype_spec = args
        elif len(args) == 1:
            (shape_spec,) = args
            dtype_spec = GenericAlias(np.dtype, Any)
            # The `dtype_spec` default here should match the default in `_DTypeT_co`.
        else:
            raise TypeError

        if dtype is None:
            dtype = _resolve_dtype(dtype_spec)

        # Create `numpy.ndarray` object
        arr = base(
            object, dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like
        )

        # Runtime shape validation
        shape_args = get_args(shape_spec)
        arr_shape = arr.shape
        _validate_shape(shape_args, arr_shape)
        _validate_shape_against_contexts(shape_args, arr_shape)

        return arr.view(TypedNDArray)
