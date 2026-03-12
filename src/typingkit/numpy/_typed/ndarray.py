"""
NDArray
=======

``TypedNDArray`` is a ``numpy.ndarray`` subclass with static shape + dtype
typing 'and' runtime shape/dtype validation powered by ``RuntimeGeneric``.

Construction lifecycle
----------------------
Because ``np.ndarray`` instances are created via ``np.array(...).view(cls)``
rather than a normal ``__new__`` / ``__init__`` chain, the standard
``_RuntimeGenericAlias.__call__`` path cannot fire
``__runtime_generic_post_init__`` directly.

Instead the bridge is ``__array_finalize__``, which numpy guarantees to call
after 'every' ``.view()`` — including the one triggered during a specialised
``TypedNDArray[...]()`` call::

    _RuntimeGenericAlias.__call__
        sets _runtime_alias_ctx = TypedNDArray[tuple[3], np.dtype[np.int32]]
        -> TypedNDArray.__new__
            -> np.array(...)
            -> arr.view(TypedNDArray)
                -> TypedNDArray.__array_finalize__   ← bridge fires here
                    alias = __runtime_generic_pending_alias__()  # consumes ctx
                    __runtime_generic_post_init__(alias)         # validates
        resets _runtime_alias_ctx (already None — consumed above)

``__runtime_generic_pending_alias__`` resets the ContextVar on first read, so
subsequent ``__array_finalize__`` calls from slicing / further views see
``None`` and skip validation gracefully.

``_TypedNDArrayGenericAlias`` has been removed; ``_RuntimeGenericAlias``
(returned by ``RuntimeGeneric.__class_getitem__``) provides equivalent
deferred-binding behaviour for all ``RuntimeGeneric`` subclasses.
"""
# src/typingkit/numpy/_typed/ndarray.py

# pyright: reportPrivateUsage = false

import builtins
from collections.abc import Iterable
from types import GenericAlias, UnionType
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

from typingkit.core.generics import RuntimeGeneric, get_runtime_args

## ── Typings ──────────────────────────────────────────────────────────────────

_ShapeRest = TypeVarTuple("_ShapeRest")

# ``numpy`` privates
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


## ── Exceptions ───────────────────────────────────────────────────────────────


class ShapeError(Exception):
    """Raised when array shape doesn't match expected shape."""


class RankError(ShapeError):
    """Raised when array rank doesn't match expected dimensions."""


class DimensionError(ShapeError):
    """Raised when some dimension doesn't match expected."""


class DTypeError(Exception):
    """Raised when array dtype doesn't match expected dtype."""


## ── Runtime validation ───────────────────────────────────────────────────────


def _validate_dtype(
    expected: GenericAlias | TypeVar | type[np.dtype], actual: np.dtype
) -> None:
    """
    Validate dtype at runtime.

    ### References:
    - https://numpy.org/doc/stable/reference/arrays.dtypes.html#checking-the-data-type
    """

    # ~TypeVar
    if isinstance(expected, TypeVar):
        # [TODO] Verify bounds, contraints, default
        return None

    # np.dtype[...]
    if isinstance(expected, GenericAlias):
        if get_origin(expected) is np.dtype:
            if not (args := get_args(expected)):
                return None
            assert len(args) == 1
            exp = args[0]

            # np.dtype[Any]
            if exp is Any:
                return None

            # np.dtype[~TypeVar]
            elif isinstance(exp, TypeVar):
                # [TODO] Verify bounds, contraints, default
                return None

            # np.dtype[A | B | ...]
            elif get_origin(exp) is UnionType:
                args = get_args(exp)
                for arg in args:
                    if actual == arg:
                        break
                    if isinstance(arg, TypeVar):
                        return None
                else:
                    raise DTypeError(f"expected {exp}, got {actual}")

            # np.dtype[<subclass of np.generic>]
            else:
                if actual != exp:
                    raise DTypeError(f"expected {exp.__name__}, got {actual}")
        else:
            # [TODO]: Handle typing.Annotated
            raise TypeError(f"Invalid dtype specification. {expected} is not a dtype")

    # <class np.dtype>
    if expected is np.dtype:
        return None

    return None  # Fallback


def _resolve_shape(args: _AnyShape) -> _AnyShape:
    from typingkit.numpy._typed.dimexpr import _resolve_dim

    # [TODO]: Handle TypeAliasType
    return tuple(_resolve_dim(arg) for arg in args)


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
    from typingkit.numpy._typed.context import (
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


## ── Typed NDArray ─────────────────────────────────────────────────────────────


class TypedNDArray(
    RuntimeGeneric[_ShapeT_co, _DTypeT_co],
    np.ndarray[_ShapeT_co, _DTypeT_co],
):
    """
    Generic ``numpy.ndarray`` subclass with static shape typing and runtime
    shape/dtype validation.

    Specialise with a shape tuple and an optional dtype::

        Array3x4   = TypedNDArray[tuple[Literal[3], Literal[4]]]
        IntArray3  = TypedNDArray[tuple[Literal[3]], np.dtype[np.int32]]

    Calling the specialised alias constructs and validates the array::

        arr = Array3x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        # ShapeError raised if shape doesn't match

    TypeVars in the shape are supported for deferred binding::

        N = TypeVar("N")
        ArrayN  = TypedNDArray[tuple[N]]
        Array5  = ArrayN[Literal[5]]   # bind N=5
        arr     = Array5([1, 2, 3, 4, 5])

    Exotic construction bridge
    --------------------------
    ``np.ndarray`` subclasses are constructed via ``np.array(...).view(cls)``,
    which bypasses the normal ``__new__`` / ``__init__`` chain.  The
    ``RuntimeGeneric`` machinery therefore cannot fire
    ``__runtime_generic_post_init__`` through ``_RuntimeGenericAlias.__call__``
    directly.

    The bridge is ``__array_finalize__``: numpy guarantees to call it after
    every ``.view()``, including the one triggered during a specialised
    ``TypedNDArray[...]()`` call.  ``__array_finalize__`` calls
    ``__runtime_generic_pending_alias__()`` to consume the stashed alias from
    ``_runtime_alias_ctx``, then fires ``__runtime_generic_post_init__``
    manually.

    Because ``__runtime_generic_pending_alias__`` resets the ContextVar on
    first read, subsequent ``__array_finalize__`` calls from slicing or
    further views see ``None`` and skip validation silently.
    """

    # ── RuntimeGeneric hooks ──────────────────────────────────────────────────

    def __runtime_generic_iter_children__(
        self, mapping: dict[Any, Any]
    ) -> Iterable[tuple[Any, Any]]:
        # For non-object dtypes, elements are numpy scalars — never RuntimeGeneric.
        # Iterating them would be expensive and always a no-op, so skip early.
        # For object dtype, elements may be arbitrary Python objects including
        # RuntimeGeneric instances, so we do propagate.
        if self.dtype != object:
            yield ((), ())
            return

        # 0-d object arrays hold a single Python object, not iterable via __iter__.
        if self.ndim == 0:
            item_type = mapping.get(_DTypeT_co, Any)  # type: ignore[misc]
            yield self.item(), item_type
            return

        item_type = mapping.get(_DTypeT_co, Any)  # type: ignore[misc]
        for elem in self:
            yield elem, item_type

    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
        """
        Validate shape and dtype against the specialised alias.

        Extracts ``_ShapeT_co`` and ``_DTypeT_co`` from ``alias``, resolves
        any dimension expressions, then delegates to the shape/dtype
        validators.  Calls ``super().__runtime_generic_post_init__`` at the
        end to propagate into any child ``RuntimeGeneric`` instances (no-op
        for plain scalar arrays, but correct for object arrays).

        Parameters
        ----------
        alias:
            The fully-specialised alias, e.g.
            ``TypedNDArray[tuple[Literal[3]], np.dtype[np.int32]]``.
        """
        if getattr(self, "_runtime_validated", False):
            return  # Already validated at construction time, skip re-validation
        self._runtime_validated = True

        args = get_runtime_args(alias)

        # Both type params have defaults, so args may be shorter than 2.
        shape_spec: Any = args[0] if len(args) > 0 else _ShapeT_co.__default__  # type: ignore[misc]
        dtype_spec: Any = args[1] if len(args) > 1 else _DTypeT_co.__default__  # type: ignore[misc]

        # Resolve dimension expressions (e.g. symbolic constants) before
        # passing the shape to the validators.
        shape_args = _resolve_shape(get_args(shape_spec))

        _validate_shape(shape_args, self.shape)
        _validate_shape_against_contexts(shape_args, self.shape)
        _validate_dtype(dtype_spec, self.dtype)

        # Propagate to children (future-proof for object arrays).
        super().__runtime_generic_post_init__(alias)

    # ── numpy construction bridge ─────────────────────────────────────────────

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
    # NOTE: This is prolly the best we can do without HKTs
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
        # NOTE: This is prolly the best we can do without HKTs
        *,
        copy: bool | np._CopyMode | None = True,
        order: np._OrderKACF = "K",
        subok: bool = False,
        ndmin: int = 0,
        like: npt_._SupportsArrayFunc | None = None,
    ) -> Self:
        """
        Construct a ``TypedNDArray`` by delegating to ``numpy.array``.

        Does not follow ``numpy.ndarray.__new__`` directly; instead mimics
        ``numpy.array(...)`` so the full array-creation pipeline (copy,
        order, subok, ndmin) is respected.

        The ``.view(cls)`` call here triggers ``__array_finalize__``, which
        is where the ``RuntimeGeneric`` validation bridge fires when a
        specialised alias is active.
        """
        arr = np.array(
            object, dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like
        )
        obj = arr.view(cls)
        obj = cast(Self, obj)  # pyright: ignore[reportUnnecessaryCast]  # pyrefly: ignore [redundant-cast]
        return obj

    def __array_finalize__(self, obj: npt.NDArray[Any] | None, /) -> None:
        """
        numpy post-construction hook — bridge to ``RuntimeGeneric`` validation.

        Called by numpy after every ``.view()`` operation, including the one
        inside ``TypedNDArray.__new__``.  When a specialised alias is active
        (i.e. the array is being constructed through ``TypedNDArray[...]()``),
        ``__runtime_generic_pending_alias__`` returns and 'consumes' that alias,
        and ``__runtime_generic_post_init__`` fires to validate shape and dtype.

        For all subsequent ``__array_finalize__`` calls (slices, views,
        ufunc outputs, etc.) ``__runtime_generic_pending_alias__`` returns
        ``None`` and this method is a no-op, so no spurious validation occurs.
        """
        if obj is None:
            # Called during ndarray.__new__ for brand-new allocations;
            # no source array exists yet, nothing to validate.
            return

        # Attempt to consume the pending alias set by _RuntimeGenericAlias.__call__.
        # Returns None (and skips validation) for slices, views, and any
        # construction that did not go through a specialised alias.
        alias = self.__runtime_generic_pending_alias__()
        if alias is not None:
            self.__runtime_generic_post_init__(alias)

    # ── numpy API ─────────────────────────────────────────────────────────────

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

    @overload  # type: ignore
    def __getitem__(
        self: "TypedNDArray[tuple[int], np.dtype[_ScalarT_co]]", key: int
    ) -> _ScalarT_co: ...
    @overload
    def __getitem__(
        self: "TypedNDArray[tuple[int, *_ShapeRest], _DTypeT_co]",  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
        key: int,
    ) -> "TypedNDArray[tuple[*_ShapeRest], _DTypeT_co]": ...  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
    @overload
    def __getitem__(
        self: "TypedNDArray[tuple[int, *_ShapeRest], _DTypeT_co]",  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
        key: slice,
    ) -> "TypedNDArray[tuple[int, *_ShapeRest], _DTypeT_co]": ...  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
    @overload
    def __getitem__(
        self: "TypedNDArray[tuple[int, *_ShapeRest], np.dtype[_ScalarT_co]]",  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
        key: tuple[int, *_ShapeRest],
    ) -> _ScalarT_co: ...
    @overload
    def __getitem__(
        self: "TypedNDArray[tuple[int, int, *_ShapeRest], _DTypeT_co]",  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
        key: tuple[int, slice],
    ) -> "TypedNDArray[tuple[int, *_ShapeRest], _DTypeT_co]": ...  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
    @overload
    def __getitem__(
        self: "TypedNDArray[tuple[int, int, int, *_ShapeRest], _DTypeT_co]",  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
        key: tuple[int, int],
    ) -> "TypedNDArray[tuple[int, *_ShapeRest], _DTypeT_co]": ...  # type: ignore[type-var]  # ty: ignore[unused-ignore-comment]
    @overload
    def __getitem__(self, key: Any) -> Any: ...
    #
    def __getitem__(self, key: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().__getitem__(key)  # pyright: ignore[reportUnknownVariableType]
