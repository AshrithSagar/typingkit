"""
Generics
========

Public API
----------
RuntimeGeneric               – base class for runtime-inspectable generic types
get_runtime_args             – like typing.get_args, but walks the inheritance graph
get_runtime_mapping          – TypeVar -> concrete-type dict for a specialised type
get_runtime_origin           – like typing.get_origin, resolves through RuntimeGeneric
is_runtime_specialised       – True when every TypeVar in the type is concretely bound
resolve_runtime_annotation   – substitute TypeVars in an arbitrary annotation
propagate_runtime            – push a resolved alias into a RuntimeGeneric instance

Framework hooks
---------------
The lifecycle of a ``RuntimeGeneric`` instance during specialised construction is::

    _RuntimeGenericAlias.__call__
        sets _runtime_typevar_ctx   (TypeVar -> type mapping, for __class_getitem__)
        sets _runtime_alias_ctx     (the full alias, for exotic constructors)
        -> __new__ / __init__        (normal Python construction path)
            -> __runtime_generic_pre_init__   (hook: fired just before post-init)
            -> __runtime_generic_post_init__  (hook: validate + propagate to children)
        resets both ContextVars

For classes whose construction bypasses ``__call__`` (e.g. ``np.ndarray`` subclasses
that are built via ``.view()`` and ``__array_finalize__``), subclasses should:

1. Override ``__array_finalize__`` (or equivalent exotic hook).
2. Call ``self.__runtime_generic_pending_alias__()`` to consume the stashed alias.
3. Manually invoke ``self.__runtime_generic_post_init__(alias)`` with it.

The ``__runtime_generic_pending_alias__`` classmethod resets the ContextVar on read,
preventing double-firing if the exotic hook is called more than once (e.g. numpy
calls ``__array_finalize__`` on every slice / view).
"""
# src/typingkit/core/generics.py

from collections import deque
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from functools import lru_cache
from itertools import chain
from types import GenericAlias, get_original_bases
from typing import (
    Any,
    Generic,
    NoDefault,
    Self,
    TypeAliasType,
    TypeVar,
    TypeVarTuple,
    Unpack,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from typing_extensions import TypeForm

__all__ = [
    "RuntimeGeneric",
    "get_runtime_args",
    "get_runtime_mapping",
    "get_runtime_origin",
    "is_runtime_specialised",
    "resolve_runtime_annotation",
    "propagate_runtime",
]

Ts = TypeVarTuple("Ts")

# ── ContextVars ───────────────────────────────────────────────────────────────

_runtime_typevar_ctx: ContextVar[dict[Any, Any]] = ContextVar("_runtime_typevar_ctx")
_runtime_alias_ctx: ContextVar[GenericAlias | None] = ContextVar(
    "_runtime_alias_ctx", default=None
)

# Module-level bound methods — avoids per-call attribute lookup on hot paths.
_get_typevar_ctx = _runtime_typevar_ctx.get
_set_alias_ctx = _runtime_alias_ctx.set

# Shared empty dict used as a sentinel/default; never mutated.
_EMPTY_CTX: dict[Any, Any] = {}

_SENTINEL: object = object()

# frozenset membership is ~2x faster than isinstance(x, (TypeVar, TypeVarTuple))
# on the negative (concrete-type) path, which dominates _substitute's inner loop.
_TV_TYPES: frozenset[type] = frozenset({TypeVar, TypeVarTuple})


## ── Internal helpers — mapping construction ──────────────────────────────────


def _build_mapping(
    params: tuple[Any, ...], args: tuple[Any, ...]
) -> tuple[dict[Any, Any], bool]:
    """
    Zip type parameters -> type arguments.

    Handles plain TypeVars (with optional defaults) and TypeVarTuple (greedy).

    Returns ``(mapping, has_tvt_value)`` where ``has_tvt_value`` is ``True``
    when a ``TypeVarTuple`` parameter was bound.  Callers use this flag to
    choose between the fast ``tuple(vals)`` flatten and the slower
    ``chain.from_iterable`` flatten — a ~15x difference in the common case.
    """
    mapping: dict[Any, Any] = {}
    has_tvt_value = False
    it = iter(args)

    for idx, param in enumerate(params):
        if isinstance(param, TypeVarTuple):
            remaining_params = len(params) - idx - 1
            remaining_args = tuple(it)
            size = len(remaining_args) - remaining_params
            if size < 0:
                raise TypeError("Not enough type arguments")
            mapping[param] = remaining_args[:size]
            it = iter(remaining_args[size:])
            has_tvt_value = True
        else:
            val = next(it, _SENTINEL)
            if val is _SENTINEL:
                default = getattr(param, "__default__", NoDefault)
                if default is not NoDefault:
                    mapping[param] = default
                else:
                    raise TypeError(f"Missing type argument for {param!r}")
            else:
                mapping[param] = val

    if next(it, _SENTINEL) is not _SENTINEL:
        raise TypeError("Too many type arguments")

    return mapping, has_tvt_value


def _flatten_mapping_values(
    mapping: dict[Any, Any], has_tvt_value: bool
) -> tuple[Any, ...]:
    """
    Flatten ``mapping.values()`` into a tuple, expanding any ``TypeVarTuple``
    values (stored as a tuple of types) into the output sequence.

    When ``has_tvt_value`` is ``False`` (the common case) every value is a
    single type; ``tuple(mapping.values())`` is used — ~15x faster than the
    ``chain.from_iterable`` path needed for ``TypeVarTuple`` expansions.
    """
    if not has_tvt_value:
        return tuple(mapping.values())
    return tuple(
        chain.from_iterable(
            v if isinstance(v, tuple) else (v,)
            for v in mapping.values()  # pyright: ignore[reportUnknownArgumentType]
        )
    )


@lru_cache(maxsize=256)
def _collect_inherited_bindings_cached(
    cls: type, known_key: tuple[tuple[Any, Any], ...]
) -> tuple[tuple[Any, Any], ...]:
    """
    Walk ``cls``'s full MRO and collect TypeVar bindings from every specialised
    generic base not already present in the caller's mapping.

    Returns a tuple of ``(TypeVar, type)`` pairs (extra bindings only) so the
    result is hashable and cache-friendly.

    ``known_key`` is ``tuple(mapping.items())`` — using items-tuples directly
    (rather than a flat alternating sequence) is ~2x cheaper to construct and
    equally hashable.
    """
    known: dict[Any, Any] = dict(known_key)
    extra: dict[Any, Any] = {}

    for klass in cls.__mro__:
        orig_bases = getattr(klass, "__orig_bases__", None)
        if orig_bases is None:
            continue
        for base in orig_bases:
            base_origin = get_origin(base)
            if base_origin is None:
                continue
            base_params = getattr(base_origin, "__parameters__", ())
            base_args = get_args(base)
            if not base_params or not base_args:
                continue
            merged = known | extra  # more-derived bindings win
            resolved_args = tuple(_substitute(arg, merged) for arg in base_args)
            try:
                inherited, _ = _build_mapping(base_params, resolved_args)
            except TypeError:
                continue
            for k, v in inherited.items():
                if k not in known and k not in extra:
                    extra[k] = v

    return tuple(extra.items())


def _augment_with_inherited(cls: type, mapping: dict[Any, Any]) -> None:
    """
    Mutate ``mapping`` in-place by adding all inherited TypeVar bindings
    not already present.  More-derived (outer) bindings always win.

    Uses a per-(cls, mapping) LRU cache so repeated constructions of the
    same specialised type pay only one MRO walk.
    """
    known_key: tuple[tuple[Any, Any], ...] = tuple(mapping.items())
    for k, v in _collect_inherited_bindings_cached(cls, known_key):
        mapping.setdefault(k, v)


## ── Internal helpers — substitution & runtime context ────────────────────────


def _substitute(tp: Any, mapping: dict[Any, Any]) -> Any:
    """Recursively substitute TypeVars in ``tp`` using ``mapping``."""
    if not mapping:
        return tp

    # frozenset membership on type() is ~2x faster than isinstance on the
    # dominant negative path (plain concrete types like int, str).
    tp_type = type(tp)  # pyright: ignore[reportUnknownVariableType]
    if tp_type in _TV_TYPES:
        return mapping.get(tp, tp)

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)

    if origin is Unpack:
        inner = _substitute(args[0], mapping)
        return inner if isinstance(inner, tuple) else Unpack[inner]  # pyright: ignore[reportUnknownVariableType]

    resolved: list[Any] = []
    changed = False
    for arg in args:
        val = _substitute(arg, mapping)
        if val is not arg:
            changed = True
        if isinstance(val, tuple):
            resolved.extend(val)  # pyright: ignore[reportUnknownArgumentType]
        else:
            resolved.append(val)

    # Skip alias reconstruction when nothing changed — avoids origin[...] overhead.
    if not changed:
        return tp

    try:
        return origin[resolved[0] if len(resolved) == 1 else tuple(resolved)]
    except TypeError:
        return tp


def _resolve_runtime_ctx(tp: Any) -> Any:
    """
    Resolve ``tp`` against the active ContextVar mapping set during instantiation.
    """
    ctx = _get_typevar_ctx(_EMPTY_CTX)

    if isinstance(tp, TypeVar):
        return ctx.get(tp, tp)

    if isinstance(tp, tuple):
        resolved = tuple(_resolve_runtime_ctx(arg) for arg in tp)  # pyright: ignore[reportUnknownVariableType]
        return resolved if resolved != tp else tp  # pyright: ignore[reportUnknownVariableType]

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)
    resolved_args = tuple(_resolve_runtime_ctx(arg) for arg in args)

    if resolved_args == args:
        return tp

    try:
        return origin[resolved_args[0] if len(resolved_args) == 1 else resolved_args]
    except TypeError:
        return tp


## ── Internal helpers — graph traversal ───────────────────────────────────────


def _bfs_upto(
    origin: Any, mapping: dict[Any, Any], upto: type
) -> tuple[Any, ...] | None:
    """
    BFS over the inheritance graph starting from ``origin`` with ``mapping``.

    Returns the resolved args at the first node whose origin is ``upto``,
    or ``None`` if ``upto`` is not reachable.
    """
    queue: deque[tuple[Any, dict[Any, Any]]] = deque([(origin, mapping)])
    visited: set[Any] = set()

    while queue:
        current, cur_mapping = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        orig_bases = getattr(current, "__orig_bases__", None)
        if orig_bases is None:
            continue

        for base in get_original_bases(current):  # type: ignore[arg-type]
            base_origin = get_origin(base) or base
            resolved = _substitute(base, cur_mapping)
            resolved_origin = get_origin(resolved) or resolved

            if resolved_origin is upto:
                return get_args(resolved)

            parent_params = getattr(base_origin, "__parameters__", ())
            parent_args = get_args(resolved)

            if not parent_params and not parent_args:
                queue.append((base_origin, cur_mapping))
            elif not parent_params:
                # Builtin generic (e.g. list[int]) — no param schema, enqueue bare.
                queue.append((base_origin, {}))
            else:
                try:
                    child_mapping, _ = _build_mapping(parent_params, parent_args)
                except TypeError:
                    continue
                queue.append((base_origin, child_mapping))

    return None


def _walk_to_generic_anchor(
    origin: Any, mapping: dict[Any, Any], has_tvt_value: bool
) -> tuple[Any, ...]:
    """
    Walk up the first-generic-base chain to the bare ``Generic[...]`` anchor,
    then flatten mapping values into an args tuple.

    ``has_tvt_value`` is forwarded from ``_build_mapping`` so
    ``_flatten_mapping_values`` can take the fast ``tuple(vals)`` path when
    no ``TypeVarTuple`` expansions are present.
    """
    current = origin
    while True:
        orig_bases = getattr(current, "__orig_bases__", None)
        if orig_bases is None:
            break
        for base in get_original_bases(current):  # type: ignore[arg-type]
            base_origin = get_origin(base) or base
            resolved = _substitute(base, mapping)

            if base_origin is Generic:
                return _flatten_mapping_values(mapping, has_tvt_value)

            parent_params = getattr(base_origin, "__parameters__", ())
            parent_args = get_args(resolved)

            if not parent_params:
                continue  # skip plain mixins and builtins with no param schema

            mapping, has_tvt_value = _build_mapping(parent_params, parent_args)
            current = base_origin
            break
        else:
            break

    return _flatten_mapping_values(mapping, has_tvt_value)


## ── Runtime Generic ──────────────────────────────────────────────────────────


class RuntimeGeneric(Generic[Unpack[Ts]]):
    """
    Base class for generic types that need runtime type-argument introspection.

    Subclass this instead of ``Generic`` to gain:

    - Full TypeVar -> concrete-type resolution through the inheritance graph.
    - Automatic propagation of resolved types to child objects stored as
      attributes (dataclass fields or ``__dict__`` entries).
    - A clean hook pair (``__runtime_generic_pre_init__`` /
      ``__runtime_generic_post_init__``) for custom validation or
      initialisation logic that depends on the concrete type arguments.
    - Support for 'exotic' construction paths (e.g. ``np.ndarray.view()``)
      via ``__runtime_generic_pending_alias__``.

    Typical usage::

        T = TypeVar("T")

        class Box(RuntimeGeneric[T]):
            def __init__(self, value: T) -> None:
                self.value = value

        box = Box[int](42)   # __runtime_generic_post_init__ fires automatically
    """

    @classmethod
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        # Resolve any TypeVars already bound in the active construction context
        # (set by ``_RuntimeGenericAlias.__call__``).  This handles cases like
        # ``Box[T]`` appearing inside another specialised generic.
        item = _resolve_runtime_ctx(item)
        return _RuntimeGenericAlias(cls, item)

    # ── Exotic-constructor support ────────────────────────────────────────────

    @classmethod
    def __runtime_generic_pending_alias__(cls) -> GenericAlias | None:
        """
        Consume and return the alias stashed by ``_RuntimeGenericAlias.__call__``,
        but only if it belongs to this class (or a subclass of it).

        Returns ``None`` if no alias is pending or if the pending alias belongs
        to a different ``RuntimeGeneric`` subclass (e.g. a ``TypedList`` whose
        ``__init__`` iterates a ``TypedNDArray``, triggering numpy's
        ``__array_finalize__`` while the ``TypedList`` alias is still active).

        Reset-on-read: the ContextVar is set to ``None`` on first successful
        consumption, so repeated exotic-hook calls (e.g. numpy fires
        ``__array_finalize__`` on every slice and view) see ``None`` and skip
        cleanly.

        Typical use inside ``__array_finalize__`` (or equivalent hook)::

            def __array_finalize__(self, obj, /):
                if obj is None:
                    return
                alias = self.__runtime_generic_pending_alias__()
                if alias is not None:
                    self.__runtime_generic_post_init__(alias)
        """
        alias = _runtime_alias_ctx.get()
        if alias is None:
            return None
        origin = get_origin(alias)
        if not (origin is cls or issubclass(origin, cls)):
            return None
        _set_alias_ctx(None)
        return alias

    # ── Child-iteration hook ──────────────────────────────────────────────────

    def __runtime_generic_iter_children__(
        self, mapping: dict[Any, Any]
    ) -> Iterable[tuple[Any, Any]]:
        """
        Yield ``(value, resolved_annotation)`` pairs for runtime propagation.

        The default implementation inspects ``__dict__`` (or dataclass fields)
        and resolves each annotated attribute's type using ``mapping``.

        Override in subclasses that store children outside ``__dict__`` — for
        example a container whose elements live in a C buffer.

        Parameters
        ----------
        mapping:
            The fully-resolved TypeVar -> concrete-type dict for this instance,
            as returned by ``mapping_from_alias``.
        """
        if is_dataclass(self):
            hints = get_type_hints(type(self))
            for f in fields(self):
                ann = hints.get(f.name)
                if ann:
                    yield getattr(self, f.name), _substitute(ann, mapping)
        else:
            anns = getattr(type(self), "__annotations__", {})
            for name, val in vars(self).items():
                if name in anns:
                    yield val, _substitute(anns[name], mapping)

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    def __runtime_generic_pre_init__(self, alias: GenericAlias) -> None:
        """
        Called just before ``__runtime_generic_post_init__``.

        Override to perform setup that must happen before children are iterated
        or validated.  Default implementation is a no-op.
        """

    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
        """
        Called after the instance has been constructed with a specialised alias.

        Builds the TypeVar -> concrete-type mapping from ``alias``, then walks
        ``__runtime_generic_iter_children__`` to propagate resolved types into
        any child ``RuntimeGeneric`` instances.

        Performance note: child propagation is skipped for non-``RuntimeGeneric``
        children — plain values (``int``, tensors, etc.) can never gain type
        bindings, so this avoids O(N) traversal of large containers.

        Override to add custom validation::

            def __runtime_generic_post_init__(self, alias):
                args = get_runtime_args(alias)
                validate_something(args, self.data)
                super().__runtime_generic_post_init__(alias)  # propagate

        Always call ``super()`` unless you deliberately want to suppress child
        propagation.
        """
        self.__runtime_generic_pre_init__(alias)
        mapping = mapping_from_alias(alias, type(self))
        self.__runtime_generic_propagate_to_children__(mapping)

    def __runtime_generic_propagate_to_children__(
        self, mapping: dict[Any, Any]
    ) -> None:
        """
        Walk ``__runtime_generic_iter_children__`` and recursively fire
        ``__runtime_generic_post_init__`` on any child ``RuntimeGeneric``
        instances.
        """
        for val, resolved in self.__runtime_generic_iter_children__(mapping):
            if isinstance(val, RuntimeGeneric):
                val.__runtime_generic_post_init__(resolved)


class _RuntimeGenericAlias(GenericAlias):
    """
    Deferred ``RuntimeGeneric`` constructor.

    Returned by ``RuntimeGeneric.__class_getitem__`` in place of a plain
    ``GenericAlias``.  Behaves like a type-level curry: each ``[]`` application
    binds more TypeVars, and ``()`` finally constructs the instance.

    Construction lifecycle
    ----------------------
    1. Set ``_runtime_typevar_ctx`` so nested ``__class_getitem__`` calls can
       resolve in-flight TypeVars.
    2. Set ``_runtime_alias_ctx`` so exotic constructors (e.g. numpy's
       ``__array_finalize__``) can retrieve the alias without going through
       the normal ``__call__`` path.
    3. Call ``super().__call__`` to invoke ``__new__`` / ``__init__``.
    4. Call ``__runtime_generic_post_init__`` on the resulting instance.
    5. Reset both ContextVars via their tokens regardless of exceptions.
    """

    def __new__(cls, origin: type, args: Any, /) -> Self:
        inst = super().__new__(cls, origin, args)

        # [TODO]: Validate TypeVars specialisation, against bounds/constraints too [?]
        params = getattr(origin, "__parameters__", ())
        args_tuple: tuple[Any, ...] = args if isinstance(args, tuple) else (args,)  # pyright: ignore[reportUnknownVariableType]

        # TypeVarTuple is greedy and unbounded — its presence makes the
        # upper-arity check meaningless, so skip it.
        if not any(isinstance(p, TypeVarTuple) for p in params) and len(
            args_tuple
        ) > len(params):
            raise TypeError(
                f"Too many arguments for {origin.__name__}: "
                f"actual {len(args_tuple)}, expected {len(params)}"
            )

        return inst

    @classmethod
    def from_generic_alias(cls, alias: GenericAlias) -> Self:
        """Wrap an existing ``GenericAlias`` in a ``_RuntimeGenericAlias``."""
        return cls(get_origin(alias), get_args(alias))

    def __getitem__(self, typeargs: Any) -> Self:
        """Support progressive binding: ``MyClass[T][int]`` -> ``MyClass[int]``."""
        return type(self).from_generic_alias(super().__getitem__(typeargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Construct a specialised ``RuntimeGeneric`` instance.

        Sets both runtime ContextVars, delegates to ``super().__call__``, then
        fires ``__runtime_generic_post_init__`` on the result.  Both ContextVars
        are reset in a ``finally`` block so they cannot leak across coroutines or
        nested constructions.

        For subclasses with exotic construction paths (e.g. ``np.ndarray``
        subclasses built via ``.view()``), ``__runtime_generic_post_init__``
        will not be reached through this path.  Those subclasses should override
        their exotic hook (e.g. ``__array_finalize__``) and call
        ``self.__runtime_generic_pending_alias__()`` to retrieve the stashed alias.
        """
        origin = get_origin(self)
        mapping = mapping_from_alias(self, origin)

        typevar_token = _runtime_typevar_ctx.set(mapping)
        alias_token = _runtime_alias_ctx.set(self)  # type: ignore[arg-type]
        try:
            obj: RuntimeGeneric[Unpack[Ts]] = super().__call__(*args, **kwargs)  # type: ignore[misc, valid-type]
            obj.__runtime_generic_post_init__(self)  # pyright: ignore[reportUnknownMemberType]
        finally:
            _runtime_typevar_ctx.reset(typevar_token)
            _runtime_alias_ctx.reset(alias_token)

        return obj  # pyright: ignore[reportUnknownVariableType]


## ── Public API ───────────────────────────────────────────────────────────────


def propagate_runtime(obj: Any, resolved_type: Any) -> None:
    """
    Push a resolved generic alias into a RuntimeGeneric instance, triggering
    ``__runtime_generic_post_init__`` so it can propagate type info to its own
    children.  No-op for non-``RuntimeGeneric`` objects.
    """
    if isinstance(obj, RuntimeGeneric):
        obj.__runtime_generic_post_init__(resolved_type)


def mapping_from_alias(alias: Any, cls: Any) -> dict[Any, Any]:
    """
    Build a fully-augmented TypeVar -> type mapping from a specialised alias,
    then fill in any remaining bindings from cls's inheritance chain.
    """
    origin = get_origin(alias) or alias
    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping, _ = _build_mapping(parameters, get_args(alias))
    _augment_with_inherited(cls, mapping)
    return mapping


def get_runtime_origin(tp: Any) -> Any:
    """
    Like ``typing.get_origin``, but returns ``tp`` itself when there is no
    origin (instead of ``None``), so callers never need a ``None``-guard.

    Examples::

        get_runtime_origin(list[int])   # list
        get_runtime_origin(MyClass[T])  # MyClass
        get_runtime_origin(MyClass)     # MyClass  (no origin -> tp itself)
    """
    return get_origin(tp) or tp


@overload
def get_runtime_args(
    tp: TypeForm[RuntimeGeneric[Unpack[Ts]]], upto: None = None
) -> tuple[Unpack[Ts]]: ...
@overload
def get_runtime_args(
    tp: TypeForm[Any] | GenericAlias | TypeAliasType, upto: type | None = None
) -> tuple[Any, ...]: ...
#
def get_runtime_args(
    tp: TypeForm[Any] | GenericAlias | TypeAliasType, upto: type | None = None
) -> tuple[Any, ...]:
    """
    Return the resolved type arguments for a (potentially inherited) generic type.

    Mirrors ``typing.get_args`` but walks the full inheritance graph so that
    concrete bindings set in parent classes are visible::

        class Base(RuntimeGeneric[A, B]): ...
        class Child(Base[int, str]): ...

        get_runtime_args(Child)            # (int, str)
        get_runtime_args(Child, upto=Base) # (int, str)

    Parameters
    ----------
    tp:
        A specialised generic alias (``MyClass[int]``), a fully-bound subclass,
        or any type expression accepted by ``typing.get_args``.
    upto:
        When given, BFS the inheritance graph and return the args as seen from
        that specific ancestor.  Useful when a class inherits from multiple
        generic bases.
    """
    origin = get_runtime_origin(tp)

    if not hasattr(origin, "__orig_bases__"):
        return get_args(tp)

    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping, has_tvt_value = _build_mapping(parameters, get_args(tp))
    _augment_with_inherited(origin, mapping)

    if upto is not None:
        return _bfs_upto(origin, mapping, upto) or get_args(tp)

    return _walk_to_generic_anchor(origin, mapping, has_tvt_value)


def get_runtime_mapping(
    tp: TypeForm[Any] | GenericAlias | TypeAliasType,
) -> dict[Any, Any]:
    """
    Return the full TypeVar -> concrete-type mapping for a specialised generic.

    Unbound TypeVars appear as their own values (i.e. ``{T: T}``), so callers
    can always check whether a binding is concrete with
    ``isinstance(v, TypeVar)``.

    Examples::

        class Pair(RuntimeGeneric[A, B]): ...

        get_runtime_mapping(Pair[int, str])   # {A: int, B: str}
        get_runtime_mapping(Pair[int, B])     # {A: int, B: B}  <- B unbound
        get_runtime_mapping(Pair)             # {}  (no params supplied)
    """
    origin = get_runtime_origin(tp)

    if not hasattr(origin, "__orig_bases__"):
        return {}

    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping, _ = _build_mapping(parameters, get_args(tp))
    _augment_with_inherited(origin, mapping)
    return mapping


def is_runtime_specialised(
    tp: TypeForm[Any] | GenericAlias | TypeAliasType,
) -> bool:
    """
    Return ``True`` when every TypeVar in ``tp``'s mapping is concretely bound
    (i.e. no value in the mapping is itself a TypeVar or TypeVarTuple).

    Useful as a pre-condition check in runtime validators::

        assert is_runtime_specialised(MyList[int]), "Need a concrete element type"
    """
    mapping = get_runtime_mapping(tp)
    # No parameters at all — trivially specialised (e.g. a plain class).
    if not mapping:
        return True
    return not any(type(v) in _TV_TYPES for v in mapping.values())


def resolve_runtime_annotation(
    annotation: Any,
    tp: TypeForm[Any] | GenericAlias | TypeAliasType | RuntimeGeneric[Unpack[Ts]],
) -> Any:
    """
    Substitute TypeVars in ``annotation`` using the bindings of ``tp``.

    ``tp`` may be:
    - A specialised alias: ``MyClass[int]``
    - A fully-bound subclass: ``class Child(Base[int]): ...``
    - A ``RuntimeGeneric`` instance — bindings are taken from its class

    Examples::

        class Box(RuntimeGeneric[T]): ...

        resolve_runtime_annotation(list[T], Box[int])    # list[int]
        resolve_runtime_annotation(T,       Box[int])    # int

        box = Box[int]()
        resolve_runtime_annotation(list[T], box)         # list[int]
    """
    if isinstance(tp, RuntimeGeneric):
        tp = type(tp)  # type: ignore[assignment]

    mapping = get_runtime_mapping(tp)  # pyright: ignore[reportUnknownArgumentType]
    return _substitute(annotation, mapping)


## ─────────────────────────────────────────────────────────────────────────────
