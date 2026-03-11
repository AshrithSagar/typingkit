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

# ── ContextVars ───────────────────────────────────────────────────────────────

# Holds the TypeVar -> concrete-type mapping while a specialised instance is
# being constructed.  Set by ``_RuntimeGenericAlias.__call__`` so that nested
# ``__class_getitem__`` calls (triggered by field defaults, etc.) can resolve
# TypeVars that are still in-flight.
_runtime_typevar_ctx: ContextVar[dict[Any, Any]] = ContextVar("_runtime_typevar_ctx")

# Holds the 'full' specialised alias (e.g. ``MyClass[int, str]``) while an
# instance is being constructed.  Unlike ``_runtime_typevar_ctx`` this is
# intentionally readable by subclasses through the public classmethod
# ``RuntimeGeneric.__runtime_generic_pending_alias__``.
#
# This exists specifically to support exotic construction paths (e.g. numpy's
# ``.view()`` + ``__array_finalize__``) where ``_RuntimeGenericAlias.__call__``
# never reaches ``__runtime_generic_post_init__`` directly.  The subclass can
# consume the alias from ``__array_finalize__`` (or equivalent) and fire
# ``__runtime_generic_post_init__`` itself.
#
# The ContextVar is reset to ``None`` the first time it is consumed via
# ``__runtime_generic_pending_alias__``, preventing double-firing when exotic
# hooks are called multiple times (e.g. numpy fires ``__array_finalize__`` on
# every slice and view).
_runtime_alias_ctx: ContextVar[GenericAlias | None] = ContextVar(
    "_runtime_alias_ctx", default=None
)

Ts = TypeVarTuple("Ts")


## ── Internal helpers — type introspection ────────────────────────────────────


def _has_typevars(tp: Any) -> bool:
    """
    Return ``True`` if ``tp`` contains any unbound ``TypeVar`` or
    ``TypeVarTuple`` at any nesting depth.

    Used as a fast-path guard in ``__runtime_generic_post_init__`` to skip
    child propagation when the resolved annotation is already fully concrete.
    Propagating into a fully-concrete child is always a no-op, and skipping
    it avoids traversing large containers (e.g. a dataset of arrays) on every
    construction.

    Examples::

        _has_typevars(int)            # False
        _has_typevars(list[int])      # False
        _has_typevars(T)              # True
        _has_typevars(list[T])        # True
        _has_typevars(tuple[int, T])  # True
    """
    if isinstance(tp, (TypeVar, TypeVarTuple)):
        return True
    return any(_has_typevars(a) for a in get_args(tp))


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

        # Resolve any TypeVars that are already bound in the active construction
        # context (set by ``_RuntimeGenericAlias.__call__``).  This handles
        # cases like ``Box[T]`` appearing inside another specialised generic.
        item = _resolve_runtime_ctx(item)
        return _RuntimeGenericAlias(cls, item)

    # ── Exotic-constructor support ────────────────────────────────────────────

    @classmethod
    def __runtime_generic_pending_alias__(cls) -> GenericAlias | None:
        """
        Consume and return the alias stashed by ``_RuntimeGenericAlias.__call__``,
        but only if it belongs to this class (or a subclass of it).

        Returns ``None`` if no alias is pending, or if the pending alias
        belongs to a 'different' ``RuntimeGeneric`` subclass that happens to
        be constructing at the same call-stack level (e.g. a ``TypedList``
        whose ``__init__`` iterates a ``TypedNDArray``, triggering numpy's
        ``__array_finalize__`` while the ``TypedList`` alias is still active).

        The origin-check guard prevents exotic hooks from accidentally
        consuming a foreign alias and misvalidating the current instance.

        Reset-on-read semantics: the ContextVar is set to ``None`` on first
        successful consumption, so repeated exotic-hook calls (e.g. numpy
        fires ``__array_finalize__`` on every slice and view) see ``None``
        and skip validation cleanly.

        Typical use inside ``__array_finalize__`` (or any equivalent hook)::

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

        # Only consume the alias if it was actually meant for this class.
        origin = get_origin(alias)
        if not (origin is cls or issubclass(origin, cls)):
            return None  # Not ours — leave it for the right consumer
        _runtime_alias_ctx.set(None)
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
        example a container whose elements live in a C buffer rather than as
        Python attributes.

        Parameters
        ----------
        mapping:
            The fully-resolved TypeVar -> concrete-type dict for this instance,
            as returned by ``_mapping_from_alias``.
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
        return None

    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
        """
        Called after the instance has been constructed with a specialised alias.

        Builds the TypeVar -> concrete-type mapping from ``alias``, then walks
        ``__runtime_generic_iter_children__`` to propagate resolved types into
        any child ``RuntimeGeneric`` instances.

        Performance note: child propagation is skipped for any child whose
        resolved annotation contains no remaining ``TypeVar`` or
        ``TypeVarTuple``.  A fully-concrete annotation can never gain new
        bindings from propagation, so skipping it avoids O(N) traversal of
        large containers (e.g. a ``TypedList`` of ``TypedNDArray``s) during
        construction of enclosing ``RuntimeGeneric`` dataclasses.

        Override to add custom validation::

            def __runtime_generic_post_init__(self, alias):
                args = get_runtime_args(alias)
                validate_something(args, self.data)
                super().__runtime_generic_post_init__(alias)  # propagate

        Always call ``super()`` unless you deliberately want to suppress child
        propagation.
        """
        self.__runtime_generic_pre_init__(alias)
        mapping = _mapping_from_alias(alias, type(self))
        for val, resolved in self.__runtime_generic_iter_children__(mapping):
            # Skip propagation when the resolved annotation still contains
            # unbound TypeVars AND the value is not already a RuntimeGeneric
            # instance.  If the value is a RuntimeGeneric, we must always
            # propagate regardless of the annotation — the child needs its own
            # __runtime_generic_post_init__ fired with the resolved alias.
            # If the value is not a RuntimeGeneric (e.g. a plain int, a torch
            # Tensor, a large list of non-generic objects), propagation is
            # always a no-op regardless of annotation, so skip it cheaply.
            if not isinstance(val, RuntimeGeneric):
                continue
            propagate_runtime(val, resolved)
        return None


class _RuntimeGenericAlias(GenericAlias):
    """
    Deferred ``RuntimeGeneric`` constructor.

    Returned by ``RuntimeGeneric.__class_getitem__`` in place of a plain
    ``GenericAlias``.  Behaves like a type-level curry: each ``[]`` application
    binds more TypeVars, and ``()`` finally constructs the instance.

    Construction lifecycle
    ----------------------
    1. Set ``_runtime_typevar_ctx`` so nested ``__class_getitem__`` calls can
       resolve in-flight TypeVars (used by field defaults, etc.).
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

        # Count max capacity: TypeVarTuple is greedy so its presence
        # means the upper bound is unbounded — skip the check.
        has_tvt = any(isinstance(p, TypeVarTuple) for p in params)
        if not has_tvt and len(args_tuple) > len(params):
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
        will 'not' be reached through this path.  Those subclasses should
        instead override their exotic hook (e.g. ``__array_finalize__``) and
        call ``self.__runtime_generic_pending_alias__()`` to retrieve the alias
        stashed in ``_runtime_alias_ctx``.
        """
        origin = get_origin(self)
        mapping = _mapping_from_alias(self, origin)

        typevar_token = _runtime_typevar_ctx.set(mapping)
        alias_token = _runtime_alias_ctx.set(self)  # type: ignore[arg-type]
        try:
            obj: RuntimeGeneric[Unpack[Ts]] = super().__call__(*args, **kwargs)  # type: ignore[misc, valid-type]

            # For normal construction paths the alias is still set here; fire
            # post_init directly.  For exotic paths (numpy etc.) the alias has
            # already been consumed and reset by ``__runtime_generic_pending_alias__``,
            # so this call is a no-op (alias would be None if we checked, but
            # post_init has already fired from the exotic hook).
            obj.__runtime_generic_post_init__(self)  # pyright: ignore[reportUnknownMemberType]
        finally:
            _runtime_typevar_ctx.reset(typevar_token)
            _runtime_alias_ctx.reset(alias_token)

        return obj  # pyright: ignore[reportUnknownVariableType]


## ── Internal helpers — mapping construction ──────────────────────────────────


def _mapping_from_alias(alias: Any, cls: Any) -> dict[Any, Any]:
    """
    Build a fully-augmented TypeVar -> type mapping from a specialised alias,
    then fill in any remaining bindings from cls's inheritance chain.
    """
    origin = get_origin(alias) or alias
    args = get_args(alias)
    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping = _build_mapping(parameters, args)
    _augment_with_inherited(cls, mapping)
    return mapping


def _build_mapping(params: tuple[Any, ...], args: tuple[Any, ...]) -> dict[Any, Any]:
    """
    Zip type parameters -> type arguments.

    Handles plain TypeVars (with optional defaults) and TypeVarTuple (greedy).
    """
    mapping: dict[Any, Any] = {}
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
        else:
            try:
                mapping[param] = next(it)
            except StopIteration:
                default = getattr(param, "__default__", NoDefault)
                if default is not NoDefault:
                    mapping[param] = default
                else:
                    raise TypeError(f"Missing type argument for {param!r}") from None

    if any(True for _ in it):
        raise TypeError("Too many type arguments")
    return mapping


def _collect_inherited_bindings(cls: type, known: dict[Any, Any]) -> dict[Any, Any]:
    """
    Walk cls's full MRO and collect TypeVar bindings from every specialised
    generic base not already present in `known`.
    """
    extra: dict[Any, Any] = {}
    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()):
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
                inherited = _build_mapping(base_params, resolved_args)
            except TypeError:
                continue
            for k, v in inherited.items():
                if k not in known and k not in extra:
                    extra[k] = v
    return extra


def _augment_with_inherited(cls: type, mapping: dict[Any, Any]) -> None:
    """
    Mutate `mapping` in-place, adding all inherited TypeVar bindings
    not already present.  More-derived (outer) bindings always win.
    """
    mapping.update(_collect_inherited_bindings(cls, mapping))


## ── Internal helpers — substitution & runtime context ────────────────────────


def _substitute(tp: Any, mapping: dict[Any, Any]) -> Any:
    """Recursively substitute TypeVars in `tp` using `mapping`."""
    if not mapping:
        return tp

    if isinstance(tp, (TypeVar, TypeVarTuple)):
        return mapping.get(tp, tp)

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)

    if origin is Unpack:
        inner = _substitute(args[0], mapping)
        return inner if isinstance(inner, tuple) else Unpack[inner]  # pyright: ignore[reportUnknownVariableType]

    resolved: list[Any] = []
    for arg in args:
        val = _substitute(arg, mapping)
        resolved.extend(val if isinstance(val, tuple) else (val,))  # pyright: ignore[reportUnknownArgumentType]

    try:
        return origin[resolved[0] if len(resolved) == 1 else tuple(resolved)]
    except TypeError:
        return tp


def _resolve_runtime_ctx(tp: Any) -> Any:
    """
    Resolve `tp` against the active ContextVar mapping set during instantiation.
    """
    ctx = _runtime_typevar_ctx.get({})

    if isinstance(tp, TypeVar):
        return ctx.get(tp, tp)

    if isinstance(tp, tuple):
        return tuple(_resolve_runtime_ctx(arg) for arg in tp)  # pyright: ignore[reportUnknownVariableType]

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)
    resolved = tuple(_resolve_runtime_ctx(arg) for arg in args)

    if resolved == args:
        return tp

    try:
        return origin[resolved[0] if len(resolved) == 1 else tuple(resolved)]
    except TypeError:
        return tp


## ── Internal helpers — graph traversal ───────────────────────────────────────


def _bfs_upto(
    origin: Any, mapping: dict[Any, Any], upto: type
) -> tuple[Any, ...] | None:
    """
    BFS over the inheritance graph starting from `origin` with `mapping`.
    Returns the resolved args at the first node whose origin is `upto`,
    or None if `upto` is not reachable.
    """
    queue: deque[tuple[Any, dict[Any, Any]]] = deque([(origin, mapping)])
    visited: set[Any] = set()

    while queue:
        current, cur_mapping = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if not hasattr(current, "__orig_bases__"):
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
                # Builtin generic (e.g. list[int]) — no param schema, enqueue bare
                queue.append((base_origin, {}))
            else:
                try:
                    child_mapping = _build_mapping(parent_params, parent_args)
                except TypeError:
                    continue
                queue.append((base_origin, child_mapping))

    return None


def _walk_to_generic_anchor(origin: Any, mapping: dict[Any, Any]) -> tuple[Any, ...]:
    """
    Walk up the first-generic-base chain to the bare Generic[...] anchor,
    then flatten mapping values into an args tuple.
    """
    current = origin
    while hasattr(current, "__orig_bases__"):
        for base in get_original_bases(current):  # type: ignore[arg-type]
            base_origin = get_origin(base) or base
            resolved = _substitute(base, mapping)

            if base_origin is Generic:
                return tuple(
                    chain.from_iterable(
                        v if isinstance(v, tuple) else (v,)
                        for v in mapping.values()  # pyright: ignore[reportUnknownArgumentType]
                    )
                )

            parent_params = getattr(base_origin, "__parameters__", ())
            parent_args = get_args(resolved)

            if not parent_params:
                continue  # skip plain mixins and builtins with no param schema

            mapping = _build_mapping(parent_params, parent_args)
            current = base_origin
            break
        else:
            break

    return tuple(
        chain.from_iterable(
            v if isinstance(v, tuple) else (v,)
            for v in mapping.values()  # pyright: ignore[reportUnknownArgumentType]
        )
    )


## ── Public API ───────────────────────────────────────────────────────────────


def propagate_runtime(obj: Any, resolved_type: Any) -> None:
    """
    Push a resolved generic alias into a RuntimeGeneric instance, triggering
    __runtime_generic_post_init__ so the instance can propagate type info to
    its own children.  No-op for non-RuntimeGeneric objects.
    """
    if isinstance(obj, RuntimeGeneric):
        obj.__runtime_generic_post_init__(resolved_type)


def get_runtime_origin(tp: Any) -> Any:
    """
    Like ``typing.get_origin``, but returns ``tp`` itself when there is no
    origin (instead of ``None``), so callers can always treat the result as
    a type without a None-guard.

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
        When given, BFS the inheritance graph and return the args as seen
        from that specific ancestor.  Useful when a class inherits from
        multiple generic bases.
    """
    origin = get_runtime_origin(tp)
    args = get_args(tp)

    if not hasattr(origin, "__orig_bases__"):
        return args  # builtin generics: list[int], dict[str, int], etc.

    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping = _build_mapping(parameters, args)
    _augment_with_inherited(origin, mapping)

    if upto is not None:
        return _bfs_upto(origin, mapping, upto) or args

    return _walk_to_generic_anchor(origin, mapping)


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
    args = get_args(tp)

    if not hasattr(origin, "__orig_bases__"):
        return {}  # builtins have no TypeVar schema to expose

    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping = _build_mapping(parameters, args)
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
    if not mapping:
        # No parameters at all — trivially specialised (e.g. a plain class)
        return True
    return not any(isinstance(v, (TypeVar, TypeVarTuple)) for v in mapping.values())


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
    # Accept instances: resolve against their concrete class alias
    if isinstance(tp, RuntimeGeneric):
        tp = type(tp)  # type: ignore[assignment]

    mapping = get_runtime_mapping(tp)  # pyright: ignore[reportUnknownArgumentType]
    return _substitute(annotation, mapping)


## ─────────────────────────────────────────────────────────────────────────────
