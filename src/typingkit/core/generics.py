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

_runtime_typevar_ctx = ContextVar[dict[Any, Any]]("_runtime_typevar_ctx")

Ts = TypeVarTuple("Ts")


## ── Runtime Generic ──────────────────────────────────────────────────────────


class RuntimeGeneric(Generic[Unpack[Ts]]):
    @classmethod
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        item = _resolve_runtime_ctx(item)
        return _RuntimeGenericAlias(cls, item)

    def __runtime_generic_iter_children__(
        self, mapping: dict[Any, Any]
    ) -> Iterable[tuple[Any, Any]]:
        """
        Yields (value, resolved_annotation) pairs for runtime propagation.
        Override in subclasses that store children outside __dict__.
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

    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
        mapping = _mapping_from_alias(alias, type(self))
        for val, resolved in self.__runtime_generic_iter_children__(mapping):
            propagate_runtime(val, resolved)
        return None


class _RuntimeGenericAlias(GenericAlias):
    """
    Deferred RuntimeGeneric constructor.
    Enables progressive type specialisation, behaving like a type-level curry.
    """

    @classmethod
    def from_generic_alias(cls, alias: GenericAlias) -> Self:
        return cls(get_origin(alias), get_args(alias))

    def __getitem__(self, typeargs: Any) -> Self:
        return type(self).from_generic_alias(super().__getitem__(typeargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        origin = get_origin(self)
        mapping = _mapping_from_alias(self, origin)

        token = _runtime_typevar_ctx.set(mapping)
        try:
            obj: RuntimeGeneric[Unpack[Ts]] = super().__call__(*args, **kwargs)  # type: ignore[misc, valid-type]
            obj.__runtime_generic_post_init__(self)  # pyright: ignore[reportUnknownMemberType]
        finally:
            _runtime_typevar_ctx.reset(token)

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

    Handles:
    - Plain TypeVars (with optional defaults for missing trailing args)
    - TypeVarTuple (greedy consumption, leaving room for subsequent params)
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
    generic base not already present in `known`.  Returns new bindings only;
    does not mutate `known`.
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
            merged = {**known, **extra}  # more-derived bindings win
            resolved_args = tuple(_substitute(a, merged) for a in base_args)
            try:
                inherited = _build_mapping(base_params, resolved_args)
            except TypeError:
                continue
            for k, v in inherited.items():
                if k not in known and k not in extra:
                    extra[k] = v
    return extra


def _augment_with_inherited(cls: Any, mapping: dict[Any, Any]) -> None:
    """
    Mutate `mapping` in-place, adding all inherited TypeVar bindings
    not already present.  More-derived (outer) bindings always win.
    """
    mapping.update(_collect_inherited_bindings(cls, mapping))


## ── Internal helpers — substitution & runtime context ────────────────────────


def _substitute(tp: Any, mapping: dict[Any, Any]) -> Any:
    """
    Recursively substitute TypeVars in `tp` using `mapping`.
    Handles nested generics, TypeVarTuples, and Unpack.
    """
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
    Used internally by __class_getitem__ to evaluate deferred TypeVars.
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
    Linear walk up the first-generic-base chain until the bare Generic[...]
    anchor is reached, then flatten mapping values into an args tuple.
    Skips plain (non-generic) mixins and builtin generics with no params.
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
