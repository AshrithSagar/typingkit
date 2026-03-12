"""
Generics
========

Public API
----------
RuntimeGeneric               – base class for runtime-inspectable generic types
get_runtime_args             – like typing.get_args, but walks the inheritance graph
get_runtime_mapping          – TypeVar -> concrete-type dict for a specialised type
get_runtime_origin           – like typing.get_origin, but never returns None
is_runtime_specialised       – True when every TypeVar in the type is concretely bound
resolve_runtime_annotation   – substitute TypeVars in an arbitrary annotation
propagate_runtime            – push a resolved alias into a RuntimeGeneric instance
clear_runtime_caches         – invalidate all alias-keyed LRU caches

Construction lifecycle
----------------------
::

    _RuntimeGenericAlias.__call__
        resolves effective RuntimeOptions (class-level merged with scoped)
        if not options.validate -> skip post_init entirely (pure Generic path)
        else:
            sets _runtime_typevar_ctx   (TypeVar -> type mapping)
            sets _runtime_alias_ctx     (full alias, for exotic constructors)
            -> __new__ / __init__
                -> __runtime_generic_post_init__   (guard + validate + propagate)
                    -> __runtime_generic_validate__ (subclass override point)
                    -> __runtime_generic_propagate_children__ (if options.propagate)
            resets both ContextVars
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

from typingkit.core._options import RuntimeOptions, global_default_runtime_options

__all__ = [
    "RuntimeGeneric",
    "get_runtime_args",
    "get_runtime_mapping",
    "get_runtime_origin",
    "is_runtime_specialised",
    "resolve_runtime_annotation",
    "propagate_runtime",
    "clear_runtime_caches",
]

Ts = TypeVarTuple("Ts")

# ── ContextVars ───────────────────────────────────────────────────────────────

_runtime_typevar_ctx: ContextVar[dict[Any, Any]] = ContextVar("_runtime_typevar_ctx")
_runtime_alias_ctx: ContextVar[GenericAlias | None] = ContextVar(
    "_runtime_alias_ctx", default=None
)

_get_typevar_ctx = _runtime_typevar_ctx.get
_set_alias_ctx = _runtime_alias_ctx.set

_EMPTY_CTX: dict[Any, Any] = {}
_SENTINEL: object = object()
_TV_TYPES: frozenset[type] = frozenset({TypeVar, TypeVarTuple})

# Sentinel used by _RuntimeValidatedDescriptor to distinguish "not set" from False.
_NOT_SET: object = object()


# ── _runtime_validated descriptor ────────────────────────────────────────────


class _RuntimeValidatedDescriptor:
    """
    One-shot guard flag stored in the instance's ``__dict__`` under a mangled key.

    Using ``__dict__`` directly avoids slot conflicts with ``dict`` and
    ``np.ndarray`` base classes. Reads return ``False`` until the flag is set.
    """

    __slots__ = ()
    _KEY: str = "__runtime_validated__"

    def __get__(self, obj: Any, objtype: type | None = None) -> bool:
        if obj is None:
            return self  # type: ignore[return-value]
        val = obj.__dict__.get(self._KEY, _NOT_SET)
        return val is not _NOT_SET and bool(val)

    def __set__(self, obj: Any, value: bool) -> None:
        obj.__dict__[self._KEY] = value


# ── Mapping construction ──────────────────────────────────────────────────────


def _build_mapping(
    params: tuple[Any, ...], args: tuple[Any, ...]
) -> tuple[dict[Any, Any], bool]:
    """
    Zip type parameters -> type arguments, handling TypeVarTuple (greedy).

    Returns ``(mapping, has_tvt_value)``; the flag enables the fast
    ``tuple(vals)`` flatten path when no TypeVarTuple is present.
    TypeVar defaults (PEP 696) are filled when args run out.
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
    """Flatten mapping values, expanding TypeVarTuple tuples inline."""
    if not has_tvt_value:
        return tuple(mapping.values())
    return tuple(
        chain.from_iterable(
            v if isinstance(v, tuple) else (v,)
            for v in mapping.values()  # pyright: ignore[reportUnknownArgumentType]
        )
    )


@lru_cache(maxsize=256)
def _collect_inherited_bindings(
    cls: type, known_key: tuple[tuple[Any, Any], ...]
) -> tuple[tuple[Any, Any], ...]:
    """
    Walk the full MRO collecting TypeVar bindings from specialised generic bases
    not already in ``known_key``. Cached by ``(cls, known_key)``.
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
            merged = known | extra
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
    """Mutate ``mapping`` in-place with inherited bindings (more-derived wins)."""
    for k, v in _collect_inherited_bindings(cls, tuple(mapping.items())):
        mapping.setdefault(k, v)


# ── Substitution ──────────────────────────────────────────────────────────────


def _substitute(tp: Any, mapping: dict[Any, Any]) -> Any:
    """Recursively substitute TypeVars in ``tp`` using ``mapping``."""
    if not mapping:
        return tp

    if type(tp) in _TV_TYPES:
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

    if not changed:
        return tp

    try:
        return origin[resolved[0] if len(resolved) == 1 else tuple(resolved)]
    except TypeError:
        return tp


def _resolve_runtime_ctx(tp: Any) -> Any:
    """Resolve ``tp`` against the active construction-time ContextVar mapping."""
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


# ── Graph traversal ───────────────────────────────────────────────────────────


def _bfs_upto(
    origin: Any, mapping: dict[Any, Any], upto: type
) -> tuple[Any, ...] | None:
    """BFS the inheritance graph from ``origin``; return args at ``upto``, or None."""
    queue: deque[tuple[Any, dict[Any, Any]]] = deque([(origin, mapping)])
    visited: set[Any] = set()

    while queue:
        current, cur_mapping = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if getattr(current, "__orig_bases__", None) is None:
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
    """Walk up to the bare Generic[...] anchor and return the flattened args tuple."""
    current = origin
    while True:
        if getattr(current, "__orig_bases__", None) is None:
            break
        for base in get_original_bases(current):  # type: ignore[arg-type]
            base_origin = get_origin(base) or base
            resolved = _substitute(base, mapping)

            if base_origin is Generic:
                return _flatten_mapping_values(mapping, has_tvt_value)

            parent_params = getattr(base_origin, "__parameters__", ())
            if not parent_params:
                continue

            mapping, has_tvt_value = _build_mapping(parent_params, get_args(resolved))
            current = base_origin
            break
        else:
            break

    return _flatten_mapping_values(mapping, has_tvt_value)


# ── Alias-keyed caches ────────────────────────────────────────────────────────


@lru_cache(maxsize=1024)
def _cached_alias_info(
    tp: Any,
) -> tuple[tuple[Any, ...], tuple[tuple[Any, Any], ...]]:
    """
    Cached MRO walk returning ``(runtime_args, mapping_items)`` for ``tp``.
    Shared by ``get_runtime_args`` and ``get_runtime_mapping`` to pay one walk.
    """
    origin = get_origin(tp) or tp
    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping, has_tvt_value = _build_mapping(parameters, get_args(tp))
    _augment_with_inherited(origin, mapping)
    args = _walk_to_generic_anchor(origin, mapping, has_tvt_value)
    return args, tuple(mapping.items())


def mapping_from_alias(alias: Any, cls: Any) -> dict[Any, Any]:
    """
    Build a TypeVar -> type mapping from ``alias``, augmented with inherited
    bindings for ``cls``. Always returns a fresh dict (safe to mutate).
    """
    origin = get_origin(alias) or alias
    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping, _ = _build_mapping(parameters, get_args(alias))
    _augment_with_inherited(cls, mapping)
    return mapping


# ── Options resolution helper ─────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _find_class_options(cls: type) -> RuntimeOptions | None:
    """Walk the MRO and return the first ``_runtime_options_`` found."""
    for klass in cls.__mro__:
        opts = klass.__dict__.get("_runtime_options_")
        if opts is not None:
            return opts
    return None


def _get_class_options(cls: type) -> RuntimeOptions:
    return _find_class_options(cls) or global_default_runtime_options


# ── RuntimeGeneric ────────────────────────────────────────────────────────────


class RuntimeGeneric(Generic[Unpack[Ts]]):
    """
    Base class for generic types needing runtime type-argument introspection.

    Subclass instead of ``Generic`` to gain full TypeVar resolution through the
    inheritance graph, automatic child propagation, and a validation hook.

    Usage::

        T = TypeVar("T")

        class Box(RuntimeGeneric[T]):
            def __init__(self, value: T) -> None:
                self.value = value

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                (t,) = get_runtime_args(alias)
                if not isinstance(self.value, t):
                    raise TypeError(f"Expected {t}, got {type(self.value)}")

        box = Box[int](42)

    Per-class options::

        class Box(RuntimeGeneric[T], options=RuntimeOptions(propagate=False)):
            ...
    """

    _runtime_validated = _RuntimeValidatedDescriptor()
    _runtime_options_: RuntimeOptions = global_default_runtime_options

    def __init_subclass__(
        cls,
        options: RuntimeOptions | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if options is not None:
            cls._runtime_options_ = options

    @classmethod
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        item = _resolve_runtime_ctx(item)
        return _RuntimeGenericAlias(cls, item)

    @classmethod
    def __runtime_generic_pending_alias__(cls) -> GenericAlias | None:
        """
        Consume and return the alias stashed during construction, or ``None``
        if absent or belonging to a different class. Reset-on-read.

        Use in exotic construction hooks (e.g. ``__array_finalize__``)::

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

    def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
        """Subclass validation hook. Override this, not ``__runtime_generic_post_init__``."""
        pass

    def __runtime_generic_iter_children__(
        self, mapping: dict[Any, Any]
    ) -> Iterable[tuple[Any, Any]]:
        """
        Yield ``(value, resolved_annotation)`` pairs for child propagation.

        Default inspects ``__dict__`` (or dataclass fields). Override for
        containers storing children outside ``__dict__`` (e.g. a C buffer).
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
        """
        Called after construction with the specialised alias.

        Owns the ``_runtime_validated`` guard, ``RuntimeOptions`` checks,
        ``__runtime_generic_validate__`` dispatch, and child propagation.
        Only override this for exotic construction paths; normal subclasses
        should override ``__runtime_generic_validate__`` instead.
        """
        if self._runtime_validated:
            return
        self._runtime_validated = True

        opts = RuntimeOptions.resolve(_get_class_options(type(self)))

        if not opts.validate:
            return

        self.__runtime_generic_validate__(alias)

        if opts.propagate:
            self.__runtime_generic_propagate_children__(
                mapping_from_alias(alias, type(self))
            )

    def __runtime_generic_propagate_children__(self, mapping: dict[Any, Any]) -> None:
        """
        Push ``__runtime_generic_post_init__`` into child ``RuntimeGeneric``
        instances from ``__runtime_generic_iter_children__``.

        Pass a pre-built mapping to avoid rebuilding it when calling from a
        custom ``__runtime_generic_post_init__`` override.
        """
        for val, resolved in self.__runtime_generic_iter_children__(mapping):
            if isinstance(val, RuntimeGeneric):
                val.__runtime_generic_post_init__(resolved)


class _RuntimeGenericAlias(GenericAlias):
    """
    Deferred constructor returned by ``__class_getitem__``.

    Each ``[]`` application binds more TypeVars; ``()`` constructs the instance.
    On call: resolves ``RuntimeOptions``, sets ContextVars, constructs, fires
    ``post_init``, then resets. Skips all machinery when ``opts.validate`` is False.
    """

    def __new__(cls, origin: type, args: Any, /) -> Self:
        inst = super().__new__(cls, origin, args)

        # [TODO]: Validate TypeVars specialisation, against bounds/constraints too [?]
        params = getattr(origin, "__parameters__", ())
        args_tuple: tuple[Any, ...] = args if isinstance(args, tuple) else (args,)  # pyright: ignore[reportUnknownVariableType]
        has_tvt = any(isinstance(p, TypeVarTuple) for p in params)
        if not has_tvt and len(args_tuple) > len(params):
            raise TypeError(
                f"Too many arguments for {origin.__name__}: "
                f"got {len(args_tuple)}, expected {len(params)}"
            )
        return inst

    @classmethod
    def from_generic_alias(cls, alias: GenericAlias) -> Self:
        return cls(get_origin(alias), get_args(alias))

    def __getitem__(self, typeargs: Any) -> Self:
        """Support progressive binding: ``MyClass[T][int]`` -> ``MyClass[int]``."""
        return type(self).from_generic_alias(super().__getitem__(typeargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        origin = get_origin(self)

        opts = RuntimeOptions.resolve(_get_class_options(origin))
        if not opts.validate:
            return super().__call__(*args, **kwargs)  # type: ignore[misc]

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


# ── Public API ────────────────────────────────────────────────────────────────


def propagate_runtime(obj: Any, resolved_type: Any) -> None:
    """Fire ``__runtime_generic_post_init__`` on a RuntimeGeneric instance. No-op otherwise."""
    if isinstance(obj, RuntimeGeneric):
        obj.__runtime_generic_post_init__(resolved_type)


def get_runtime_origin(tp: Any) -> Any:
    """Like ``typing.get_origin`` but returns ``tp`` itself when there is no origin."""
    return get_origin(tp) or tp


@overload
def get_runtime_args(
    tp: TypeForm[RuntimeGeneric[Unpack[Ts]]], upto: None = None
) -> tuple[Unpack[Ts]]: ...
@overload
def get_runtime_args(
    tp: TypeForm[Any] | GenericAlias | TypeAliasType, upto: type | None = None
) -> tuple[Any, ...]: ...
def get_runtime_args(
    tp: TypeForm[Any] | GenericAlias | TypeAliasType, upto: type | None = None
) -> tuple[Any, ...]:
    """
    Return resolved type arguments, walking the full inheritance graph.

    Unlike ``typing.get_args``:
    - Bindings set in parent classes are visible.
    - Always returns a full-length tuple, filling PEP 696 TypeVar defaults::

        length, key, value = get_runtime_args(TypedDict[Literal[2]])
        # -> (Literal[2], Any, Any)

    - TypeVarTuple expands inline::

        a, *ts, b = get_runtime_args(MyClass[int, str, float, bool])

    Pass ``upto`` to stop the walk at a specific ancestor.
    The common case (``upto=None``, specialised alias) is LRU-cached.
    """
    origin = get_runtime_origin(tp)

    if not hasattr(origin, "__orig_bases__"):
        return get_args(tp)

    if upto is None:
        try:
            args, _ = _cached_alias_info(tp)  # type: ignore[arg-type]
            return args
        except TypeError:
            pass  # unhashable tp — fall through

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
    Return the full TypeVar -> type mapping for a specialised generic.

    Unbound TypeVars map to themselves. Returns a fresh dict each call.
    Results are LRU-cached per alias.
    """
    origin = get_runtime_origin(tp)

    if not hasattr(origin, "__orig_bases__"):
        return {}

    try:
        _, items = _cached_alias_info(tp)  # type: ignore[arg-type]
        return dict(items)
    except TypeError:
        pass

    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping, _ = _build_mapping(parameters, get_args(tp))
    _augment_with_inherited(origin, mapping)
    return mapping


def is_runtime_specialised(
    tp: TypeForm[Any] | GenericAlias | TypeAliasType,
) -> bool:
    """
    Return True when every TypeVar in ``tp``'s mapping is concretely bound::

        assert is_runtime_specialised(MyList[int])
    """
    origin = get_runtime_origin(tp)
    if not hasattr(origin, "__orig_bases__"):
        return True

    try:
        _, items = _cached_alias_info(tp)  # type: ignore[arg-type]
    except TypeError:
        items = tuple(get_runtime_mapping(tp).items())

    return not any(type(v) in _TV_TYPES for _, v in items)


def resolve_runtime_annotation(
    annotation: Any,
    tp: TypeForm[Any] | GenericAlias | TypeAliasType | RuntimeGeneric[Unpack[Ts]],
) -> Any:
    """
    Substitute TypeVars in ``annotation`` using the bindings of ``tp``.

    ``tp`` may be a specialised alias, a fully-bound subclass, or a
    ``RuntimeGeneric`` instance::

        resolve_runtime_annotation(list[T], Box[int])   # list[int]
        resolve_runtime_annotation(T, Box[int]())        # int
    """
    if isinstance(tp, RuntimeGeneric):
        tp = type(tp)  # type: ignore[assignment]
    return _substitute(annotation, get_runtime_mapping(tp))  # pyright: ignore[reportUnknownArgumentType]


def clear_runtime_caches() -> None:
    """
    Invalidate all alias-keyed LRU caches.

    Not needed in normal usage — useful for test isolation when many distinct
    specialisations are constructed and you want to reclaim memory between runs.
    """
    _cached_alias_info.cache_clear()
    _collect_inherited_bindings.cache_clear()
