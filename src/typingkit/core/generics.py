"""
Generics
=======
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

_runtime_typevar_ctx = ContextVar[dict[Any, Any]]("_runtime_typevar_ctx")

Ts = TypeVarTuple("Ts")


class RuntimeGeneric(Generic[Unpack[Ts]]):
    @classmethod
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        item = _resolve_runtime(item)
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
        origin = get_origin(alias)
        args = get_args(alias)
        parameters = getattr(origin, "__parameters__", ())
        mapping = _build_mapping(parameters, args)
        _augment_with_inherited(type(self), mapping)
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
        origin = get_origin(alias)
        typeargs = get_args(alias)
        return cls(origin, typeargs)

    def __getitem__(self, typeargs: Any) -> Self:
        ga = super().__getitem__(typeargs)
        return type(self).from_generic_alias(ga)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        origin = get_origin(self)
        typeargs = get_args(self)
        parameters = getattr(origin, "__parameters__", ())
        mapping = _build_mapping(parameters, typeargs)

        _augment_with_inherited(origin, mapping)

        token = _runtime_typevar_ctx.set(mapping)
        try:
            obj: RuntimeGeneric[Unpack[Ts]] = super().__call__(*args, **kwargs)  # type: ignore[misc, valid-type]
            obj.__runtime_generic_post_init__(self)  # pyright: ignore[reportUnknownMemberType]
        finally:
            _runtime_typevar_ctx.reset(token)

        return obj  # pyright: ignore[reportUnknownVariableType]


## Generics resolution


def _collect_inherited_bindings(cls: type, known: dict[Any, Any]) -> dict[Any, Any]:
    """
    Walk cls MRO, collect bindings from specialised bases not already in `known`.
    Returns new bindings only (does not mutate `known`).
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
            merged = {**known, **extra}  # outer wins
            resolved_args = tuple(_substitute(a, merged) for a in base_args)
            try:
                inherited = _build_mapping(base_params, resolved_args)
            except TypeError:
                continue
            for k, v in inherited.items():
                if k not in known and k not in extra:
                    extra[k] = v
    return extra


def _resolve_runtime(tp: Any) -> Any:
    """Resolves types using runtime context."""

    ctx = _runtime_typevar_ctx.get({})

    if isinstance(tp, TypeVar):
        return ctx.get(tp, tp)

    if isinstance(tp, tuple):
        tp = tuple(_resolve_runtime(arg) for arg in tp)  # pyright: ignore[reportUnknownVariableType]

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)
    resolved = tuple(_resolve_runtime(arg) for arg in args)

    if resolved == args:
        return tp

    try:
        return origin[resolved[0] if len(resolved) == 1 else tuple(resolved)]
    except TypeError:
        return tp


def propagate_runtime(obj: Any, resolved_type: Any) -> None:
    if isinstance(obj, RuntimeGeneric):
        obj.__runtime_generic_post_init__(resolved_type)
    return None


def _augment_with_inherited(cls: Any, mapping: dict[Any, Any]) -> None:
    """
    Walk the full MRO of cls, collecting type bindings from all
    specialised generic bases. Outer / more-derived bindings win; we never overwrite.
    """
    return mapping.update(_collect_inherited_bindings(cls, mapping))


def _substitute(tp: Any, mapping: dict[Any, Any]) -> Any:
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


def _build_mapping(params: tuple[Any, ...], args: tuple[Any, ...]) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    it = iter(args)

    for idx, param in enumerate(params):
        if isinstance(param, TypeVarTuple):
            # Handle tuple unpacking
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
                # No argument supplied, try default
                default = getattr(param, "__default__")
                if default is not NoDefault:
                    mapping[param] = default
                else:
                    raise TypeError(f"Missing type argument for {param}")

    # Check if any leftover arguments remain
    if any(True for _ in it):
        raise TypeError("Too many type arguments")
    return mapping


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
    origin = get_origin(tp) or tp
    args = get_args(tp)

    # Non-class generics (builtins, alias expansions)
    if not hasattr(origin, "__orig_bases__"):
        return args

    parameters: tuple[Any, ...] = getattr(origin, "__parameters__", ())
    mapping = _build_mapping(parameters, args)
    _augment_with_inherited(origin, mapping)

    # If upto is specified, do a full-graph BFS so we don't miss bases
    # that are reachable only via non-first inheritance slots.
    if upto is not None:
        # Queue entries: (class_to_search, mapping_at_that_level)
        queue: deque[tuple[Any, dict[Any, Any]]] = deque([(origin, mapping)])
        visited: set[Any] = set()
        while queue:
            current, cur_mapping = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if not hasattr(current, "__orig_bases__"):
                continue
            for base in get_original_bases(current):
                base_origin = get_origin(base) or base
                resolved = _substitute(base, cur_mapping)
                resolved_origin = get_origin(resolved) or resolved

                if resolved_origin is upto:
                    return get_args(resolved)

                # Build child mapping and enqueue
                parent_params = getattr(base_origin, "__parameters__", ())
                parent_args = get_args(resolved)
                if not parent_params and not parent_args:
                    queue.append((base_origin, cur_mapping))
                    continue
                if not parent_params:
                    # builtin generic or similar — can't map, but enqueue bare
                    queue.append((base_origin, {}))
                    continue
                try:
                    child_mapping = _build_mapping(parent_params, parent_args)
                except TypeError:
                    continue
                queue.append((base_origin, child_mapping))
        # upto not found — fall through to return args
        return args

    # No upto: linear walk to Generic anchor, while being mixin safe
    current = origin
    while hasattr(current, "__orig_bases__"):
        orig_bases = get_original_bases(current)  # type: ignore[arg-type]
        for base in orig_bases:
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
            if not parent_params and not parent_args:
                continue  # skip plain mixins
            if not parent_params:
                continue  # skip builtins with args but no params

            mapping = _build_mapping(parent_params, parent_args)
            current = base_origin
            break
        else:
            break

    return args
