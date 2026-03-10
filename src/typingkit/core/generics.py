"""
Generics
=======
"""
# src/typingkit/core/generics.py

from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from types import GenericAlias, get_original_bases
from typing import (
    Any,
    Generic,
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

        item = _resolve_runtime_with_inherited(cls, item)
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


def _resolve_runtime_with_inherited(cls: type, item: Any) -> Any:
    """
    Resolves `item` using the current runtime context,
    augmented with any inherited bindings from cls's specialised bases.
    """
    ctx = _runtime_typevar_ctx.get({}).copy()

    # Walk orig_bases of cls to find already-specialised parents
    for base in getattr(cls, "__orig_bases__", ()):
        base_origin = get_origin(base)
        if base_origin is None:
            continue
        base_params = getattr(base_origin, "__parameters__", ())
        base_args = get_args(base)
        if not base_params:
            continue
        # Substitute known ctx into base_args first (handles partial specialisation)
        resolved_base_args = tuple(_substitute(a, ctx) for a in base_args)
        inherited = _build_mapping(base_params, resolved_base_args)
        # Only add bindings not already present (outer wins)
        for k, v in inherited.items():
            if k not in ctx:
                ctx[k] = v

    # Now resolve item with the augmented context
    token = _runtime_typevar_ctx.set(ctx)
    try:
        return _resolve_runtime(item)
    finally:
        _runtime_typevar_ctx.reset(token)


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
    elif isinstance(obj, (list, tuple)):
        args = get_args(resolved_type)
        item_type = args[0] if args else None
        if item_type is not None:
            for item in obj:  # pyright: ignore[reportUnknownVariableType]
                propagate_runtime(item, item_type)
    elif isinstance(obj, dict):
        args = get_args(resolved_type)
        value_type = args[1] if len(args) >= 2 else None
        if value_type is not None:
            for value in obj.values():  # pyright: ignore[reportUnknownVariableType]
                propagate_runtime(value, value_type)
    return None


def _augment_with_inherited(cls: type, mapping: dict[Any, Any]) -> None:
    """
    Walk the full MRO of cls, collecting type bindings from all
    specialised generic bases. Outer / more-derived bindings win; we never overwrite.
    """
    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()):
            base_origin = get_origin(base)
            if base_origin is None:
                continue
            base_params = getattr(base_origin, "__parameters__", ())
            base_args = get_args(base)
            if not base_params or not base_args:
                continue

            # Substitute already-known bindings into base_args
            resolved_args = tuple(_substitute(a, mapping) for a in base_args)
            try:
                inherited = _build_mapping(base_params, resolved_args)
            except TypeError:
                continue
            for k, v in inherited.items():
                if k not in mapping:  # outer wins
                    mapping[k] = v
    return None


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
                default = getattr(param, "__default__", None)
                if default is not None:
                    mapping[param] = default
                else:
                    raise TypeError(f"Missing type argument for {param}")

    # Check if any leftover arguments remain
    if any(True for _ in it):
        raise TypeError("Too many type arguments")
    return mapping


def _flatten_mapping(
    mapping: dict[Any, Any], params: tuple[Any, ...] | None = None
) -> tuple[Any, ...]:
    out: list[Any] = []
    if params is None:
        params = tuple(mapping.keys())
    for param in params:
        value = mapping[param]
        if isinstance(value, tuple):
            out.extend(value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            out.append(value)
    return tuple(out)


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

    current = origin
    while hasattr(current, "__orig_bases__"):
        orig_bases = get_original_bases(current)  # type: ignore[arg-type]
        for base in orig_bases:
            base_origin = get_origin(base) or base
            resolved = _substitute(base, mapping)
            resolved_origin = get_origin(resolved) or resolved

            if upto is not None and resolved_origin is upto:
                return get_args(resolved)
            if base_origin is Generic:
                return _flatten_mapping(mapping)

            parent_params = getattr(base_origin, "__parameters__", ())
            parent_args = get_args(resolved)
            mapping = _build_mapping(parent_params, parent_args)
            current = base_origin
            break
        else:
            break

    return args
