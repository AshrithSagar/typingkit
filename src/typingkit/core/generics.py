"""
Generics
=======
"""
# src/typingkit/core/generics.py

from types import GenericAlias, get_original_bases
from typing import (
    Any,
    Generic,
    Self,
    TypeAliasType,
    TypeVar,
    TypeVarTuple,
    Unpack,
    cast,
    get_args,
    get_origin,
    overload,
)

from typing_extensions import TypeForm

Ts = TypeVarTuple("Ts")


class RuntimeGeneric(Generic[Unpack[Ts]]):
    @classmethod
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        try:
            ga = cast(GenericAlias, super().__class_getitem__(item))  # type: ignore[misc]
        except (AttributeError, TypeError):
            # Fallback if superclass does not implement `__class_getitem__`
            ga = GenericAlias(cls, item)
        return _RuntimeGenericAlias.from_generic_alias(ga)

    def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
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
        obj: RuntimeGeneric[Unpack[Ts]] = super().__call__(*args, **kwargs)  # type: ignore[misc, valid-type]
        obj.__runtime_generic_post_init__(self)  # pyright: ignore[reportUnknownMemberType]
        return obj  # pyright: ignore[reportUnknownVariableType]


## Generics resolution


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
