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

    @classmethod
    def __pre_new__(cls, alias: GenericAlias, *args: Any, **kwargs: Any) -> Self:
        return cls(*args, **kwargs)


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
        origin: type[RuntimeGeneric[Unpack[Ts]]] = get_origin(self)  # type: ignore[valid-type]
        obj = origin.__pre_new__(self, *args, **kwargs)
        return obj


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
        (inner,) = args
        value = _substitute(inner, mapping)

        # If the inner resolved to a tuple (TypeVarTuple binding),
        # return the tuple directly so the parent can expand it.
        if isinstance(value, tuple):
            return value  # pyright: ignore[reportUnknownVariableType]

        return Unpack[value]

    new_args_list: list[Any] = []
    for arg in args:
        val = _substitute(arg, mapping)
        if isinstance(val, tuple):
            new_args_list.extend(val)  # pyright: ignore[reportUnknownArgumentType]
        else:
            new_args_list.append(val)
    new_args = tuple(new_args_list)

    try:
        args = new_args[0] if len(new_args) == 1 else new_args
        return origin[args]
    except TypeError:
        return tp


def _build_mapping(params: tuple[Any, ...], args: tuple[Any, ...]) -> dict[Any, Any]:
    mapping = dict[Any, Any]()
    i: int = 0
    j: int = 0

    while i < len(params):
        p = params[i]

        if isinstance(p, TypeVarTuple):
            # Handle tuple unpacking
            remaining_params = len(params) - i - 1
            remaining_args = len(args) - j
            size = remaining_args - remaining_params
            if size < 0:
                raise TypeError(
                    f"Not enough type arguments to bind TypeVarTuple {p}. "
                    f"Expected at least {len(params)} but got {len(args)}"
                )
            mapping[p] = args[j : j + size]
            j += size
            i += 1
            continue

        if j >= len(args):
            # No argument supplied, try default
            default = getattr(p, "__default__", None)
            if default is not None:
                mapping[p] = default
            else:
                raise TypeError(
                    f"Missing type argument for {p}. Expected {len(params)} args, got {len(args)}"
                )
        else:
            mapping[p] = args[j]
            j += 1

        i += 1

    if j < len(args):
        # Extra arguments leftover
        raise TypeError(
            f"Too many type arguments. Expected {len(params)}, got {len(args)}"
        )

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
    while True:
        orig_bases = get_original_bases(current)  # type: ignore[arg-type]
        for base in orig_bases:
            origin = get_origin(base) or base
            resolved = _substitute(base, mapping)
            resolved_origin = get_origin(resolved) or resolved

            if upto is not None and resolved_origin is upto:
                return get_args(resolved)
            if origin is Generic:
                return _flatten_mapping(mapping)

            parent_params = getattr(origin, "__parameters__", ())
            parent_args = get_args(resolved)
            mapping = _build_mapping(parent_params, parent_args)
            current = origin
            break
        else:
            break

    return args
