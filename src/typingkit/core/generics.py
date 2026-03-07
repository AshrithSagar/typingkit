"""
Generics
=======
"""
# src/typingkit/core/generics.py

from types import GenericAlias
from typing import (
    Any,
    Generic,
    Self,
    TypeVar,
    TypeVarTuple,
    Unpack,
    cast,
    get_args,
    get_origin,
)

Ts = TypeVarTuple("Ts")


class RuntimeGeneric(Generic[Unpack[Ts]]):
    @classmethod
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        try:
            ga = cast(GenericAlias, super().__class_getitem__(item))  # type: ignore[misc]
        except:  # noqa: E722
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
    if isinstance(tp, (TypeVar, TypeVarTuple)):
        return mapping.get(tp, tp)

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)
    if not args:
        return tp

    new_args = tuple(_substitute(arg, mapping) for arg in args)

    if origin is Unpack:
        assert len(new_args) == 1
        new_args = new_args[0]
        if isinstance(new_args, tuple):
            return cast(Any, new_args)

    return origin[new_args]


def _resolve_args(args: tuple[Any, ...], mapping: dict[Any, Any]) -> tuple[Any, ...]:
    resolved = list[Any]()
    for arg in args:
        sub = _substitute(arg, mapping)
        if isinstance(sub, tuple):
            resolved.extend(sub)  # pyright: ignore[reportUnknownArgumentType]
        else:
            resolved.append(sub)
    return tuple(resolved)


def _build_mapping(
    parameters: tuple[Any, ...], args: tuple[Any, ...]
) -> dict[Any, Any]:
    mapping = dict[Any, Any]()

    arg_i = int(0)
    for i, param in enumerate(parameters):
        if isinstance(param, TypeVarTuple):
            remaining_params = parameters[i + 1 :]
            remaining_non_variadic = sum(
                not isinstance(p, TypeVarTuple) for p in remaining_params
            )

            ts_len = len(args) - arg_i - remaining_non_variadic
            if ts_len < 0:
                ts_len = 0

            mapping[param] = args[arg_i : arg_i + ts_len]
            arg_i += ts_len
        else:
            if arg_i >= len(args):
                raise TypeError("Not enough type arguments")

            mapping[param] = args[arg_i]
            arg_i += 1

    return mapping


def get_runtime_args(tp: Any, cls: type) -> tuple[Any, ...]:
    args = get_args(tp)

    parameters: tuple[Any, ...] = getattr(cls, "__parameters__", ())
    mapping = _build_mapping(parameters, args)

    current_cls = cls
    while True:
        orig_bases = getattr(current_cls, "__orig_bases__", ())
        found_base = False

        for base in orig_bases:
            origin = get_origin(base)
            if origin is None:
                continue

            base_args = get_args(base)
            resolved = _resolve_args(base_args, mapping)

            # Propagate mapping
            parent_params = getattr(origin, "__parameters__", ())
            mapping = _build_mapping(parent_params, resolved)
            current_cls = origin
            found_base = True
            break

        if not found_base:
            break

    return args
