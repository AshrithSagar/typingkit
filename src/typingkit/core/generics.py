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
    if isinstance(tp, TypeVar):
        return mapping.get(tp, tp)

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)
    if not args:
        return tp

    new_args = tuple(_substitute(arg, mapping) for arg in args)
    return origin[new_args]


def get_runtime_args(tp: Any, cls: type) -> tuple[Any, ...]:
    args = get_args(tp)

    parameters: tuple[Any, ...] = getattr(cls, "__parameters__", ())
    mapping = dict(zip(parameters, args))

    current_cls = cls
    while True:
        orig_bases = getattr(current_cls, "__orig_bases__", ())
        found_base = False

        for base in orig_bases:
            origin = get_origin(base)
            if origin is None:
                continue

            base_args = get_args(base)
            resolved = tuple(_substitute(arg, mapping) for arg in base_args)

            if origin is RuntimeGeneric:
                return resolved

            # Propagate mapping
            parent_params = getattr(origin, "__parameters__", ())
            mapping = dict(zip(parent_params, resolved))
            current_cls = origin
            found_base = True
            break

        if not found_base:
            break

    return args
