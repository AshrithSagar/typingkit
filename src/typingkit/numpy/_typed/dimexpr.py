"""
DimExpr
=======
"""
# src/typingkit/numpy/_typed/dimexpr.py

import math
from typing import (
    Any,
    Generic,
    Literal,
    NoReturn,
    TypeAliasType,
    TypeVar,
    get_args,
    get_origin,
    override,
)

Arg1 = TypeVar("Arg1", bound=int)
Arg2 = TypeVar("Arg2", bound=int)


## Base


class DimExpr(int):
    def __new__(cls, *args: Any) -> NoReturn:
        raise TypeError("Shape expressions cannot be instantiated")

    @classmethod
    def expr(cls, args: tuple[Any, ...], /) -> int:
        raise NotImplementedError


class UnaryOp(Generic[Arg1], DimExpr):
    @override
    @classmethod
    def expr(cls, args: tuple[Arg1], /) -> int:
        raise NotImplementedError


class BinaryOp(Generic[Arg1, Arg2], DimExpr):
    @override
    @classmethod
    def expr(cls, args: tuple[Arg1, Arg2], /) -> int:
        raise NotImplementedError


## Core operations


class Neg(UnaryOp[Arg1]):
    @override
    @classmethod
    def expr(cls, args: tuple[Arg1], /) -> int:
        return -args[0]


class Add(BinaryOp[Arg1, Arg2]):
    @override
    @classmethod
    def expr(cls, args: tuple[Arg1, Arg2], /) -> int:
        return args[0] + args[1]


class Mul(BinaryOp[Arg1, Arg2]):
    @override
    @classmethod
    def expr(cls, args: tuple[Arg1, Arg2], /) -> int:
        return args[0] * args[1]


class Pow(BinaryOp[Arg1, Arg2]):
    @override
    @classmethod
    def expr(cls, args: tuple[Arg1, Arg2], /) -> int:
        return args[0] ** args[1]


## Additional operations

# TypeAliases and PEP-695 style TypeAliasTypes,
# don't currently work with the mypy_plugin


class Sub(Add[Arg1, Neg[Arg2]]): ...


class PlusOne(Add[Arg1, Literal[1]]): ...


class MinusOne(Sub[Arg1, Literal[1]]): ...


# Through Subclassing
class Squared(Mul[Arg1, Arg1]): ...


# Custom operations
# Can also directly define with a `expr`, with some custom logic, say
class Cubed(UnaryOp[Arg1]):
    @override
    @classmethod
    def expr(cls, args: tuple[Arg1], /) -> int:
        return args[0] ** 3


class Log(BinaryOp[Arg1, Arg2]):
    @override
    @classmethod
    def expr(cls, args: tuple[Arg1, Arg2], /) -> int:
        x = math.log(args[0], args[1])
        xi = round(x)
        if math.isclose(x, xi, rel_tol=0, abs_tol=1e-12):
            return xi
        raise TypeError("Invalid dimension. Not an integer.")


class Log2(Log[Arg1, Literal[2]]): ...


class Log10(Log[Arg1, Literal[10]]): ...


## Evaluation


def _resolve_dim(tp: Any) -> Any:
    # type[Any] / type[int] / EllipsisType / TypeVar
    if tp is Any or tp is int or tp is Ellipsis or isinstance(tp, TypeVar):
        return tp

    origin = get_origin(tp)

    # Literal[N]
    if origin is Literal:
        # [TODO] How should we handle multiple args case?
        return get_args(tp)[0]

    # TypeAliasType
    if isinstance(origin, TypeAliasType):
        return _resolve_dim(origin.__value__[get_args(tp)])

    # DimExpr
    if origin and issubclass(origin, DimExpr):
        args = get_args(tp)

        # Case 1: class defines its own expr
        if "expr" in origin.__dict__:
            values = tuple(_resolve_dim(arg) for arg in args)
            if not all(isinstance(v, int) for v in values):
                return origin[values]  # pyright: ignore[reportInvalidTypeArguments]
            return origin.expr(values)

        # Case 2: subclassing
        if hasattr(origin, "__orig_bases__"):
            for base in getattr(origin, "__orig_bases__"):
                base_origin = get_origin(base)
                if base_origin and issubclass(base_origin, DimExpr):
                    base_args = get_args(base)

                    parameters = getattr(origin, "__parameters__", ())
                    param_map = dict(zip(parameters, args))
                    substituted = tuple(param_map.get(a, a) for a in base_args)

                    return _resolve_dim(base_origin[substituted])  # pyright: ignore[reportInvalidTypeArguments]

    raise TypeError(f"Cannot evaluate {tp}")
