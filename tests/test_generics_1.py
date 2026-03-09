"""
Tests for RuntimeGeneric core functionality - 1
"""
# tests/test_generics_1.py

# pyright: reportPrivateUsage = false
# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false
# pyright: reportUnknownVariableType = false

from types import GenericAlias
from typing import Any, TypeVar, TypeVarTuple

import pytest

from typingkit.core.generics import RuntimeGeneric, get_runtime_args

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")
Ts = TypeVarTuple("Ts")


class TestSimpleInstantiation:
    def test_no_type_args(self):
        class Simple(RuntimeGeneric[*Ts]): ...

        Simple()
        args = get_runtime_args(Simple)
        assert args == ()

    def test_single_type_arg(self):
        class Simple(RuntimeGeneric[T]): ...

        Simple[int]()
        args = get_runtime_args(Simple[int])
        assert args == (int,)

    def test_multiple_type_args(self):
        class Simple(RuntimeGeneric[A, B]): ...

        Simple[int, str]()
        args = get_runtime_args(Simple[int, str])
        assert args == (int, str)


class TestTypeVarTuple:
    def test_unpack_empty(self):
        class VarTuple(RuntimeGeneric[*Ts]): ...

        VarTuple()
        args = get_runtime_args(VarTuple)
        assert args == ()

    def test_unpack_multiple(self):
        class VarTuple(RuntimeGeneric[*Ts]): ...

        VarTuple[int, str, float]()
        args = get_runtime_args(VarTuple[int, str, float])
        assert args == (int, str, float)

    def test_combined_fixed_and_tuple(self):
        class Mixed(RuntimeGeneric[int, *Ts, str]): ...

        Mixed[float]()
        Mixed[float, list[int]]()
        args1 = get_runtime_args(Mixed[float])
        args2 = get_runtime_args(Mixed[float, list[int]])
        assert args1 == (int, float, str)
        assert args2 == (int, float, list[int], str)


class TestProgressiveBinding:
    def test_deferred_type_binding(self):
        class Base(RuntimeGeneric[A, *Ts]): ...

        BaseInt = Base[int, *Ts]
        BaseInt[str]()
        args = get_runtime_args(BaseInt[str])
        assert args == (int, str)

    def test_double_binding(self):
        class Base(RuntimeGeneric[A, *Ts]): ...

        B1 = Base[int, *Ts]
        B2 = B1[str, float]
        args = get_runtime_args(B2)
        assert args == (int, str, float)


class TestInheritance:
    def test_single_inheritance(self):
        class Base(RuntimeGeneric[A, B]): ...

        class Child(Base[int, B]): ...

        Child[str]()
        args = get_runtime_args(Child[str])
        assert args == (int, str)

    def test_multi_level_inheritance(self):
        class Base(RuntimeGeneric[A, *Ts]): ...

        class Mid(Base[int, *Ts]): ...

        class Leaf(Mid[float, str]): ...

        args = get_runtime_args(Leaf)
        assert args == (int, float, str)


class TestTypeVarDefaults:
    def test_typevar_default_used(self):
        TDef = TypeVar("TDef", default=int)

        class Box(RuntimeGeneric[TDef]): ...

        # Using default because no type arg provided
        Box()
        args = get_runtime_args(Box)
        assert args == (int,)

        # Explicit type argument overrides default
        Box[str]()
        args = get_runtime_args(Box[str])
        assert args == (str,)

    def test_combined_default_and_explicit(self):
        ADef = TypeVar("ADef", default=float)
        BDef = TypeVar("BDef", default=str)

        class Pair(RuntimeGeneric[ADef, BDef]): ...

        # Both defaults
        Pair()
        args = get_runtime_args(Pair)
        assert args == (float, str)

        # One explicit, one default
        Pair[int]()
        args = get_runtime_args(Pair[int])
        assert args == (int, str)

        # Both explicit
        Pair[int, bool]()
        args = get_runtime_args(Pair[int, bool])
        assert args == (int, bool)

    def test_typevartuple_default_not_supported(self):
        TsDef = TypeVarTuple("TsDef")

        class TupleBox(RuntimeGeneric[*TsDef]): ...

        # TypeVarTuple cannot have a default; missing args just give empty tuple
        TupleBox()
        args = get_runtime_args(TupleBox)
        assert args == ()


class TestEdgeCases:
    def test_no_type_parameters(self):
        class Plain(RuntimeGeneric[*Ts]): ...

        Plain()
        args = get_runtime_args(Plain)
        assert args == ()

    def test_typevar_only_tuple(self):
        class OnlyTuple(RuntimeGeneric[*Ts]): ...

        OnlyTuple[int, str]()
        args = get_runtime_args(OnlyTuple[int, str])
        assert args == (int, str)

    def test_typevar_empty_tuple(self):
        class OnlyTuple(RuntimeGeneric[*Ts]): ...

        OnlyTuple()
        args = get_runtime_args(OnlyTuple)
        assert args == ()


class TestInvalidUsage:
    def test_excess_args(self):
        class Base(RuntimeGeneric[A, B]):
            def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
                _ = get_runtime_args(alias)  # Enforce check
                return None

        with pytest.raises(TypeError):
            Base[int, str, float]()  # Too many type args

    def test_partial_binding(self):
        class Base(RuntimeGeneric[A, B]): ...

        # Partial binding works at runtime
        Base[int, Any]()
        args = get_runtime_args(Base[int, Any])
        assert args == (int, Any)
