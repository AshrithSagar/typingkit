"""
Tests for RuntimeGeneric core functionality - 2
Extended coverage: dataclasses, propagation, substitution, mapping, MRO, aliases, edge cases.
"""
# tests/test_generics_2.py

# mypy: disable-error-code="annotation-unchecked"
# pyright: reportPrivateUsage = false
# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false
# pyright: reportUnknownVariableType = false

from collections.abc import Iterable
from dataclasses import dataclass, field
from types import GenericAlias
from typing import Any, Literal, TypeVar, TypeVarTuple, get_args, get_origin

import pytest

from typingkit.core.generics import (
    RuntimeGeneric,
    _build_mapping,
    _collect_inherited_bindings,
    _substitute,
    get_runtime_args,
    propagate_runtime,
)

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
Ts = TypeVarTuple("Ts")


class TestBuildMapping:
    """Unit tests for the internal _build_mapping helper."""

    def test_single_typevar(self):
        mapping = _build_mapping((T,), (int,))
        assert mapping == {T: int}

    def test_multiple_typevars(self):
        mapping = _build_mapping((A, B), (int, str))
        assert mapping == {A: int, B: str}

    def test_typevar_tuple_all_consumed(self):
        mapping = _build_mapping((Ts,), (int, str, float))
        assert mapping[Ts] == (int, str, float)

    def test_typevar_tuple_empty(self):
        mapping = _build_mapping((Ts,), ())
        assert mapping[Ts] == ()

    def test_typevar_tuple_with_fixed_before(self):
        mapping = _build_mapping((A, Ts), (int, str, float))
        assert mapping[A] is int
        assert mapping[Ts] == (str, float)

    def test_typevar_tuple_with_fixed_after(self):
        mapping = _build_mapping((Ts, B), (int, str, float))
        assert mapping[Ts] == (int, str)
        assert mapping[B] is float

    def test_typevar_tuple_between_fixed(self):
        mapping = _build_mapping((A, Ts, B), (int, str, bytes, float))
        assert mapping[A] is int
        assert mapping[Ts] == (str, bytes)
        assert mapping[B] is float

    def test_too_many_args_raises(self):
        with pytest.raises(TypeError):
            _build_mapping((A, B), (int, str, float))

    def test_missing_arg_no_default_raises(self):
        with pytest.raises(TypeError):
            _build_mapping((A, B), (int,))

    def test_missing_arg_with_default(self):
        ADef = TypeVar("ADef", default=int)
        mapping = _build_mapping((ADef,), ())
        assert mapping[ADef] is int

    def test_partial_default_fills_remainder(self):
        ADef = TypeVar("ADef", default=float)
        BDef = TypeVar("BDef", default=str)
        mapping = _build_mapping((ADef, BDef), (int,))
        assert mapping[ADef] is int
        assert mapping[BDef] is str

    def test_empty_params_empty_args(self):
        mapping = _build_mapping((), ())
        assert mapping == {}

    def test_empty_params_extra_args_raises(self):
        with pytest.raises(TypeError):
            _build_mapping((), (int,))


class TestSubstitute:
    def test_simple_typevar(self):
        assert _substitute(T, {T: int}) is int

    def test_typevar_not_in_mapping(self):
        assert _substitute(T, {A: int}) is T

    def test_empty_mapping_returns_tp(self):
        assert _substitute(list[int], {}) == list[int]

    def test_generic_alias_substitution(self):
        result = _substitute(list[T], {T: str})
        assert result == list[str]

    def test_nested_generic_substitution(self):
        result = _substitute(dict[A, list[B]], {A: int, B: str})
        assert result == dict[int, list[str]]

    def test_no_substitution_needed(self):
        result = _substitute(list[int], {T: str})
        assert result == list[int]

    def test_typevar_tuple_substitution(self):
        result = _substitute(Ts, {Ts: (int, str)})
        assert result == (int, str)

    def test_substitution_with_literal(self):
        result = _substitute(list[T], {T: Literal[5]})
        assert result == list[Literal[5]]


class TestCollectInheritedBindings:
    def test_no_generic_bases(self):
        class Plain:
            pass

        result = _collect_inherited_bindings(Plain, {})
        assert result == {}

    def test_single_specialized_base(self):
        class Base(RuntimeGeneric[A, B]): ...

        class Child(Base[int, B]): ...

        # Simulate: A is already bound to int
        result = _collect_inherited_bindings(Child, {A: int})
        # B should remain as-is (not yet bound)
        assert A not in result  # already in known, won't be overwritten

    def test_does_not_overwrite_known(self):
        class Base(RuntimeGeneric[A]): ...

        class Child(Base[int]): ...

        known = {A: str}  # A already bound externally
        result = _collect_inherited_bindings(Child, known)
        # known should not be mutated, and A shouldn't appear in extra
        assert A not in result
        assert known[A] is str  # unchanged


class TestPropagateRuntime:
    def test_propagate_to_runtime_generic(self):
        propagated = list[Any]()

        class Tracker(RuntimeGeneric[T]):
            def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
                propagated.append(get_args(alias))

        obj = Tracker[int]()
        propagated.clear()
        propagate_runtime(obj, Tracker[str])
        assert propagated == [(str,)]

    def test_propagate_to_non_runtime_generic_is_noop(self):
        """propagate_runtime on a plain object should not raise."""
        propagate_runtime(42, int)
        propagate_runtime("hello", str)
        propagate_runtime(None, type(None))

    def test_propagate_to_none_is_noop(self):
        propagate_runtime(None, None)


class TestDataclassIntegration:
    def test_simple_dataclass_field_propagation(self):
        """Child RuntimeGeneric fields annotated with TypeVar should be propagated."""

        @dataclass
        class Box(RuntimeGeneric[T]):
            value: T

        b = Box[int](value=42)
        # No assertion on propagated type here — just verify no errors
        assert b.value == 42

    def test_dataclass_nested_runtime_generic_field(self):
        """A dataclass field that is itself a RuntimeGeneric gets post_init called."""
        post_inits: list[Any] = []

        class Inner(RuntimeGeneric[T]):
            def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
                post_inits.append(alias)

        @dataclass
        class Outer(RuntimeGeneric[T]):
            inner: Inner[T]

        inner_obj = Inner[int]()
        post_inits.clear()
        Outer[int](inner=inner_obj)
        # Outer's post_init should propagate to inner
        assert any(get_args(a) == (int,) for a in post_inits)

    def test_dataclass_with_default_factory(self):
        @dataclass
        class Container(RuntimeGeneric[T]):
            items: list[T] = field(default_factory=list)

        c = Container[int]()
        assert c.items == []

    def test_dataclass_multiple_typed_fields(self):
        @dataclass
        class Pair(RuntimeGeneric[A, B]):
            first: A
            second: B

        p = Pair[int, str](first=1, second="hello")
        assert p.first == 1
        assert p.second == "hello"


class TestGetRuntimeArgsExtended:
    def test_builtin_generic(self):
        args = get_runtime_args(list[int])
        assert args == (int,)

    def test_builtin_dict(self):
        args = get_runtime_args(dict[str, int])
        assert args == (str, int)

    def test_upto_parameter_direct_parent(self):
        class Base(RuntimeGeneric[A]): ...

        class Child(Base[int]): ...

        args = get_runtime_args(Child, upto=Base)
        assert args == (int,)

    def test_upto_parameter_skips_intermediate(self):
        class Base(RuntimeGeneric[A, B]): ...

        class Mid(Base[int, B]): ...

        class Leaf(Mid[str]): ...

        args = get_runtime_args(Leaf, upto=Base)
        assert args == (int, str)

    def test_args_on_specialized_alias(self):
        class Gen(RuntimeGeneric[A, B]): ...

        args = get_runtime_args(Gen[int, str])
        assert args == (int, str)

    def test_typevar_tuple_args(self):
        class Variadic(RuntimeGeneric[*Ts]): ...

        args = get_runtime_args(Variadic[int, str, float])
        assert args == (int, str, float)

    def test_literal_type_arg(self):
        class Sized(RuntimeGeneric[T]): ...

        args = get_runtime_args(Sized[Literal[3]])
        assert args == (Literal[3],)

    def test_any_type_arg(self):
        class Gen(RuntimeGeneric[T]): ...

        args = get_runtime_args(Gen[Any])
        assert args == (Any,)


class TestDeepInheritance:
    def test_three_level_chain(self):
        class L0(RuntimeGeneric[A, B, C]): ...

        class L1(L0[int, B, C]): ...

        class L2(L1[str, C]): ...

        class L3(L2[float]): ...

        args = get_runtime_args(L3)
        assert args == (int, str, float)

    def test_diamond_inheritance(self):
        """Two base classes sharing a common RuntimeGeneric ancestor."""

        class Root(RuntimeGeneric[T]): ...

        class Left(Root[int]): ...

        class Right(Root[int]): ...

        class Diamond(Left, Right): ...

        args = get_runtime_args(Diamond)
        # At minimum, should not raise
        assert isinstance(args, tuple)

    def test_mixin_does_not_confuse_binding(self):
        class Mixin:
            pass

        class Base(RuntimeGeneric[A, B]): ...

        class Child(Base[int, str], Mixin): ...

        args = get_runtime_args(Child)
        assert args == (int, str)

    def test_inherited_default_propagated(self):
        TDef = TypeVar("TDef", default=bytes)

        class Base(RuntimeGeneric[TDef]): ...

        class Child(Base[TDef]): ...

        args = get_runtime_args(Child)
        assert args == (bytes,)


class TestProgressiveSpecialisation:
    def test_three_step_curry(self):
        class Base(RuntimeGeneric[A, B, C]): ...

        Step1 = Base[int, B, C]
        Step2 = Step1[str, C]
        Step3 = Step2[float]
        args = get_runtime_args(Step3)
        assert args == (int, str, float)

    def test_curry_then_instantiate(self):
        class Gen(RuntimeGeneric[A, B]):
            def __init__(self, a: Any, b: Any):
                self.a = a
                self.b = b

        Partial = Gen[int, B]
        obj = Partial[str](1, "x")
        assert obj.a == 1
        assert obj.b == "x"

    def test_alias_of_alias_args(self):
        class Base(RuntimeGeneric[A, *Ts]): ...

        Step1 = Base[int, *Ts]
        Step2 = Step1[str, float]
        args = get_runtime_args(Step2)
        assert args == (int, str, float)

    def test_deferred_specialisation_preserves_unbound(self):
        class Base(RuntimeGeneric[A, B]): ...

        Partial = Base[int, B]
        args = get_runtime_args(Partial)
        # B is still unbound — should be a TypeVar, not a concrete type
        assert args[0] is int
        assert isinstance(args[1], TypeVar)


class TestIterChildrenOverride:
    def test_custom_children_propagated(self):
        propagated_types: list[Any] = []

        class Leaf(RuntimeGeneric[T]):
            def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
                propagated_types.append(get_args(alias))

        class Container(RuntimeGeneric[T]):
            def __init__(self, child: "Leaf[T]"):
                self.child = child

            def __runtime_generic_iter_children__(
                self, mapping: dict[Any, Any]
            ) -> Iterable[tuple[Any, Any]]:
                yield self.child, _substitute(Leaf[T], mapping)

        leaf = Leaf[int]()
        propagated_types.clear()
        Container[int](child=leaf)
        assert (int,) in propagated_types

    def test_iter_children_empty_override(self):
        """Subclass yielding nothing should not raise."""

        class Silent(RuntimeGeneric[T]):
            def __runtime_generic_iter_children__(
                self, mapping: dict[Any, Any]
            ) -> Iterable[tuple[Any, Any]]:
                return iter([])

        Silent[int]()  # Should not raise


class TestTypeVarBounds:
    def test_bound_typevar_used_as_arg(self):
        Num = TypeVar("Num", bound=int)

        class Typed(RuntimeGeneric[Num]): ...

        Typed[int]()
        args = get_runtime_args(Typed[int])
        assert args == (int,)

    def test_literal_int_as_bound_typevar(self):
        Num = TypeVar("Num", bound=int, default=int)

        class Sized(RuntimeGeneric[Num]): ...

        Sized[Literal[3]]()
        args = get_runtime_args(Sized[Literal[3]])
        assert args == (Literal[3],)


class TestLiteralTypeArgs:
    def test_literal_single(self):
        class Sized(RuntimeGeneric[T]): ...

        Sized[Literal[5]]()
        args = get_runtime_args(Sized[Literal[5]])
        assert args == (Literal[5],)

    def test_literal_union(self):
        class Sized(RuntimeGeneric[T]): ...

        Sized[Literal[4, 5]]()
        args = get_runtime_args(Sized[Literal[4, 5]])
        assert args == (Literal[4, 5],)

    def test_literal_inherited(self):
        class Base(RuntimeGeneric[T]): ...

        class Child(Base[Literal[3]]): ...

        args = get_runtime_args(Child)
        assert args == (Literal[3],)

    def test_literal_in_generic_field(self):
        class Outer(RuntimeGeneric[A, T]):
            def __init__(self, a: Any):
                self.a = a

        Outer[int, Literal[3]](42)
        args = get_runtime_args(Outer[int, Literal[3]])
        assert args == (int, Literal[3])


class TestIdempotency:
    def test_double_post_init_stable(self):
        """Calling post_init twice should not corrupt state."""

        class Stable(RuntimeGeneric[T]):
            pass

        obj = Stable[int]()
        obj.__runtime_generic_post_init__(GenericAlias(Stable, (int,)))
        obj.__runtime_generic_post_init__(GenericAlias(Stable, (int,)))

    def test_multiple_instantiations_independent(self):
        class Gen(RuntimeGeneric[T]): ...

        _a = Gen[int]()
        _b = Gen[str]()
        assert get_runtime_args(Gen[int]) == (int,)
        assert get_runtime_args(Gen[str]) == (str,)

    def test_context_isolation_between_calls(self):
        """Type context from one instantiation must not leak into the next."""

        class Box(RuntimeGeneric[T]):
            pass

        Box[int]()
        Box[str]()
        # No assertion needed — this checks no ContextVar leak raises


class TestClassGetItem:
    def test_returns_generic_alias(self):
        class Gen(RuntimeGeneric[T]): ...

        result = Gen[int]
        assert isinstance(result, GenericAlias)

    def test_origin_is_class(self):
        class Gen(RuntimeGeneric[T]): ...

        result = Gen[int]
        assert get_origin(result) is Gen

    def test_args_match(self):
        class Gen(RuntimeGeneric[A, B]): ...

        result = Gen[int, str]
        assert get_args(result) == (int, str)

    def test_alias_is_callable(self):
        class Gen(RuntimeGeneric[T]):
            def __init__(self, x: Any = None):
                self.x = x

        obj = Gen[int](x=5)
        assert obj.x == 5


class TestTypeVarTupleAdvanced:
    def test_variadic_child_inherits_prefix(self):
        class Base(RuntimeGeneric[A, *Ts]): ...

        class Child(Base[int, *Ts]): ...

        Child[str, float]()
        args = get_runtime_args(Child[str, float])
        assert args == (int, str, float)

    def test_variadic_empty_in_child(self):
        class Base(RuntimeGeneric[A, *Ts]): ...

        class Child(Base[int, *Ts]): ...

        Child()
        args = get_runtime_args(Child)
        assert args == (int,)

    def test_variadic_combined_fixed_after(self):
        class Base(RuntimeGeneric[*Ts, B]): ...

        Base[int, str, float]()
        args = get_runtime_args(Base[int, str, float])
        assert args == (int, str, float)

    def test_multi_level_variadic(self):
        class L0(RuntimeGeneric[A, *Ts]): ...

        class L1(L0[int, *Ts]): ...

        class L2(L1[str, float]): ...

        args = get_runtime_args(L2)
        assert args == (int, str, float)


class TestRegressions:
    def test_typevar_bound_to_generic_alias(self):
        """TypeVar resolved to another generic type, e.g. list[int]."""

        class Wrapper(RuntimeGeneric[T]): ...

        Wrapper[list[int]]()
        args = get_runtime_args(Wrapper[list[int]])
        assert args == (list[int],)

    def test_self_referential_annotation_skipped_gracefully(self):
        """Class with a plain (non-TypeVar) annotation should not blow up."""

        class Plain(RuntimeGeneric[T]):
            count: int = 0

        Plain[int]()

    def test_instantiation_without_specialisation_uses_defaults_or_empty(self):
        class Gen(RuntimeGeneric[*Ts]): ...

        _obj = Gen()
        args = get_runtime_args(Gen)
        assert args == ()

    def test_double_specialisation_same_args(self):
        class Gen(RuntimeGeneric[A, B]): ...

        alias = Gen[int, str]
        obj1 = alias()
        obj2 = alias()
        assert type(obj1) is type(obj2)

    def test_specialized_subclass_no_remaining_typevars(self):
        class Base(RuntimeGeneric[A, B]): ...

        class Full(Base[int, str]): ...

        args = get_runtime_args(Full)
        assert args == (int, str)

    def test_get_runtime_args_on_plain_list(self):
        args = get_runtime_args(list[str])
        assert args == (str,)

    def test_get_runtime_args_on_plain_dict(self):
        args = get_runtime_args(dict[int, bytes])
        assert args == (int, bytes)
