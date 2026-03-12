"""
Stress tests for RuntimeGeneric - 3
Targets gaps in test_generics_1/2: mixin topology, concurrency, aliased inheritance,
upto edge cases, multiple generic bases, iter_children adversarial, TypeVar constraints.
"""
# tests/test_generics_3.py

# mypy: disable-error-code="annotation-unchecked"
# pyright: reportPrivateUsage = false
# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false
# pyright: reportUnknownVariableType = false

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from types import GenericAlias
from typing import Any, Generic, Optional, TypeVar, TypeVarTuple, get_args

from typingkit.core.generics import (
    RuntimeGeneric,
    _build_mapping,
    _substitute,
    get_runtime_args,
)

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
Ts = TypeVarTuple("Ts")


class TestMixinTopology:
    """Exhaustive mixin ordering and stacking scenarios."""

    def test_two_mixins_before_generic_base(self):
        class M1: ...

        class M2: ...

        class Base(RuntimeGeneric[A, B]): ...

        class Child(M1, M2, Base[int, str]): ...

        assert get_runtime_args(Child) == (int, str)

    def test_mixin_between_two_generic_bases(self):
        """Mixin sandwiched between a generic base and a second specialised base."""

        class Base(RuntimeGeneric[A, B]): ...

        class Mixin: ...

        class Child(Base[int, str], Mixin): ...

        assert get_runtime_args(Child) == (int, str)

    def test_mixin_with_init_does_not_interfere(self):
        class Mixin:
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)

        class Base(RuntimeGeneric[T]):
            def __init__(self, v: Any = None):
                self.v = v

        class Child(Mixin, Base[int]):
            pass

        obj = Child(v=42)
        assert obj.v == 42
        assert get_runtime_args(Child) == (int,)

    def test_unrelated_generic_mixin_does_not_pollute(self):
        """A mixin that is itself generic (but unrelated) should not bleed its params."""

        class GenMixin(RuntimeGeneric[T]):
            pass

        class Base(RuntimeGeneric[A, B]): ...

        class Child(GenMixin[bytes], Base[int, str]): ...

        # Child specialises Base with (int, str); GenMixin binding should not override
        args = get_runtime_args(Child, upto=Base)
        assert args == (int, str)

    def test_three_mixins_all_plain(self):
        class M1: ...

        class M2: ...

        class M3: ...

        class Base(RuntimeGeneric[A]): ...

        class Child(M1, M2, M3, Base[float]): ...

        assert get_runtime_args(Child) == (float,)


class TestUptoWithMixins:
    def test_upto_through_mixin_layer(self):
        class Root(RuntimeGeneric[A, B]): ...

        class Mixin: ...

        class Mid(Mixin, Root[int, B]): ...

        class Leaf(Mid[str]): ...

        args = get_runtime_args(Leaf, upto=Root)
        assert args == (int, str)

    def test_upto_direct_with_mixin_sibling(self):
        class Base(RuntimeGeneric[T]): ...

        class Mixin: ...

        class Child(Mixin, Base[float]): ...

        assert get_runtime_args(Child, upto=Base) == (float,)

    def test_upto_not_in_mro_returns_gracefully(self):
        """upto targeting a class not in the MRO should not raise."""

        class Base(RuntimeGeneric[T]): ...

        class Unrelated(RuntimeGeneric[T]): ...

        class Child(Base[int]): ...

        # upto=Unrelated is not in Child's MRO — should not raise
        result = get_runtime_args(Child, upto=Unrelated)
        assert isinstance(result, tuple)

    def test_upto_intermediate_skipped_correctly(self):
        class Root(RuntimeGeneric[A, B]): ...

        class Mid(Root[int, B]): ...

        class Mixin: ...

        class Leaf(Mixin, Mid[str]): ...

        assert get_runtime_args(Leaf, upto=Root) == (int, str)


class TestCurriedAliasInheritance:
    def test_subclass_of_curried_alias(self):
        class Base(RuntimeGeneric[A, B, C]): ...

        Partial = Base[int, B, C]

        class Child(Partial[str, C]): ...  # type: ignore[valid-type]

        args = get_runtime_args(Child[float])
        assert args == (int, str, float)

    def test_instantiate_subclass_of_curried_alias(self):
        class Base(RuntimeGeneric[A, B]):
            def __init__(self, a: Any = None, b: Any = None):
                self.a = a
                self.b = b

        Partial = Base[int, B]

        class Child(Partial[str]): ...  # type: ignore[valid-type]

        obj = Child(a=1, b="x")
        assert obj.a == 1
        assert obj.b == "x"

    def test_deep_curry_chain_then_subclass(self):
        class Base(RuntimeGeneric[A, B, C, D]): ...

        S1 = Base[int, B, C, D]
        S2 = S1[str, C, D]

        class Leaf(S2[float, D]): ...  # type: ignore[valid-type]

        args = get_runtime_args(Leaf[bytes])
        assert args == (int, str, float, bytes)


class TestMultipleGenericBases:
    def test_two_independent_generic_bases(self):
        """Child inherits from two different fully-specialised RuntimeGeneric bases."""

        class Base1(RuntimeGeneric[A]): ...

        class Base2(RuntimeGeneric[B]): ...

        class Child(Base1[int], Base2[str]): ...

        assert get_runtime_args(Child, upto=Base1) == (int,)
        assert get_runtime_args(Child, upto=Base2) == (str,)

    def test_two_generic_bases_same_typevar_name(self):
        """Both bases use TypeVar named 'T' (different objects); bindings must not collide."""
        T1 = TypeVar("T1")
        T2 = TypeVar("T2")

        class Left(RuntimeGeneric[T1]): ...

        class Right(RuntimeGeneric[T2]): ...

        class Child(Left[int], Right[str]): ...

        assert get_runtime_args(Child, upto=Left) == (int,)
        assert get_runtime_args(Child, upto=Right) == (str,)

    def test_diamond_two_paths_same_arg(self):
        """Diamond: both paths bind the same TypeVar to the same concrete type."""

        class Root(RuntimeGeneric[T]): ...

        class Left(Root[int]): ...

        class Right(Root[int]): ...

        class Diamond(Left, Right): ...

        # Should resolve consistently, not raise
        args = get_runtime_args(Diamond, upto=Root)
        assert args == (int,)


class TestConcurrency:
    def test_concurrent_instantiation_no_cross_contamination(self):
        """
        Multiple threads instantiating different specialisations simultaneously
        must not bleed ContextVar state into each other.
        """
        collected: dict[int, tuple[Any, ...]] = {}
        errors: list[Exception] = []

        class Probe(RuntimeGeneric[T]):
            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                # Record which args this instance received
                collected[threading.get_ident()] = get_args(alias)

        type_map = [int, str, float, bytes, bool, list, dict, set, tuple, complex]

        def run(tp: type) -> None:
            try:
                Probe[tp]()  # type: ignore[valid-type]
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(tp,)) for tp in type_map]  # pyright: ignore[reportUnknownArgumentType]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        # Each thread should have recorded exactly one-element args
        for args in collected.values():
            assert len(args) == 1
            assert args[0] in type_map

    def test_nested_instantiation_context_isolation(self):
        """
        Instantiating a RuntimeGeneric *inside* another's __init__ must not
        corrupt the outer ContextVar.
        """
        outer_args: list[Any] = []
        inner_args: list[Any] = []

        class Inner(RuntimeGeneric[T]):
            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                inner_args.append(get_args(alias))

        class Outer(RuntimeGeneric[T]):
            def __init__(self) -> None:
                self._inner = Inner[str]()  # Different type arg than Outer

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                outer_args.append(get_args(alias))

        Outer[int]()

        assert (int,) in outer_args
        # Inner must have been bound to str, not int
        assert (str,) in inner_args
        assert (int,) not in inner_args


class TestIterChildrenAdversarial:
    def test_iter_children_yields_same_child_twice(self):
        """Yielding the same child object twice should not raise."""
        post_inits: list[Any] = []

        class Leaf(RuntimeGeneric[T]):
            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                post_inits.append(get_args(alias))

        class Container(RuntimeGeneric[T]):
            def __init__(self, child: Any):
                self.child = child

            def __runtime_generic_iter_children__(
                self, mapping: dict[Any, Any]
            ) -> Iterable[tuple[Any, Any]]:
                yield self.child, _substitute(Leaf[T], mapping)
                yield self.child, _substitute(Leaf[T], mapping)  # duplicate

        leaf = Leaf[int]()
        post_inits.clear()
        leaf._runtime_validated = False  # allow propagation to fire
        Container[int](child=leaf)
        # Should be called twice, not raise
        assert post_inits.count((int,)) == 1  # second yield blocked by guard

    def test_iter_children_yields_non_runtime_generic(self):
        """Yielding a plain int as a child should be a no-op, not raise."""

        class Container(RuntimeGeneric[T]):
            def __runtime_generic_iter_children__(
                self, mapping: dict[Any, Any]
            ) -> Iterable[tuple[Any, Any]]:
                yield 42, int  # plain value, not a RuntimeGeneric

        Container[int]()  # must not raise

    def test_iter_children_yields_none(self):
        class Container(RuntimeGeneric[T]):
            def __runtime_generic_iter_children__(
                self, mapping: dict[Any, Any]
            ) -> Iterable[tuple[Any, Any]]:
                yield None, type(None)

        Container[int]()  # must not raise

    def test_iter_children_large_fanout(self):
        """Many children — ensures no quadratic blowup or recursion limit hit."""
        N = 500
        post_inits: list[Any] = []

        class Leaf(RuntimeGeneric[T]):
            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                post_inits.append(1)

        class Wide(RuntimeGeneric[T]):
            def __init__(self, children: list[Any]):
                self.children = children

            def __runtime_generic_iter_children__(
                self, mapping: dict[Any, Any]
            ) -> Iterable[tuple[Any, Any]]:
                for child in self.children:
                    yield child, _substitute(Leaf[T], mapping)

        leaves = [Leaf[int]() for _ in range(N)]
        post_inits.clear()
        for leaf in leaves:
            leaf._runtime_validated = False  # allow propagation
        Wide[int](children=leaves)
        assert len(post_inits) == N


class TestTypeVarConstraints:
    def test_constrained_typevar_concrete_arg(self):
        NumOrStr = TypeVar("NumOrStr", int, str)

        class Typed(RuntimeGeneric[NumOrStr]): ...

        Typed[int]()
        assert get_runtime_args(Typed[int]) == (int,)

        Typed[str]()
        assert get_runtime_args(Typed[str]) == (str,)

    def test_constrained_typevar_in_mapping(self):
        NumOrStr = TypeVar("NumOrStr", int, str)
        mapping, _ = _build_mapping((NumOrStr,), (int,))
        assert mapping[NumOrStr] is int

    def test_constrained_typevar_inherited(self):
        NumOrStr = TypeVar("NumOrStr", int, str)

        class Base(RuntimeGeneric[NumOrStr]): ...

        class Child(Base[int]): ...

        assert get_runtime_args(Child) == (int,)


class TestPostInitAliasCoercion:
    def test_post_init_with_parent_alias_on_child_instance(self):
        """
        Calling __runtime_generic_post_init__ on a Child instance using
        a Base alias should not raise and should propagate the base's args.
        """
        received: list[Any] = []

        class Base(RuntimeGeneric[T]):
            def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
                received.append(get_args(alias))

        class Child(Base[T]): ...

        obj = Child[int]()
        received.clear()
        obj._runtime_validated = False  # reset so manual call fires
        obj.__runtime_generic_post_init__(Base[str])  # type: ignore[arg-type]
        assert (str,) in received

    def test_double_post_init_different_args(self):
        """Two consecutive post_inits with different args — last one wins for propagation."""
        log: list[Any] = []

        class Leaf(RuntimeGeneric[T]):
            def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
                log.append(get_args(alias))

        obj = Leaf[int]()
        log.clear()
        obj._runtime_validated = False
        obj.__runtime_generic_post_init__(GenericAlias(Leaf, (int,)))
        obj._runtime_validated = False
        obj.__runtime_generic_post_init__(GenericAlias(Leaf, (str,)))
        assert log == [(int,), (str,)]


class TestNonRuntimeGenericWithOrigBases:
    def test_plain_generic_subclass(self):
        """A class that inherits from a plain Generic (not RuntimeGeneric)."""

        U = TypeVar("U")

        class PlainGen(Generic[U]): ...

        class Child(PlainGen[int]): ...

        # Should not raise; behaviour may be empty tuple or (int,)
        result = get_runtime_args(Child)
        assert isinstance(result, tuple)

    def test_builtin_subclass_with_orig_bases(self):
        """list subclass — has __orig_bases__ via Generic machinery."""

        class MyList(list[int]): ...

        result = get_runtime_args(MyList)
        assert isinstance(result, tuple)


class TestDataclassComplexFields:
    def test_dataclass_optional_field(self):

        @dataclass
        class MaybeBox(RuntimeGeneric[T]):
            value: Optional[T] = None

        obj = MaybeBox[int](value=None)
        assert obj.value is None

        obj2 = MaybeBox[int](value=5)
        assert obj2.value == 5

    def test_dataclass_list_of_t_field(self):
        @dataclass
        class Bag(RuntimeGeneric[T]):
            items: list[T]

        obj = Bag[int](items=[1, 2, 3])
        assert obj.items == [1, 2, 3]

    def test_dataclass_nested_generic_field_two_levels(self):
        """Three-level nesting: Outer[T] -> Mid[T] -> Inner[T]."""
        leaf_log: list[Any] = []

        class Inner(RuntimeGeneric[T]):
            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                leaf_log.append(get_args(alias))

        @dataclass
        class Mid(RuntimeGeneric[T]):
            inner: Inner[T]

        @dataclass
        class Outer(RuntimeGeneric[T]):
            mid: Mid[T]

        inner = Inner[int]()
        mid = Mid[int](inner=inner)
        leaf_log.clear()
        inner._runtime_validated = False  # allow Outer's propagation to reach Inner
        mid._runtime_validated = (
            False  # allow Outer's propagation to reach Mid (and through it, Inner)
        )
        Outer[int](mid=mid)
        assert any(a == (int,) for a in leaf_log)


class TestRespecialisationStability:
    def test_alias_reused_across_many_instances(self):
        class Gen(RuntimeGeneric[A, B]): ...

        alias = Gen[int, str]
        for _ in range(200):
            obj = alias()
            assert isinstance(obj, Gen)

    def test_get_runtime_args_stable_after_many_calls(self):
        class Gen(RuntimeGeneric[T]): ...

        alias = Gen[int]
        for _ in range(200):
            assert get_runtime_args(alias) == (int,)

    def test_independent_subclass_specialisations_do_not_share_state(self):
        class Base(RuntimeGeneric[T]): ...

        class Child1(Base[int]): ...

        class Child2(Base[str]): ...

        assert get_runtime_args(Child1) == (int,)
        assert get_runtime_args(Child2) == (str,)
        # Re-check to ensure no mutation
        assert get_runtime_args(Child1) == (int,)


class TestTypeVarTupleEdgeCases:
    def test_singleton_variadic(self):
        class V(RuntimeGeneric[*Ts]): ...

        V[int]()
        assert get_runtime_args(V[int]) == (int,)

    def test_variadic_with_generic_args(self):
        class V(RuntimeGeneric[*Ts]): ...

        V[list[int], dict[str, float]]()
        assert get_runtime_args(V[list[int], dict[str, float]]) == (
            list[int],
            dict[str, float],
        )

    def test_variadic_prefix_suffix_fixed(self):
        class V(RuntimeGeneric[A, *Ts, B]): ...

        V[int, str, bytes, float]()
        args = get_runtime_args(V[int, str, bytes, float])
        assert args == (int, str, bytes, float)

    def test_variadic_only_prefix(self):
        """Only fixed prefix provided, variadic empty."""

        class V(RuntimeGeneric[A, *Ts]): ...

        V[int]()
        assert get_runtime_args(V[int]) == (int,)

    def test_variadic_subclass_adds_to_suffix(self):
        class Base(RuntimeGeneric[*Ts, B]): ...

        class Child(Base[*Ts, int]): ...

        Child[str, float]()
        args = get_runtime_args(Child[str, float])
        assert args == (str, float, int)
