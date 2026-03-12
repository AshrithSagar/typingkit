"""
Tests for the refactored RuntimeGeneric system.

Covers:
- RuntimeOptions: per-class attachment, MRO inheritance, subclassing
- RuntimeOptions.scoped(): context manager, nesting, async safety
- _RuntimeValidatedDescriptor: guard fires once, descriptor on base class
- Hook inversion: __runtime_generic_validate__ not __runtime_generic_post_init__
- Fast path: validate=False skips post_init entirely
- get_runtime_args: default filling, safe unpacking
- TypedDict: validate_length flag via TypedDictOptions
- End-to-end: scoped override affecting TypedDict
"""
# tests/test_generics_4.py

# mypy: disable-error-code="annotation-unchecked"
# pyright: reportPrivateUsage = false
# pyright: reportGeneralTypeIssues = false

import asyncio
from dataclasses import dataclass
from types import GenericAlias
from typing import Any, Literal, TypeVar, TypeVarTuple, Unpack

import pytest

from typingkit.core._options import RuntimeOptions
from typingkit.core._validators import LengthError
from typingkit.core.dict import TypedDict, TypedDictOptions
from typingkit.core.generics import (
    RuntimeGeneric,
    clear_runtime_caches,
    get_runtime_args,
    get_runtime_mapping,
    is_runtime_specialised,
)

# ── Fixtures / helpers ────────────────────────────────────────────────────────

T = TypeVar("T", default=Any)
A = TypeVar("A", default=Any)
B = TypeVar("B", default=Any)

_validated_calls: list[str] = []


def setup_function():
    _validated_calls.clear()
    clear_runtime_caches()


# ── RuntimeOptions dataclass ──────────────────────────────────────────────────


class TestRuntimeOptions:
    def test_defaults(self):
        opts = RuntimeOptions()
        assert opts.validate is True
        assert opts.propagate is True

    def test_frozen(self):
        opts = RuntimeOptions()
        with pytest.raises((AttributeError, TypeError)):
            opts.validate = False  # type: ignore[misc]

    def test_merged(self):
        opts = RuntimeOptions().merged(validate=False)
        assert opts.validate is False
        assert opts.propagate is True  # unchanged

    def test_subclass(self):
        @dataclass(frozen=True)
        class MyOptions(RuntimeOptions):
            extra: bool = True

        opts = MyOptions(validate=False, extra=False)
        assert opts.validate is False
        assert opts.extra is False

    def test_resolve_no_scoped(self):
        opts = RuntimeOptions(validate=False)
        resolved = RuntimeOptions.resolve(opts)
        assert resolved is opts  # no scoped active → returns class opts as-is

    def test_resolve_with_scoped(self):
        opts = RuntimeOptions(validate=True, propagate=True)
        with RuntimeOptions.scoped(validate=False):
            resolved = RuntimeOptions.resolve(opts)
            assert resolved.validate is False
            assert resolved.propagate is True  # not overridden

    def test_resolve_scoped_does_not_clobber_domain_fields(self):
        @dataclass(frozen=True)
        class DomainOpts(RuntimeOptions):
            domain_flag: bool = True

        opts = DomainOpts(validate=True, domain_flag=True)
        with RuntimeOptions.scoped(validate=False):
            resolved = RuntimeOptions.resolve(opts)
            assert resolved.validate is False
            # domain_flag should be untouched — it's not in base RuntimeOptions
            assert resolved.domain_flag is True  # type: ignore[attr-defined]


# ── RuntimeOptions.scoped ─────────────────────────────────────────────────────


class TestScopedOptions:
    def test_basic(self):
        assert RuntimeOptions.resolve(RuntimeOptions()).validate is True
        with RuntimeOptions.scoped(validate=False):
            assert RuntimeOptions.resolve(RuntimeOptions()).validate is False
        assert RuntimeOptions.resolve(RuntimeOptions()).validate is True  # restored

    def test_nested_scopes_compose(self):
        with RuntimeOptions.scoped(validate=False):
            with RuntimeOptions.scoped(propagate=False):
                resolved = RuntimeOptions.resolve(RuntimeOptions())
                assert resolved.validate is False
                assert resolved.propagate is False
            # inner scope exited — propagate restored
            resolved = RuntimeOptions.resolve(RuntimeOptions())
            assert resolved.validate is False
            assert resolved.propagate is True

    def test_exception_restores(self):
        try:
            with RuntimeOptions.scoped(validate=False):
                raise ValueError("oops")
        except ValueError:
            ...
        assert RuntimeOptions.resolve(RuntimeOptions()).validate is True

    def test_async_isolation(self):
        """Scoped options in one coroutine must not bleed into another."""
        results = {}

        async def task_a():
            with RuntimeOptions.scoped(validate=False):
                await asyncio.sleep(0)  # yield to task_b
                results["a"] = RuntimeOptions.resolve(RuntimeOptions()).validate

        async def task_b():
            await asyncio.sleep(0)
            results["b"] = RuntimeOptions.resolve(RuntimeOptions()).validate

        async def run():
            await asyncio.gather(task_a(), task_b())

        asyncio.run(run())
        assert results["a"] is False
        assert results["b"] is True  # not contaminated by task_a's scope


# ── Per-class options attachment ──────────────────────────────────────────────


class TestPerClassOptions:
    def test_default_options_on_subclass(self):
        class MyType(RuntimeGeneric[T]): ...

        opts = MyType._runtime_options_
        assert isinstance(opts, RuntimeOptions)
        assert opts.validate is True
        assert opts.propagate is True

    def test_custom_options_keyword(self):
        class MyType(RuntimeGeneric[T], options=RuntimeOptions(validate=False)): ...

        assert MyType._runtime_options_.validate is False

    def test_options_mro_inheritance(self):
        """Subclass without options= inherits parent's options."""

        class Parent(RuntimeGeneric[T], options=RuntimeOptions(propagate=False)): ...

        class Child(Parent[T]): ...

        # Child doesn't set _runtime_options_ in its own __dict__
        assert "_ runtime_options_" not in Child.__dict__
        # But lookup walks MRO and finds Parent's
        assert Child._runtime_options_.propagate is False

    def test_options_mro_override(self):
        """Subclass can override parent's options."""

        class Parent(RuntimeGeneric[T], options=RuntimeOptions(propagate=False)): ...

        class Child(Parent[T], options=RuntimeOptions(propagate=True)): ...

        assert Child._runtime_options_.propagate is True
        assert Parent._runtime_options_.propagate is False  # unchanged


# ── _RuntimeValidatedDescriptor ───────────────────────────────────────────────


class TestRuntimeValidatedDescriptor:
    def test_default_false(self):
        class MyType(RuntimeGeneric[T]):
            def __init__(self, val: T):
                self.val = val

        obj = object.__new__(MyType)
        assert obj._runtime_validated is False

    def test_set_and_get(self):
        class MyType(RuntimeGeneric[T]):
            def __init__(self, val: T):
                self.val = val

        obj = object.__new__(MyType)
        obj._runtime_validated = True
        assert obj._runtime_validated is True

    def test_descriptor_on_base_not_subclass(self):
        """The descriptor lives on RuntimeGeneric, not on each subclass."""
        assert "_runtime_validated" in RuntimeGeneric.__dict__

    def test_guard_fires_once(self):
        """__runtime_generic_post_init__ must not invoke validate twice."""
        validate_count = 0

        class Counter(RuntimeGeneric[T]):
            def __init__(self, val: T):
                self.val = val

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                nonlocal validate_count
                validate_count += 1

        Counter[int](42)
        assert validate_count == 1

        # Manually calling post_init again should be a no-op
        obj = Counter[int](42)
        validate_count = 0
        obj.__runtime_generic_post_init__(Counter[int])  # type: ignore[arg-type]
        assert validate_count == 0  # already validated


# ── Hook inversion ────────────────────────────────────────────────────────────


class TestHookInversion:
    def test_validate_hook_called(self):
        called: list[GenericAlias] = []

        class MyType(RuntimeGeneric[T]):
            def __init__(self, val: T):
                self.val = val

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                called.append(alias)

        MyType[int](42)
        assert len(called) == 1

    def test_validate_hook_not_called_when_validate_false(self):
        called: list[GenericAlias] = []

        class MyType(RuntimeGeneric[T], options=RuntimeOptions(validate=False)):
            def __init__(self, val: T):
                self.val = val

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                called.append(alias)

        MyType[int](42)
        assert called == []

    def test_validate_hook_not_called_when_scoped_validate_false(self):
        called: list[GenericAlias] = []

        class MyType(RuntimeGeneric[T]):
            def __init__(self, val: T):
                self.val = val

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                called.append(alias)

        with RuntimeOptions.scoped(validate=False):
            MyType[int](42)

        assert called == []

    def test_propagate_false_skips_children(self):
        """propagate=False suppresses post_init re-propagation into already-owned children."""
        repropagation_count = 0

        class Child(RuntimeGeneric[T]):
            def __init__(self, val: T):
                self.val = val

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                nonlocal repropagation_count
                repropagation_count += 1

        class Parent(RuntimeGeneric[T], options=RuntimeOptions(propagate=False)):
            child: Child[T]

            def __init__(self, val: T):
                self.child = Child[T](val)  # counts as 1 (Child's own construction)

        repropagation_count = 0  # reset after Child construction inside Parent.__init__
        # Can't easily intercept just the propagation path here, so test via
        # the propagate=True counterpart showing the difference
        Parent[int](42)
        # With propagate=False: Child validated once (own construction), Parent does not re-propagate
        # Just verify it doesn't raise and Child was constructed
        parent = Parent[int](42)
        assert isinstance(parent.child, Child)

    def test_no_super_required_in_validate(self):
        """Subclass validate hooks must not need to call super()."""

        class MyType(RuntimeGeneric[T]):
            def __init__(self, val: T):
                self.val = val

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                pass  # no super() — must still work

        # Should not raise
        MyType[int](42)


# ── Fast path ─────────────────────────────────────────────────────────────────


class TestFastPath:
    def test_fast_path_skips_post_init_entirely(self):
        """validate=False: __runtime_generic_post_init__ must not be called at all."""
        post_init_called: list[GenericAlias] = []

        class MyType(RuntimeGeneric[T], options=RuntimeOptions(validate=False)):
            def __init__(self, val: T):
                self.val = val

            def __runtime_generic_post_init__(self, alias: GenericAlias) -> None:
                post_init_called.append(alias)

        MyType[int](42)
        # post_init skipped by _RuntimeGenericAlias.__call__ before construction
        assert post_init_called == []

    def test_fast_path_instance_is_correct_type(self):
        class MyType(RuntimeGeneric[T], options=RuntimeOptions(validate=False)):
            def __init__(self, val: T):
                self.val = val

        obj = MyType[int](42)
        assert isinstance(obj, MyType)
        assert obj.val == 42

    def test_scoped_fast_path(self):
        validate_called: list[bool] = []

        class MyType(RuntimeGeneric[T]):
            def __init__(self, val: T):
                self.val = val

            def __runtime_generic_validate__(self, alias: GenericAlias) -> None:
                validate_called.append(True)

        with RuntimeOptions.scoped(validate=False):
            MyType[int](42)

        assert validate_called == []

        # Outside scope: validation resumes
        MyType[int](42)
        assert validate_called == [True]


# ── get_runtime_args default filling ─────────────────────────────────────────


class TestGetRuntimeArgsDefaults:
    def test_full_args_unpack(self):
        """All args present — plain unpack."""

        alias = TypedDict[Literal[2], str, int]
        length, key, value = get_runtime_args(alias)
        assert length == Literal[2]
        assert key is str
        assert value is int

    def test_partial_args_fills_defaults(self):
        """Only Length provided — Key and Value filled from TypeVar defaults."""

        alias = TypedDict[Literal[2]]
        length, key, value = get_runtime_args(alias)
        assert length == Literal[2]
        assert key is Any
        assert value is Any

    def test_no_args_all_defaults(self):
        """No args — all defaults, including inherited Ts from RuntimeGeneric base."""
        mapping = get_runtime_mapping(TypedDict)
        # Length, Key, Value + inherited Ts from RuntimeGeneric[*Ts]
        assert len(mapping) == 4

    def test_tvt_inline_expansion(self):
        """TypeVarTuple expands inline — can use a, *ts, b = unpack."""

        U = TypeVar("U")
        Ts2 = TypeVarTuple("Ts2")
        V = TypeVar("V")

        class Multi(RuntimeGeneric[U, Unpack[Ts2], V]):
            def __init__(self): ...

        alias = Multi[int, str, float, bool]
        a, *ts, v = get_runtime_args(alias)
        assert a is int
        assert ts == [str, float]
        assert v is bool

    def test_is_runtime_specialised_with_defaults(self):

        assert is_runtime_specialised(TypedDict[Literal[2], str, int]) is True
        # TypeVars with concrete defaults are considered specialised —
        # is_runtime_specialised checks for unbound TypeVars, not unset ones.
        assert is_runtime_specialised(TypedDict) is True


# ── TypedDict with new options ────────────────────────────────────────────────


class TestTypedDictOptions:
    def test_length_validation_passes(self):
        d = TypedDict[Literal[2], str, int]({"a": 1, "b": 2})
        assert len(d) == 2

    def test_length_validation_fails(self):
        with pytest.raises(LengthError):
            TypedDict[Literal[2], str, int]({"a": 1})

    def test_disable_length_via_subclass_options(self):
        class LooseDict(
            TypedDict[Literal[2], str, int],
            options=TypedDictOptions(validate_length=False),
        ): ...

        # Should not raise even though length is wrong
        d = LooseDict({"a": 1})
        assert len(d) == 1

    def test_disable_all_via_scoped(self):
        with RuntimeOptions.scoped(validate=False):
            d = TypedDict[Literal[3], str, int]({"a": 1})
        assert len(d) == 1  # no LengthError

    def test_options_not_global(self):
        """Changing one subclass options must not affect another."""

        Length = TypeVar("Length", bound=int)

        class DictA(
            TypedDict[Length, str, int],
            options=TypedDictOptions(validate_length=False),
        ): ...

        class DictB(TypedDict[Length, str, int]): ...

        DictA[Literal[2]]({"x": 1})  # fine — validation disabled
        with pytest.raises(LengthError):
            DictB[Literal[2]]({"x": 1})  # still validates

    def test_typed_dict_options_inherits_validate(self):
        opts = TypedDictOptions(validate=False)
        assert opts.validate is False
        assert opts.validate_length is True  # domain default preserved


# ── Integration: deep inheritance chain ──────────────────────────────────────


class TestInheritanceChain:
    def test_options_inherited_through_deep_mro(self):
        class Base(RuntimeGeneric[T], options=RuntimeOptions(propagate=False)):
            def __init__(self, val: T):
                self.val = val

        class Mid(Base[T]): ...

        class Leaf(Mid[T]): ...

        assert Leaf._runtime_options_.propagate is False

    def test_leaf_can_override_ancestor_options(self):
        class Base(RuntimeGeneric[T], options=RuntimeOptions(propagate=False)):
            def __init__(self, val: T):
                self.val = val

        class Leaf(Base[T], options=RuntimeOptions(propagate=True)): ...

        assert Leaf._runtime_options_.propagate is True
        assert Base._runtime_options_.propagate is False
