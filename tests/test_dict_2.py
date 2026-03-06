"""
Tests for TypedDict core functionality - 2
"""
# tests/test_dict_2.py

# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false

from typing import Any, Literal, TypeVar

import pytest

from typingkit.core.dict import LengthError, TypedDict, TypedDictConfig


class TestDeferredBinding:
    def test_progressive_length_binding(self) -> None:
        N = TypeVar("N", bound=int)

        DictN = TypedDict[N]

        Dict3 = DictN[Literal[3]]

        d = Dict3({"a": 1, "b": 2, "c": 3})

        assert len(d) == 3

    def test_bind_length_key_value(self) -> None:
        N = TypeVar("N", bound=int)

        DictNStrInt = TypedDict[N, str, int]

        Dict2 = DictNStrInt[Literal[2]]

        d = Dict2({"a": 1, "b": 2})

        assert len(d) == 2

    def test_too_many_arguments(self) -> None:
        DictN = TypedDict[int]

        with pytest.raises(TypeError):
            DictN[int, int, int, int]  # type: ignore


class TestLengthProperty:
    def test_length_property(self) -> None:
        Dict3 = TypedDict[Literal[3]]

        d = Dict3({"a": 1, "b": 2, "c": 3})

        assert d.length == 3


class TestConfigToggles:
    def test_disable_length_validation(self) -> None:
        TypedDictConfig.disable_all()

        Dict2 = TypedDict[Literal[2]]

        d = Dict2({"a": 1})

        assert len(d) == 1

        TypedDictConfig.enable_all()


class TestErrorMessages:
    def test_length_error_message(self) -> None:
        Dict2 = TypedDict[Literal[2]]

        with pytest.raises(LengthError) as exc:
            Dict2({"a": 1})

        msg = str(exc.value)

        assert "expected 2" in msg
        assert "got 1" in msg


class TestEdgeCases:
    def test_empty_dict_literal_zero(self) -> None:
        Dict0 = TypedDict[Literal[0]]

        d = Dict0({})

        assert len(d) == 0

    def test_any_length(self) -> None:
        DictAny = TypedDict[Any]

        d = DictAny({"a": 1, "b": 2, "c": 3})

        assert len(d) == 3

    def test_nested_typeddict(self) -> None:
        Inner = TypedDict[Literal[1]]
        Outer = TypedDict[Literal[2], str, Inner]

        d = Outer(
            {
                "x": Inner({"a": 1}),
                "y": Inner({"b": 2}),
            }
        )

        assert len(d) == 2
