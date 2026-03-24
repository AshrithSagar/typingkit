"""
Tests for TypedDict core functionality - 1
"""
# tests/test_dict_1.py

# pyright: reportGeneralTypeIssues = false

from typing import Literal, TypeVar

import pytest

from typingkit.core._validators import LengthError
from typingkit.core.dict import TypedDict


class TestBasicCreation:
    def test_create_untyped(self) -> None:
        d = TypedDict({"a": 1, "b": 2})

        assert isinstance(d, TypedDict)
        assert len(d) == 2

    def test_create_with_length(self) -> None:
        Dict2 = TypedDict[Literal[2]]

        d = Dict2({"a": 1, "b": 2})

        assert len(d) == 2

    def test_create_with_length_key_value(self) -> None:
        Dict2StrInt = TypedDict[Literal[2], str, int]

        d = Dict2StrInt({"a": 1, "b": 2})

        assert len(d) == 2
        assert all(isinstance(k, str) for k in d)
        assert all(isinstance(v, int) for v in d.values())

    def test_length_mismatch_raises(self) -> None:
        Dict2 = TypedDict[Literal[2]]

        with pytest.raises(LengthError):
            Dict2({"a": 1})


class TestLiteralLengths:
    def test_literal_single(self) -> None:
        Dict3 = TypedDict[Literal[3]]

        d = Dict3({"a": 1, "b": 2, "c": 3})

        assert len(d) == 3

    def test_literal_multiple(self) -> None:
        Dict2or4 = TypedDict[Literal[2, 4]]

        d2 = Dict2or4({"a": 1, "b": 2})
        d4 = Dict2or4({"a": 1, "b": 2, "c": 3, "d": 4})

        assert len(d2) == 2
        assert len(d4) == 4

    def test_literal_multiple_failure(self) -> None:
        Dict2or4 = TypedDict[Literal[2, 4]]

        with pytest.raises(LengthError):
            Dict2or4({"a": 1, "b": 2, "c": 3})


class TestUnionLengths:
    def test_union_lengths(self) -> None:
        DictUnion = TypedDict[Literal[1] | Literal[3]]

        d1 = DictUnion({"a": 1})
        d3 = DictUnion({"a": 1, "b": 2, "c": 3})

        assert len(d1) == 1
        assert len(d3) == 3

    def test_union_failure(self) -> None:
        DictUnion = TypedDict[Literal[1] | Literal[3]]

        with pytest.raises(LengthError):
            DictUnion({"a": 1, "b": 2})


class TestTypeVarLengths:
    def test_typevar_unbound(self) -> None:
        N = TypeVar("N", bound=int)

        DictN = TypedDict[N]

        d = DictN({"a": 1, "b": 2})

        assert len(d) == 2

    def test_typevar_with_default(self) -> None:
        N = TypeVar("N", bound=int, default=Literal[2])

        DictN = TypedDict[N]

        d = DictN({"a": 1, "b": 2})

        assert len(d) == 2

    def test_typevar_default_mismatch(self) -> None:
        N = TypeVar("N", bound=int, default=Literal[2])

        DictN = TypedDict[N]

        with pytest.raises(LengthError):
            DictN({"a": 1})
