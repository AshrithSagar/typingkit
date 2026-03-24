"""
Tests for TypedList core functionality - 1
"""
# tests/test_list_1.py

# pyright: reportArgumentType = false
# pyright: reportGeneralTypeIssues = false

from typing import Literal, TypeVar

import pytest

from typingkit.core._validators import LengthError
from typingkit.core.list import ItemError, TypedList


class TestBasicCreation:
    def test_create_untyped(self) -> None:
        lst = TypedList([1, 2, 3])
        assert isinstance(lst, TypedList)
        assert len(lst) == 3

    def test_create_with_length(self) -> None:
        List3 = TypedList[Literal[3]]
        lst = List3([1, 2, 3])
        assert len(lst) == 3

    def test_create_with_length_and_item(self) -> None:
        List3Int = TypedList[Literal[3], int]
        lst = List3Int([1, 2, 3])
        assert len(lst) == 3
        assert all(isinstance(x, int) for x in lst)

    def test_length_mismatch_raises(self) -> None:
        List3 = TypedList[Literal[3]]
        with pytest.raises(LengthError):
            List3([1, 2])


class TestItemValidation:
    def test_item_type_success(self) -> None:
        ListInt = TypedList[int, int]
        lst = ListInt([1, 2, 3])
        assert all(isinstance(x, int) for x in lst)

    def test_item_type_failure(self) -> None:
        ListInt = TypedList[int, int]
        with pytest.raises(ItemError):
            ListInt([1, "a", 3])  # type: ignore

    def test_float_accepts_int(self) -> None:
        ListFloat = TypedList[int, float]
        lst = ListFloat([1, 2.5, 3])
        assert len(lst) == 3

    def test_complex_accepts_int_and_float(self) -> None:
        ListComplex = TypedList[int, complex]
        lst = ListComplex([1, 2.5, 3 + 4j])
        assert len(lst) == 3


class TestLiteralLengths:
    def test_literal_single(self) -> None:
        List5 = TypedList[Literal[5]]
        lst = List5([1, 2, 3, 4, 5])
        assert len(lst) == 5

    def test_literal_multiple(self) -> None:
        List2or4 = TypedList[Literal[2, 4]]
        lst2 = List2or4([1, 2])
        lst4 = List2or4([1, 2, 3, 4])
        assert len(lst2) == 2
        assert len(lst4) == 4

    def test_literal_multiple_failure(self) -> None:
        List2or4 = TypedList[Literal[2, 4]]
        with pytest.raises(LengthError):
            List2or4([1, 2, 3])


class TestUnionLengths:
    def test_union_lengths(self) -> None:
        ListUnion = TypedList[Literal[2] | Literal[3]]
        lst2 = ListUnion([1, 2])
        lst3 = ListUnion([1, 2, 3])
        assert len(lst2) == 2
        assert len(lst3) == 3

    def test_union_failure(self) -> None:
        ListUnion = TypedList[Literal[2] | Literal[3]]
        with pytest.raises(LengthError):
            ListUnion([1])


class TestTypeVarLengths:
    def test_typevar_unbound(self) -> None:
        N = TypeVar("N", bound=int)
        ListN = TypedList[N]
        lst = ListN([1, 2, 3])
        assert len(lst) == 3

    def test_typevar_with_default(self) -> None:
        N = TypeVar("N", bound=int, default=Literal[5])
        ListN = TypedList[N]
        lst = ListN([1, 2, 3, 4, 5])
        assert len(lst) == 5

    def test_typevar_default_mismatch(self) -> None:
        N = TypeVar("N", bound=int, default=Literal[5])
        ListN = TypedList[N]
        with pytest.raises(LengthError):
            ListN([1, 2, 3])
