"""
Extended Tests for TypedList - 2
"""
# tests/test_list_2.py

# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false

from typing import Any, Literal, TypeVar

import pytest

from typingkit.core import TypedCollectionConfig
from typingkit.core.list import ItemError, LengthError, TypedList


class TestDeferredBinding:
    def test_progressive_length_binding(self) -> None:
        N = TypeVar("N", bound=int)
        ListN = TypedList[N]

        List5 = ListN[Literal[5]]
        lst = List5([1, 2, 3, 4, 5])
        assert len(lst) == 5

    def test_bind_length_and_item(self) -> None:
        N = TypeVar("N", bound=int)
        ListNInt = TypedList[N, int]

        List3Int = ListNInt[Literal[3]]
        lst = List3Int([1, 2, 3])
        assert len(lst) == 3

    def test_too_many_arguments(self) -> None:
        ListN = TypedList[int]
        with pytest.raises(TypeError):
            ListN[int, int, int]  # type: ignore


class TestFullConstructor:
    def test_full_with_constant(self) -> None:
        List3 = TypedList[Literal[3], int]
        lst = List3.full(3, 42)
        assert lst == [42, 42, 42]

    def test_full_with_callable(self) -> None:
        List5 = TypedList[Literal[5], int]
        lst = List5.full(5, lambda i: i * 2)
        assert lst == [0, 2, 4, 6, 8]


class TestCopyAndLengthProperty:
    def test_copy_preserves_type(self) -> None:
        List3 = TypedList[Literal[3], int]
        lst = List3([1, 2, 3])
        copied = lst.copy()

        assert isinstance(copied, TypedList)
        assert copied == lst

    def test_length_property(self) -> None:
        List4 = TypedList[Literal[4]]
        lst = List4([1, 2, 3, 4])
        assert lst.length == 4


class TestConfigToggles:
    def test_disable_length_validation(self) -> None:
        TypedCollectionConfig.disable_all()
        List3 = TypedList[Literal[3]]

        # Should not raise
        lst = List3([1, 2])

        assert len(lst) == 2
        TypedCollectionConfig.enable_all()

    def test_disable_item_validation(self) -> None:
        TypedCollectionConfig.disable_all()
        ListInt = TypedList[int, int]

        # Should not raise
        lst = ListInt([1, "oops", 3])  # type: ignore

        assert len(lst) == 3
        TypedCollectionConfig.enable_all()


class TestErrorMessages:
    def test_length_error_message(self) -> None:
        List3 = TypedList[Literal[3]]
        with pytest.raises(LengthError) as exc:
            List3([1, 2])

        msg = str(exc.value)
        assert "expected 3" in msg
        assert "got 2" in msg

    def test_item_error_message(self) -> None:
        ListInt = TypedList[int, int]
        with pytest.raises(ItemError) as exc:
            ListInt([1, "a", 3])  # type: ignore

        msg = str(exc.value)
        assert "expected 'int'" in msg
        assert "got 'str'" in msg
        assert "index 1" in msg


class TestEdgeCases:
    def test_empty_list_literal_zero(self) -> None:
        List0 = TypedList[Literal[0]]
        lst = List0([])
        assert len(lst) == 0

    def test_any_length(self) -> None:
        ListAny = TypedList[Any]
        lst = ListAny([1, 2, 3, 4])
        assert len(lst) == 4

    def test_nested_typedlist(self) -> None:
        Inner = TypedList[Literal[2], int]
        Outer = TypedList[Literal[2], Inner]

        lst = Outer([Inner([1, 2]), Inner([3, 4])])
        assert len(lst) == 2
