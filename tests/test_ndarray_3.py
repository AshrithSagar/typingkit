"""
Extended Tests for TypedNDArray - 3
"""
# tests/test_ndarray_3.py

# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false

from typing import Literal, TypeAlias, TypeVar

import numpy as np
import pytest

from typed_numpy._typed.context import enforce_shapes
from typed_numpy._typed.helpers import FOUR, THREE, TWO
from typed_numpy._typed.ndarray import (
    DimensionError,
    RankError,
    ShapeError,
    TypedNDArray,
)


class TestRepeatedTypeVarConsistency:
    def test_square_matrix_success(self) -> None:
        N = TypeVar("N", bound=int, default=int)
        Square = TypedNDArray[tuple[N, N]]
        arr = Square(np.ones((4, 4)))
        assert arr.shape == (4, 4)

    def test_square_matrix_failure(self) -> None:
        N = TypeVar("N", bound=int, default=int)
        Square = TypedNDArray[tuple[N, N]]
        with pytest.raises(ShapeError):
            Square(np.ones((3, 4)))


class TestLiteralAndTypeVarMix:
    def test_literal_and_typevar_reuse(self):
        N = TypeVar("N")
        Array = TypedNDArray[tuple[Literal[2], N, N]]  # type: ignore
        arr = Array(np.ones((2, 3, 3)))
        assert arr.shape == (2, 3, 3)

    def test_literal_and_typevar_reuse_failure(self):
        N = TypeVar("N")
        Array = TypedNDArray[tuple[Literal[2], N, N]]  # type: ignore
        with pytest.raises(ShapeError):
            Array(np.ones((2, 3, 4)))


class TestMethodLevelIsolation:
    def test_method_level_does_not_persist(self):
        M = TypeVar("M")

        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[M]]) -> TypedNDArray[tuple[M]]:  # type: ignore
                return x

        m = Model()

        a = TypedNDArray[tuple[5]]([1, 2, 3, 4, 5])
        b = TypedNDArray[tuple[3]]([1, 2, 3])

        m.foo(a)
        # Should not conflict with previous call
        m.foo(b)


class TestReturnValidation:
    def test_return_shape_violation(self):
        N = TypeVar("N")

        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[N]]) -> TypedNDArray[tuple[N]]:  # type: ignore
                return TypedNDArray([1, 2, 3])  # always size 3

        m = Model()

        a = TypedNDArray[tuple[5]]([1, 2, 3, 4, 5])

        with pytest.raises(DimensionError):
            m.foo(a)


class TestZeroDimIteration:
    def test_zero_dim_not_iterable(self):
        Scalar = TypedNDArray[tuple[()]]
        arr = Scalar(5)

        with pytest.raises(TypeError):
            iter(arr)


class TestSubclassPreservation:
    def test_view_preserves_type(self):
        Array3 = TypedNDArray[tuple[3]]
        arr = Array3([1, 2, 3])
        view = arr.view()

        assert isinstance(view, TypedNDArray)

    def test_slice_preserves_type(self):
        Array5 = TypedNDArray[tuple[5]]
        arr = Array5([1, 2, 3, 4, 5])
        sliced = arr[1:]

        assert isinstance(sliced, TypedNDArray)
