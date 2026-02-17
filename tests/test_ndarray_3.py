"""
Extended Tests for TypedNDArray - 3
"""
# tests/test_ndarray_3.py

# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false

from typing import TypeVar

import numpy as np
import pytest

from typed_numpy._typed.context import enforce_shapes
from typed_numpy._typed.helpers import FIVE, THREE, TWO
from typed_numpy._typed.ndarray import DimensionError, ShapeError, TypedNDArray

N = TypeVar("N", bound=int, default=int)
M = TypeVar("M", bound=int, default=int)


class TestRepeatedTypeVarConsistency:
    def test_square_matrix_success(self) -> None:
        Square = TypedNDArray[tuple[N, N]]
        arr = Square(np.ones((4, 4)))
        assert arr.shape == (4, 4)

    def test_square_matrix_failure(self) -> None:
        Square = TypedNDArray[tuple[N, N]]
        with pytest.raises(ShapeError):
            Square(np.ones((3, 4)))


class TestLiteralAndTypeVarMix:
    def test_literal_and_typevar_reuse(self) -> None:
        Array = TypedNDArray[tuple[TWO, N, N]]
        arr = Array(np.ones((2, 3, 3)))
        assert arr.shape == (2, 3, 3)

    def test_literal_and_typevar_reuse_failure(self) -> None:
        Array = TypedNDArray[tuple[TWO, N, N]]
        with pytest.raises(ShapeError):
            Array(np.ones((2, 3, 4)))


class TestMethodLevelIsolation:
    def test_method_level_does_not_persist(self) -> None:
        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[M]]) -> TypedNDArray[tuple[M]]:
                return x

        m = Model()

        a = TypedNDArray[tuple[FIVE]]([1, 2, 3, 4, 5])
        b = TypedNDArray[tuple[THREE]]([1, 2, 3])

        m.foo(a)
        # Should not conflict with previous call
        m.foo(b)


class TestReturnValidation:
    def test_return_shape_violation(self) -> None:
        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[N]]) -> TypedNDArray[tuple[N]]:
                return TypedNDArray([1, 2, 3])  # always size 3

        m = Model()

        a = TypedNDArray[tuple[FIVE]]([1, 2, 3, 4, 5])

        with pytest.raises(DimensionError):
            m.foo(a)


class TestZeroDimIteration:
    def test_zero_dim_not_iterable(self) -> None:
        Scalar = TypedNDArray[tuple[()]]
        arr = Scalar(5)

        with pytest.raises(TypeError):
            iter(arr)


class TestSubclassPreservation:
    def test_view_preserves_type(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr = Array3([1, 2, 3])
        view = arr.view()

        assert isinstance(view, TypedNDArray)

    def test_slice_preserves_type(self) -> None:
        Array5 = TypedNDArray[tuple[FIVE]]
        arr = Array5([1, 2, 3, 4, 5])
        sliced = arr[1:]

        assert isinstance(sliced, TypedNDArray)


class TestDTypeTypeVar:
    def test_dtype_typevar_binding(self) -> None:
        T = TypeVar("T", bound=np.generic)

        Array = TypedNDArray[tuple[THREE], np.dtype[T]]
        Array_i32 = Array[np.int32]

        arr = Array_i32([1, 2, 3])
        assert arr.dtype == np.int32
