"""
Extended Tests for TypedNDArray - 3
"""
# tests/test_ndarray_3.py

# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false

from typing import Literal, TypeVar

import numpy as np
import pytest

from typingkit.numpy._typed.context import enforce_shapes
from typingkit.numpy._typed.helpers import FIVE, THREE, TWO
from typingkit.numpy._typed.ndarray import DimensionError, ShapeError, TypedNDArray

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
    def test_typevar_return_violation(self) -> None:
        """Return shape uses the same TypeVar as input; should raise if mismatched."""

        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[N]]) -> TypedNDArray[tuple[N]]:
                # Return always size 3, input will be size 5
                return TypedNDArray([1, 2, 3])

        m = Model()
        a = TypedNDArray[tuple[FIVE]]([1, 2, 3, 4, 5])

        with pytest.raises(DimensionError):
            m.foo(a)

    def test_literal_return_violation(self) -> None:
        """Return shape is a Literal; should raise if actual shape differs."""

        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[N]]) -> TypedNDArray[tuple[TWO]]:
                return TypedNDArray([1, 2, 3])  # Always size 3

        m = Model()
        a = TypedNDArray[tuple[FIVE]]([1, 2, 3, 4, 5])

        with pytest.raises(DimensionError):
            m.foo(a)

    def test_typevar_return_pass(self) -> None:
        """Return shape matches TypeVar-bound input; should pass."""

        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[N]]) -> TypedNDArray[tuple[N]]:
                return TypedNDArray([1, 2, 3, 4, 5])  # same as input

        m = Model()
        a = TypedNDArray[tuple[FIVE]]([1, 2, 3, 4, 5])
        result = m.foo(a)
        assert result.shape == a.shape

    def test_literal_return_pass(self) -> None:
        """Return shape matches Literal; should pass."""

        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[N]]) -> TypedNDArray[tuple[THREE]]:
                return TypedNDArray([1, 2, 3])

        m = Model()
        a = TypedNDArray[tuple[THREE]]([1, 2, 3])
        result = m.foo(a)
        assert result.shape == (3,)


class TestArgumentValidation:
    def test_literal_input_violation(self) -> None:
        """Input shape mismatches a Literal; should raise DimensionError."""

        class Model:
            @enforce_shapes
            def foo(
                self, x: TypedNDArray[tuple[Literal[4]]]
            ) -> TypedNDArray[tuple[Literal[4]]]:
                return x

        m = Model()
        a = TypedNDArray[tuple[Literal[5]]]([1, 2, 3, 4, 5])

        with pytest.raises(DimensionError):
            m.foo(a)  # type: ignore  # This type mismatch is intended

    def test_typevar_input_pass(self) -> None:
        """Input shape matches TypeVar; should pass."""

        class Model:
            @enforce_shapes
            def foo(self, x: TypedNDArray[tuple[N]]) -> TypedNDArray[tuple[N]]:
                return x

        m = Model()
        a = TypedNDArray[tuple[FIVE]]([1, 2, 3, 4, 5])
        result = m.foo(a)
        assert result.shape == a.shape

    def test_literal_input_pass(self) -> None:
        """Input shape matches Literal; should pass."""

        class Model:
            @enforce_shapes
            def foo(
                self, x: TypedNDArray[tuple[Literal[3]]]
            ) -> TypedNDArray[tuple[Literal[3]]]:
                return x

        m = Model()
        a = TypedNDArray[tuple[THREE]]([1, 2, 3])
        result = m.foo(a)
        assert result.shape == (3,)

    def test_multiple_args_typevars(self) -> None:
        """Function with multiple TypeVar arguments; mismatched shapes raise DimensionError."""

        class Model:
            @enforce_shapes
            def foo(
                self, x: TypedNDArray[tuple[N]], y: TypedNDArray[tuple[N]]
            ) -> TypedNDArray[tuple[N]]:
                return x

        m = Model()
        x = TypedNDArray[tuple[FIVE]]([1, 2, 3, 4, 5])
        y_good = TypedNDArray[tuple[FIVE]]([10, 20, 30, 40, 50])
        y_bad = TypedNDArray[tuple[THREE]]([10, 20, 30])

        # Good call
        result = m.foo(x, y_good)
        assert result.shape == (5,)

        # Bad call
        with pytest.raises(DimensionError):
            m.foo(x, y_bad)

    def test_mixed_typevar_and_literal_args(self) -> None:
        """Function with TypeVar and Literal in arguments; should validate separately."""

        class Model:
            @enforce_shapes
            def foo(
                self, x: TypedNDArray[tuple[N]], y: TypedNDArray[tuple[THREE]]
            ) -> TypedNDArray[tuple[N]]:
                return x

        m = Model()
        x_good = TypedNDArray[tuple[FIVE]]([1, 2, 3, 4, 5])
        y_good = TypedNDArray[tuple[THREE]]([10, 20, 30])
        y_bad = TypedNDArray[tuple[FIVE]]([10, 20, 30, 40, 50])

        # Good call
        result = m.foo(x_good, y_good)
        assert result.shape == (5,)

        # Bad call: Literal mismatch
        with pytest.raises(DimensionError):
            m.foo(x_good, y_bad)  # type: ignore  # This type mismatch is intended


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

        arr = Array_i32([1, 2, 3], dtype=np.int32)
        assert arr.dtype == np.int32
