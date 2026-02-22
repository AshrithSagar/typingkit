"""
Extended Tests for TypedNDArray - 2
"""
# tests/test_ndarray_2.py

# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false

from typing import Literal, TypeVar

import numpy as np
import pytest

from typed_numpy._typed.helpers import FIVE, FOUR, ONE, THREE, TWO
from typed_numpy._typed.ndarray import DTypeError, RankError, ShapeError, TypedNDArray

N = TypeVar("N", bound=int, default=int)
M = TypeVar("M", bound=int, default=int)
K = TypeVar("K", bound=int, default=int)


class TestComplexShapeSpecifications:
    """Tests for complex shape specifications."""

    def test_3d_array_with_literals(self) -> None:
        Array2x3x4 = TypedNDArray[tuple[TWO, THREE, FOUR]]
        arr = Array2x3x4(np.ones((2, 3, 4)))
        assert arr.shape == (2, 3, 4)

    def test_4d_array(self) -> None:
        Array2x2x2x2 = TypedNDArray[tuple[TWO, TWO, TWO, TWO]]
        arr = Array2x2x2x2(np.ones((2, 2, 2, 2)))
        assert arr.shape == (2, 2, 2, 2)

    def test_high_dimensional_array(self) -> None:
        # 6D array
        Array2x2x2x2x2x2 = TypedNDArray[tuple[TWO, TWO, TWO, TWO, TWO, TWO]]
        arr = Array2x2x2x2x2x2(np.ones((2, 2, 2, 2, 2, 2)))
        assert arr.shape == (2, 2, 2, 2, 2, 2)


class TestDTypeSpecification:
    """Tests for dtype specifications."""

    def test_int32_dtype(self) -> None:
        Array3 = TypedNDArray[tuple[THREE], np.dtype[np.int32]]
        arr = Array3([1, 2, 3], dtype=np.int32)
        assert arr.dtype == np.int32

    def test_float64_dtype(self) -> None:
        Array3 = TypedNDArray[tuple[THREE], np.dtype[np.float64]]
        arr = Array3([1, 2, 3], dtype=np.float64)
        assert arr.dtype == np.float64

    def test_complex_dtype(self) -> None:
        Array3 = TypedNDArray[tuple[THREE], np.dtype[np.complex128]]
        arr = Array3([1 + 2j, 3 + 4j, 5 + 6j])
        assert arr.dtype == np.complex128

    def test_bool_dtype(self) -> None:
        Array3 = TypedNDArray[tuple[THREE], np.dtype[np.bool_]]
        arr = Array3([True, False, True])
        assert arr.dtype == np.bool_

    def test_dtype_override_in_call(self) -> None:
        Array3 = TypedNDArray[tuple[THREE], np.dtype[np.int32]]
        # Override dtype in call
        with pytest.raises(DTypeError):
            Array3([1.5, 2.5, 3.5], dtype=np.float64)


class TestShapeValidationErrors:
    """Tests for various shape validation error scenarios."""

    def test_wrong_first_dimension(self) -> None:
        Array3x4 = TypedNDArray[tuple[THREE, FOUR]]
        with pytest.raises(ShapeError):
            Array3x4(np.ones((2, 4)))

    def test_wrong_second_dimension(self) -> None:
        Array3x4 = TypedNDArray[tuple[THREE, FOUR]]
        with pytest.raises(ShapeError):
            Array3x4(np.ones((3, 5)))

    def test_wrong_middle_dimension(self) -> None:
        Array2x3x4 = TypedNDArray[tuple[TWO, THREE, FOUR]]
        with pytest.raises(ShapeError):
            Array2x3x4(np.ones((2, 5, 4)))

    def test_too_few_dimensions(self) -> None:
        Array3x4 = TypedNDArray[tuple[THREE, FOUR]]
        with pytest.raises(RankError):
            Array3x4([1, 2, 3])

    def test_too_many_dimensions(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        with pytest.raises(RankError):
            Array3([[1, 2, 3]])

    def test_multiple_dimension_mismatches(self) -> None:
        Array2x3x4 = TypedNDArray[tuple[TWO, THREE, FOUR]]
        with pytest.raises(ShapeError):
            Array2x3x4(np.ones((5, 6, 7)))


class TestDeferredBindingAdvanced:
    """Advanced tests for deferred binding mechanism."""

    def test_bind_with_literal_types(self) -> None:
        ArrayN = TypedNDArray[tuple[N]]
        Array5 = ArrayN[FIVE]
        arr = Array5([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_progressive_binding_three_levels(self) -> None:
        ArrayMxNxK = TypedNDArray[tuple[M, N, K]]

        # Bind M
        Array2xNxK = ArrayMxNxK[TWO, N, K]
        # Bind N
        Array2x3xK = Array2xNxK[THREE, K]
        # Bind K
        Array2x3x4 = Array2x3xK[FOUR]

        arr = Array2x3x4(np.ones((2, 3, 4)))
        assert arr.shape == (2, 3, 4)

    def test_bind_multiple_at_once(self) -> None:
        ArrayMxNxK = TypedNDArray[tuple[M, N, K]]
        Array2x3x4 = ArrayMxNxK[TWO, THREE, FOUR]
        arr = Array2x3x4(np.ones((2, 3, 4)))
        assert arr.shape == (2, 3, 4)

    def test_bind_skip_with_typevar(self) -> None:
        ArrayMxN = TypedNDArray[tuple[M, N]]
        # Bind M but leave N as TypeVar
        Array2xK = ArrayMxN[TWO, K]
        # Now bind K
        Array2x3 = Array2xK[THREE]
        arr = Array2x3([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_validation_after_binding(self) -> None:
        ArrayN = TypedNDArray[tuple[N]]
        Array5 = ArrayN[FIVE]

        # Should validate after binding
        with pytest.raises(ShapeError):
            Array5([1, 2, 3])  # Wrong size

    def test_repr_of_ndshape(self) -> None:
        ArrayMxN = TypedNDArray[tuple[M, N]]
        repr_str = repr(ArrayMxN)
        assert "TypedNDArray" in repr_str
        assert "tuple" in repr_str


class TestTypeVarDefaults:
    """Tests for TypeVar with default values."""

    def test_typevar_with_default(self) -> None:
        N = TypeVar("N", bound=int, default=FIVE)
        ArrayN = TypedNDArray[tuple[N]]
        # Without binding, TypeVar with default shouldn't validate
        arr = ArrayN([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_bind_partial_uses_defaults(self) -> None:
        M = TypeVar("M", bound=int, default=TWO)
        N = TypeVar("N", bound=int, default=THREE)
        ArrayMxN = TypedNDArray[tuple[M, N]]

        # Bind only M; N should use default
        Array5xN = ArrayMxN[FIVE]
        Array5x3 = Array5xN  # N uses default 3
        arr = Array5x3(np.ones((5, 3)))
        assert arr.shape == (5, 3)


class TestNumpyInteroperability:
    """Tests for interoperability with numpy functions."""

    def test_numpy_sum(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr = Array3([1, 2, 3])
        result = np.sum(arr)
        assert result == 6

    def test_numpy_mean(self) -> None:
        Array4 = TypedNDArray[tuple[FOUR]]
        arr = Array4([1, 2, 3, 4])
        result = np.mean(arr)
        assert result == 2.5

    def test_numpy_dot(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr1 = Array3([1, 2, 3])
        arr2 = Array3([4, 5, 6])
        result = np.dot(arr1, arr2)
        assert result == 32

    def test_numpy_matmul(self) -> None:
        Array2x3 = TypedNDArray[tuple[TWO, THREE]]
        Array3x2 = TypedNDArray[tuple[THREE, TWO]]

        a = Array2x3([[1, 2, 3], [4, 5, 6]])
        b = Array3x2([[7, 8], [9, 10], [11, 12]])

        result = np.matmul(a, b)
        assert result.shape == (2, 2)

    def test_numpy_concatenate(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr1 = Array3([1, 2, 3])
        arr2 = Array3([4, 5, 6])

        result = np.concatenate([arr1, arr2])
        assert result.shape == (6,)

    def test_numpy_stack(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr1 = Array3([1, 2, 3])
        arr2 = Array3([4, 5, 6])

        result = np.stack([arr1, arr2])
        assert result.shape == (2, 3)

    def test_numpy_where(self) -> None:
        Array5 = TypedNDArray[tuple[FIVE]]
        arr = Array5([1, 2, 3, 4, 5])
        result = np.where(arr > 3)
        assert len(result[0]) == 2

    def test_numpy_argmax(self) -> None:
        Array5 = TypedNDArray[tuple[FIVE]]
        arr = Array5([1, 5, 3, 2, 4])
        result = np.argmax(arr)
        assert result == 1


class TestArrayOperations:
    """Tests for standard array operations."""

    def test_addition(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr1 = Array3([1, 2, 3])
        arr2 = Array3([4, 5, 6])
        result = arr1 + arr2
        np.testing.assert_array_equal(result, [5, 7, 9])

    def test_multiplication(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr = Array3([1, 2, 3])
        result = arr * 2
        np.testing.assert_array_equal(result, [2, 4, 6])

    def test_subtraction(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr1 = Array3([5, 6, 7])
        arr2 = Array3([1, 2, 3])
        result = arr1 - arr2
        np.testing.assert_array_equal(result, [4, 4, 4])

    def test_division(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr = Array3([10, 20, 30], dtype=np.float64)
        result = arr / 2
        np.testing.assert_array_equal(result, [5, 10, 15])

    def test_power(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr = Array3([2, 3, 4])
        result = arr**2
        np.testing.assert_array_equal(result, [4, 9, 16])

    def test_indexing(self) -> None:
        Array5 = TypedNDArray[tuple[FIVE]]
        arr = Array5([10, 20, 30, 40, 50])
        assert arr[0] == 10
        assert arr[2] == 30
        assert arr[-1] == 50

    def test_slicing(self) -> None:
        Array5 = TypedNDArray[tuple[FIVE]]
        arr = Array5([10, 20, 30, 40, 50])
        sliced = arr[1:4]
        np.testing.assert_array_equal(sliced, [20, 30, 40])

    def test_boolean_indexing(self) -> None:
        Array5 = TypedNDArray[tuple[FIVE]]
        arr = Array5([1, 2, 3, 4, 5])
        mask = arr > 3
        result = arr[mask]
        np.testing.assert_array_equal(result, [4, 5])

    def test_2d_indexing(self) -> None:
        Array3x3 = TypedNDArray[tuple[THREE, THREE]]
        arr = Array3x3([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert arr[0, 0] == 1
        assert arr[1, 2] == 6
        assert arr[2, 2] == 9


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""

    def test_single_element_array(self) -> None:
        Array1 = TypedNDArray[tuple[ONE]]
        arr = Array1([42])
        assert arr.shape == (1,)
        assert arr[0] == 42

    def test_large_array(self) -> None:
        Array1000 = TypedNDArray[tuple[Literal[1000]]]
        arr = Array1000(np.arange(1000))
        assert arr.shape == (1000,)
        assert arr[999] == 999

    def test_zero_dimension_in_middle(self) -> None:
        # This is valid in numpy: array with shape (2, 0, 3)
        arr = TypedNDArray(np.zeros((2, 0, 3)))
        assert arr.shape == (2, 0, 3)

    def test_very_large_dimension(self) -> None:
        Array10000 = TypedNDArray[tuple[Literal[10000]]]
        arr = Array10000(np.zeros(10000))
        assert arr.shape == (10000,)


class TestErrorMessages:
    """Tests for clear and helpful error messages."""

    def test_rank_error_message_content(self) -> None:
        Array2x3 = TypedNDArray[tuple[TWO, THREE]]
        with pytest.raises(RankError) as exc_info:
            Array2x3([1, 2, 3])

        error_msg = str(exc_info.value)
        assert "Expected 2" in error_msg
        assert "got 1" in error_msg

    def test_shape_error_message_content(self) -> None:
        Array3x4 = TypedNDArray[tuple[THREE, FOUR]]
        with pytest.raises(ShapeError) as exc_info:
            Array3x4(np.ones((3, 5)))

        error_msg = str(exc_info.value)
        assert "Shape mismatch" in error_msg
        assert "expected (3, 4)" in error_msg
        assert "got (3, 5)" in error_msg

    def test_dimension_error_message_content(self) -> None:
        ArrayN = TypedNDArray[tuple[N]]

        with pytest.raises(TypeError) as exc_info:
            ArrayN[2, 3, 4]  # type: ignore

        error_msg = str(exc_info.value)
        assert "Too many arguments" in error_msg
        assert "actual 3, expected 1" in error_msg


class TestSpecialCases:
    """Tests for special numpy array features."""

    def test_structured_array_dtype(self) -> None:
        dt = np.dtype([("x", np.int32), ("y", np.float64)])
        arr = TypedNDArray([(1, 1.5), (2, 2.5)], dtype=dt)
        assert arr.shape == (2,)
        assert arr.dtype == dt

    def test_object_dtype(self) -> None:
        arr = TypedNDArray([1, "hello", [1, 2]], dtype=object)  # type: ignore  # ty: ignore[unused-ignore-comment]
        assert arr.dtype == object
        assert arr.shape == (3,)

    def test_unicode_dtype(self) -> None:
        arr = TypedNDArray(["hello", "world"], dtype="U10")
        assert arr.shape == (2,)

    def test_from_numpy_array(self) -> None:
        np_arr = np.array([1, 2, 3, 4, 5])
        typed_arr = TypedNDArray(np_arr)
        assert typed_arr.shape == (5,)
        np.testing.assert_array_equal(typed_arr, np_arr)

    def test_from_typed_to_typed(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr1 = Array3([1, 2, 3])
        arr2 = TypedNDArray(arr1)

        np.testing.assert_array_equal(arr1, arr2)
        assert arr2.shape == (3,)
