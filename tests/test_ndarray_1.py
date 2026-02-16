"""
Tests for TypedNDArray core functionality - 1
"""
# tests/test_ndarray_1.py

# pyright: reportPrivateUsage = false
# pyright: reportGeneralTypeIssues = false
# pyright: reportInvalidTypeArguments = false

from typing import TypeVar

import numpy as np
import pytest

from typed_numpy._typed.helpers import FIVE, FOUR, THREE, TWO
from typed_numpy._typed.ndarray import (
    RankError,
    ShapeError,
    TypedNDArray,
    _Shape,
    _validate_shape,
)

N = TypeVar("N", bound=int, default=int)
M = TypeVar("M", bound=int, default=int)


class TestShapeValidation:
    """Tests for shape validation."""

    def test_validate_shape_success(self) -> None:
        expected = (3, 4, 5)
        actual = (3, 4, 5)
        _validate_shape(expected, actual)  # Should not raise

    def test_validate_shape_rank_mismatch(self) -> None:
        expected = (3, 4)
        actual = (3, 4, 5)
        with pytest.raises(RankError):
            _validate_shape(expected, actual)

    def test_validate_shape_dimension_mismatch(self) -> None:
        expected = (3, 4, 5)
        actual = (3, 4, 6)
        with pytest.raises(ShapeError):
            _validate_shape(expected, actual)


class TestTypedNDArrayCreation:
    """Tests for TypedNDArray creation and initialisation."""

    def test_create_from_list(self) -> None:
        arr = TypedNDArray([1, 2, 3])
        assert isinstance(arr, TypedNDArray)
        assert arr.shape == (3,)

    def test_create_with_dtype(self) -> None:
        arr = TypedNDArray[_Shape, np.dtype[np.float32]]([1, 2, 3])
        assert arr.dtype == np.float32


class TestTypedNDArrayClassGetitem:
    """Tests for TypedNDArray type specification syntax."""

    def test_simple_shape(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        arr = Array3([1, 2, 3])
        assert arr.shape == (3,)

    def test_shape_with_dtype(self) -> None:
        Array2x2 = TypedNDArray[tuple[TWO, TWO], np.dtype[np.float32]]
        arr = Array2x2([[1.0, 2.0], [3.0, 4.0]])
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32

    def test_shape_with_int(self) -> None:
        Array5 = TypedNDArray[tuple[FIVE]]
        arr = Array5([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_multidimensional_shape(self) -> None:
        Array3x4 = TypedNDArray[tuple[THREE, FOUR]]
        arr = Array3x4(np.ones((3, 4)))
        assert arr.shape == (3, 4)

    def test_invalid_shape_raises(self) -> None:
        Array3 = TypedNDArray[tuple[THREE]]
        with pytest.raises(ShapeError):
            Array3([1, 2, 3, 4])


class TestTypedNDArrayWithTypeVars:
    """Tests for TypedNDArray with TypeVar dimensions."""

    def test_typevar_in_shape(self) -> None:
        ArrayN = TypedNDArray[tuple[N]]
        arr = ArrayN([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_multiple_typevars(self) -> None:
        ArrayMxN = TypedNDArray[tuple[M, N]]
        arr = ArrayMxN([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_mixed_concrete_and_typevar(self) -> None:
        Array3xN = TypedNDArray[tuple[THREE, N]]
        arr = Array3xN([[1, 2], [3, 4], [5, 6]])
        assert arr.shape == (3, 2)


class TestNDShapeDeferredBinding:
    """Tests for _NDShape deferred binding mechanism."""

    def test_bind_single_typevar(self) -> None:
        ArrayN = TypedNDArray[tuple[N]]
        Array5 = ArrayN[FIVE]
        arr = Array5([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_bind_single_typevar_wrong_size(self) -> None:
        ArrayN = TypedNDArray[tuple[N]]
        Array5 = ArrayN[FIVE]
        with pytest.raises(ShapeError):
            Array5([1, 2, 3])

    def test_bind_multiple_typevars(self) -> None:
        ArrayMxN = TypedNDArray[tuple[M, N]]
        Array2x3 = ArrayMxN[TWO, THREE]
        arr = Array2x3([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_bind_partial_typevars(self) -> None:
        ArrayMxN = TypedNDArray[tuple[M, N]]
        Array2xN = ArrayMxN[TWO, N]
        Array2x3 = Array2xN[THREE]
        arr = Array2x3([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_bind_too_many_dimensions(self) -> None:
        ArrayN = TypedNDArray[tuple[N]]
        with pytest.raises(TypeError):
            ArrayN[2, 3]  # type: ignore


class TestRepr:
    """Tests for string representation."""

    def test_repr_1d(self) -> None:
        arr = TypedNDArray([1, 2, 3])
        _arr = np.array([1, 2, 3])
        assert repr(arr) == repr(_arr)

    def test_repr_2d(self) -> None:
        arr = TypedNDArray([[1, 2], [3, 4]])
        _arr = np.array([[1, 2], [3, 4]])
        assert repr(arr) == repr(_arr)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_array(self) -> None:
        arr = TypedNDArray([])
        assert arr.shape == (0,)

    def test_scalar_array(self) -> None:
        arr = TypedNDArray(5)
        assert arr.shape == ()

    def test_complex_nested_structure(self) -> None:
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        arr = TypedNDArray(data)
        assert arr.shape == (2, 2, 2)

    def test_invalid_shape_spec_type(self) -> None:
        with pytest.raises(TypeError):
            # Should be tuple
            TypedNDArray[1, 2, 3]  # type: ignore
