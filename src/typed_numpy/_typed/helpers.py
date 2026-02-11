"""
Helpers for TypedNDArray
=======
"""
# src/typed_numpy/_typed/helpers.py

from typing import Literal, TypeAlias

import numpy as np

from typed_numpy._typed.ndarray import TypedNDArray

## Helpers

# Literal type aliases for small integers
ZERO: TypeAlias = Literal[0]
"""Literal type for the integer `0`."""
ONE: TypeAlias = Literal[1]
"""Literal type for the integer `1`."""
TWO: TypeAlias = Literal[2]
"""Literal type for the integer `2`."""
THREE: TypeAlias = Literal[3]
"""Literal type for the integer `3`."""
FOUR: TypeAlias = Literal[4]
"""Literal type for the integer `4`."""

# Shape type aliases
Shape1D: TypeAlias = tuple[int]
"""A tuple representing a 1D shape, i.e., `(N,)`."""
Shape2D: TypeAlias = tuple[int, int]
"""A tuple representing a 2D shape, i.e., `(M, N)`."""
Shape3D: TypeAlias = tuple[int, int, int]
"""A tuple representing a 3D shape, i.e., shape `(L, M, N)`."""
Shape4D: TypeAlias = tuple[int, int, int, int]
"""A tuple representing a 4D shape, i.e., shape `(K, L, M, N)`."""
ShapeND: TypeAlias = tuple[int, ...]
"""A tuple representing a ND shape, i.e., shape `(N, ...)`."""

# DType aliases
def_dtype: TypeAlias = np.double
"""The default `dtype` used throughout, mostly."""


## Array type aliases

# Arrays based on dimensionality
Array1D: TypeAlias = TypedNDArray[Shape1D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""
Array2D: TypeAlias = TypedNDArray[Shape2D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(M, N)` with the default `dtype`."""
Array3D: TypeAlias = TypedNDArray[Shape3D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(L, M, N)` with the default `dtype`."""
Array4D: TypeAlias = TypedNDArray[Shape4D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(K, L, M, N)` with the default `dtype`."""
ArrayND: TypeAlias = TypedNDArray[ShapeND, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, ...)` with the default `dtype`."""

# 1D Arrays with specific small sizes
Array2: TypeAlias = TypedNDArray[tuple[TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2,)` with the default `dtype`."""
Array3: TypeAlias = TypedNDArray[tuple[THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3,)` with the default `dtype`."""
Array4: TypeAlias = TypedNDArray[tuple[FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4,)` with the default `dtype`."""
ArrayN: TypeAlias = TypedNDArray[tuple[int], np.dtype[def_dtype]]  # Same as Array1D
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""

# 2D Arrays with specific small sizes
Array2x2: TypeAlias = TypedNDArray[tuple[TWO, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 2)` with the default `dtype`."""
Array2x3: TypeAlias = TypedNDArray[tuple[TWO, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 3)` with the default `dtype`."""
Array2x4: TypeAlias = TypedNDArray[tuple[TWO, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 4)` with the default `dtype`."""
Array2xN: TypeAlias = TypedNDArray[tuple[TWO, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, N)` with the default `dtype`."""

Array3x2: TypeAlias = TypedNDArray[tuple[THREE, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 2)` with the default `dtype`."""
Array3x3: TypeAlias = TypedNDArray[tuple[THREE, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 3)` with the default `dtype`."""
Array3x4: TypeAlias = TypedNDArray[tuple[THREE, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 4)` with the default `dtype."""
Array3xN: TypeAlias = TypedNDArray[tuple[THREE, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, N)` with the default `dtype`."""

Array4x2: TypeAlias = TypedNDArray[tuple[FOUR, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 2)` with the default `dtype`."""
Array4x3: TypeAlias = TypedNDArray[tuple[FOUR, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 3)` with the default `dtype`."""
Array4x4: TypeAlias = TypedNDArray[tuple[FOUR, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 4)` with the default `dtype."""
Array4xN: TypeAlias = TypedNDArray[tuple[FOUR, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, N)` with the default `dtype`."""

ArrayNx2: TypeAlias = TypedNDArray[tuple[int, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 2)` with the default `dtype`."""
ArrayNx3: TypeAlias = TypedNDArray[tuple[int, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 3)` with the default `dtype`."""
ArrayNx4: TypeAlias = TypedNDArray[tuple[int, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 4)` with the default `dtype`."""

# 3D Arrays with specific small sizes
Array2x2x2: TypeAlias = TypedNDArray[tuple[TWO, TWO, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 2, 2)` with the default `dtype`."""
Array2x2x3: TypeAlias = TypedNDArray[tuple[TWO, TWO, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 2, 3)` with the default `dtype`."""
Array2x2x4: TypeAlias = TypedNDArray[tuple[TWO, TWO, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 2, 4)` with the default `dtype`."""
Array2x2xN: TypeAlias = TypedNDArray[tuple[TWO, TWO, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 2, N)` with the default `dtype`."""

Array2x3x2: TypeAlias = TypedNDArray[tuple[TWO, THREE, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 3, 2)` with the default `dtype`."""
Array2x3x3: TypeAlias = TypedNDArray[tuple[TWO, THREE, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 3, 3)` with the default `dtype`."""
Array2x3x4: TypeAlias = TypedNDArray[tuple[TWO, THREE, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 3, 4)` with the default `dtype`."""
Array2x3xN: TypeAlias = TypedNDArray[tuple[TWO, THREE, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 3, N)` with the default `dtype`."""

Array2x4x2: TypeAlias = TypedNDArray[tuple[TWO, FOUR, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 4, 2)` with the default `dtype`."""
Array2x4x3: TypeAlias = TypedNDArray[tuple[TWO, FOUR, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 4, 3)` with the default `dtype`."""
Array2x4x4: TypeAlias = TypedNDArray[tuple[TWO, FOUR, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 4, 4)` with the default `dtype`."""
Array2x4xN: TypeAlias = TypedNDArray[tuple[TWO, FOUR, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 4, N)` with the default `dtype`."""

Array2xNx2: TypeAlias = TypedNDArray[tuple[TWO, int, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, N, 2)` with the default `dtype`."""
Array2xNx3: TypeAlias = TypedNDArray[tuple[TWO, int, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, N, 3)` with the default `dtype`."""
Array2xNx4: TypeAlias = TypedNDArray[tuple[TWO, int, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, N, 4)` with the default `dtype`."""

Array3x2x2: TypeAlias = TypedNDArray[tuple[THREE, TWO, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 2, 2)` with the default `dtype`."""
Array3x2x3: TypeAlias = TypedNDArray[tuple[THREE, TWO, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 2, 3)` with the default `dtype`."""
Array3x2x4: TypeAlias = TypedNDArray[tuple[THREE, TWO, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 2, 4)` with the default `dtype`."""
Array3x2xN: TypeAlias = TypedNDArray[tuple[THREE, TWO, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 2, N)` with the default `dtype`."""

Array3x3x2: TypeAlias = TypedNDArray[tuple[THREE, THREE, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 3, 2)` with the default `dtype`."""
Array3x3x3: TypeAlias = TypedNDArray[tuple[THREE, THREE, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 3, 3)` with the default `dtype`."""
Array3x3x4: TypeAlias = TypedNDArray[tuple[THREE, THREE, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 3, 4)` with the default `dtype`."""
Array3x3xN: TypeAlias = TypedNDArray[tuple[THREE, THREE, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 3, N)` with the default `dtype`."""

Array3x4x2: TypeAlias = TypedNDArray[tuple[THREE, FOUR, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 4, 2)` with the default `dtype`."""
Array3x4x3: TypeAlias = TypedNDArray[tuple[THREE, FOUR, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 4, 3)` with the default `dtype`."""
Array3x4x4: TypeAlias = TypedNDArray[tuple[THREE, FOUR, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 4, 4)` with the default `dtype`."""
Array3x4xN: TypeAlias = TypedNDArray[tuple[THREE, FOUR, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, 4, N)` with the default `dtype`."""

Array3xNx2: TypeAlias = TypedNDArray[tuple[THREE, int, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, N, 2)` with the default `dtype`."""
Array3xNx3: TypeAlias = TypedNDArray[tuple[THREE, int, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, N, 3)` with the default `dtype`."""
Array3xNx4: TypeAlias = TypedNDArray[tuple[THREE, int, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(3, N, 4)` with the default `dtype`."""

Array4x2x2: TypeAlias = TypedNDArray[tuple[FOUR, TWO, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 2, 2)` with the default `dtype`."""
Array4x2x3: TypeAlias = TypedNDArray[tuple[FOUR, TWO, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 2, 3)` with the default `dtype`."""
Array4x2x4: TypeAlias = TypedNDArray[tuple[FOUR, TWO, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 2, 4)` with the default `dtype`."""
Array4x2xN: TypeAlias = TypedNDArray[tuple[FOUR, TWO, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 2, N)` with the default `dtype`."""

Array4x3x2: TypeAlias = TypedNDArray[tuple[FOUR, THREE, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 3, 2)` with the default `dtype`."""
Array4x3x3: TypeAlias = TypedNDArray[tuple[FOUR, THREE, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 3, 3)` with the default `dtype`."""
Array4x3x4: TypeAlias = TypedNDArray[tuple[FOUR, THREE, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 3, 4)` with the default `dtype`."""
Array4x3xN: TypeAlias = TypedNDArray[tuple[FOUR, THREE, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 3, N)` with the default `dtype`."""

Array4x4x2: TypeAlias = TypedNDArray[tuple[FOUR, FOUR, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 4, 2)` with the default `dtype`."""
Array4x4x3: TypeAlias = TypedNDArray[tuple[FOUR, FOUR, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 4, 3)` with the default `dtype`."""
Array4x4x4: TypeAlias = TypedNDArray[tuple[FOUR, FOUR, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 4, 4)` with the default `dtype`."""
Array4x4xN: TypeAlias = TypedNDArray[tuple[FOUR, FOUR, int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, 4, N)` with the default `dtype`."""

Array4xNx2: TypeAlias = TypedNDArray[tuple[FOUR, int, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, N, 2)` with the default `dtype`."""
Array4xNx3: TypeAlias = TypedNDArray[tuple[FOUR, int, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, N, 3)` with the default `dtype`."""
Array4xNx4: TypeAlias = TypedNDArray[tuple[FOUR, int, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(4, N, 4)` with the default `dtype`."""

ArrayNx2x2: TypeAlias = TypedNDArray[tuple[int, TWO, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 2, 2)` with the default `dtype`."""
ArrayNx2x3: TypeAlias = TypedNDArray[tuple[int, TWO, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 2, 3)` with the default `dtype`."""
ArrayNx2x4: TypeAlias = TypedNDArray[tuple[int, TWO, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 2, 4)` with the default `dtype`."""
ArrayNx3x2: TypeAlias = TypedNDArray[tuple[int, THREE, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 3, 2)` with the default `dtype`."""
ArrayNx3x3: TypeAlias = TypedNDArray[tuple[int, THREE, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 3, 3)` with the default `dtype`."""
ArrayNx3x4: TypeAlias = TypedNDArray[tuple[int, THREE, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 3, 4)` with the default `dtype`."""
ArrayNx4x2: TypeAlias = TypedNDArray[tuple[int, FOUR, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 4, 2)` with the default `dtype`."""
ArrayNx4x3: TypeAlias = TypedNDArray[tuple[int, FOUR, THREE], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 4, 3)` with the default `dtype`."""
ArrayNx4x4: TypeAlias = TypedNDArray[tuple[int, FOUR, FOUR], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 4, 4)` with the default `dtype`."""
