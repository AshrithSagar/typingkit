"""
Helpers for TypedNDArray
=======
"""
# src/typed_numpy/_typed/helpers.py

from typing import Literal, TypeAlias, TypeVar

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

# TypeVars
Dim1 = TypeVar("Dim1", bound=int, default=int)
Dim2 = TypeVar("Dim2", bound=int, default=int)
Dim3 = TypeVar("Dim3", bound=int, default=int)
Dim4 = TypeVar("Dim4", bound=int, default=int)
K = TypeVar("K", bound=int, default=int)
L = TypeVar("L", bound=int, default=int)
M = TypeVar("M", bound=int, default=int)
N = TypeVar("N", bound=int, default=int)

# Shape type aliases
Shape1D: TypeAlias = tuple[Dim1]
"""A tuple representing a 1D shape, i.e., `(N,)`."""
Shape2D: TypeAlias = tuple[Dim1, Dim2]
"""A tuple representing a 2D shape, i.e., `(M, N)`."""
Shape3D: TypeAlias = tuple[Dim1, Dim2, Dim3]
"""A tuple representing a 3D shape, i.e., shape `(L, M, N)`."""
Shape4D: TypeAlias = tuple[Dim1, Dim2, Dim3, Dim4]
"""A tuple representing a 4D shape, i.e., shape `(K, L, M, N)`."""
ShapeND: TypeAlias = tuple[int, ...]
"""A tuple representing a ND shape, i.e., shape `(N, ...)`."""

# DType aliases
DType = TypeVar("DType", bound=np.dtype, default=np.dtype, covariant=True)


## Array type aliases

# Arrays based on dimensionality
Array1D: TypeAlias = TypedNDArray[Shape1D[Dim1], DType]
"""A `numpy.ndarray` of shape `(N,)` and dtype `DType`."""
Array2D: TypeAlias = TypedNDArray[Shape2D[Dim1, Dim2], DType]
"""A `numpy.ndarray` of shape `(M, N)` and dtype `DType`."""
Array3D: TypeAlias = TypedNDArray[Shape3D[Dim1, Dim2, Dim3], DType]
"""A `numpy.ndarray` of shape `(L, M, N)` and dtype `DType`."""
Array4D: TypeAlias = TypedNDArray[Shape4D[Dim1, Dim2, Dim3, Dim4], DType]
"""A `numpy.ndarray` of shape `(K, L, M, N)` and dtype `DType`."""
ArrayND: TypeAlias = TypedNDArray[ShapeND, DType]
"""A `numpy.ndarray` of shape `(N, ...)` and dtype `DType`."""

# 1D Arrays with specific small sizes
Array2: TypeAlias = TypedNDArray[tuple[TWO], DType]
"""A `numpy.ndarray` of shape `(2,)` and dtype `DType`."""
Array3: TypeAlias = TypedNDArray[tuple[THREE], DType]
"""A `numpy.ndarray` of shape `(3,)` and dtype `DType`."""
Array4: TypeAlias = TypedNDArray[tuple[FOUR], DType]
"""A `numpy.ndarray` of shape `(4,)` and dtype `DType`."""
ArrayN: TypeAlias = TypedNDArray[tuple[N], DType]  # Same as Array1D
"""A `numpy.ndarray` of shape `(N,)` and dtype `DType`."""

# 2D Arrays with specific small sizes
Array2x2: TypeAlias = TypedNDArray[tuple[TWO, TWO], DType]
"""A `numpy.ndarray` of shape `(2, 2)` and dtype `DType`."""
Array2x3: TypeAlias = TypedNDArray[tuple[TWO, THREE], DType]
"""A `numpy.ndarray` of shape `(2, 3)` and dtype `DType`."""
Array2x4: TypeAlias = TypedNDArray[tuple[TWO, FOUR], DType]
"""A `numpy.ndarray` of shape `(2, 4)` and dtype `DType`."""
Array2xN: TypeAlias = TypedNDArray[tuple[TWO, N], DType]
"""A `numpy.ndarray` of shape `(2, N)` and dtype `DType`."""

Array3x2: TypeAlias = TypedNDArray[tuple[THREE, TWO], DType]
"""A `numpy.ndarray` of shape `(3, 2)` and dtype `DType`."""
Array3x3: TypeAlias = TypedNDArray[tuple[THREE, THREE], DType]
"""A `numpy.ndarray` of shape `(3, 3)` and dtype `DType`."""
Array3x4: TypeAlias = TypedNDArray[tuple[THREE, FOUR], DType]
"""A `numpy.ndarray` of shape `(3, 4)` with the default `dtype."""
Array3xN: TypeAlias = TypedNDArray[tuple[THREE, N], DType]
"""A `numpy.ndarray` of shape `(3, N)` and dtype `DType`."""

Array4x2: TypeAlias = TypedNDArray[tuple[FOUR, TWO], DType]
"""A `numpy.ndarray` of shape `(4, 2)` and dtype `DType`."""
Array4x3: TypeAlias = TypedNDArray[tuple[FOUR, THREE], DType]
"""A `numpy.ndarray` of shape `(4, 3)` and dtype `DType`."""
Array4x4: TypeAlias = TypedNDArray[tuple[FOUR, FOUR], DType]
"""A `numpy.ndarray` of shape `(4, 4)` with the default `dtype."""
Array4xN: TypeAlias = TypedNDArray[tuple[FOUR, N], DType]
"""A `numpy.ndarray` of shape `(4, N)` and dtype `DType`."""

ArrayNx2: TypeAlias = TypedNDArray[tuple[N, TWO], DType]
"""A `numpy.ndarray` of shape `(N, 2)` and dtype `DType`."""
ArrayNx3: TypeAlias = TypedNDArray[tuple[N, THREE], DType]
"""A `numpy.ndarray` of shape `(N, 3)` and dtype `DType`."""
ArrayNx4: TypeAlias = TypedNDArray[tuple[N, FOUR], DType]
"""A `numpy.ndarray` of shape `(N, 4)` and dtype `DType`."""

# 3D Arrays with specific small sizes
Array2x2x2: TypeAlias = TypedNDArray[tuple[TWO, TWO, TWO], DType]
"""A `numpy.ndarray` of shape `(2, 2, 2)` and dtype `DType`."""
Array2x2x3: TypeAlias = TypedNDArray[tuple[TWO, TWO, THREE], DType]
"""A `numpy.ndarray` of shape `(2, 2, 3)` and dtype `DType`."""
Array2x2x4: TypeAlias = TypedNDArray[tuple[TWO, TWO, FOUR], DType]
"""A `numpy.ndarray` of shape `(2, 2, 4)` and dtype `DType`."""
Array2x2xN: TypeAlias = TypedNDArray[tuple[TWO, TWO, N], DType]
"""A `numpy.ndarray` of shape `(2, 2, N)` and dtype `DType`."""

Array2x3x2: TypeAlias = TypedNDArray[tuple[TWO, THREE, TWO], DType]
"""A `numpy.ndarray` of shape `(2, 3, 2)` and dtype `DType`."""
Array2x3x3: TypeAlias = TypedNDArray[tuple[TWO, THREE, THREE], DType]
"""A `numpy.ndarray` of shape `(2, 3, 3)` and dtype `DType`."""
Array2x3x4: TypeAlias = TypedNDArray[tuple[TWO, THREE, FOUR], DType]
"""A `numpy.ndarray` of shape `(2, 3, 4)` and dtype `DType`."""
Array2x3xN: TypeAlias = TypedNDArray[tuple[TWO, THREE, N], DType]
"""A `numpy.ndarray` of shape `(2, 3, N)` and dtype `DType`."""

Array2x4x2: TypeAlias = TypedNDArray[tuple[TWO, FOUR, TWO], DType]
"""A `numpy.ndarray` of shape `(2, 4, 2)` and dtype `DType`."""
Array2x4x3: TypeAlias = TypedNDArray[tuple[TWO, FOUR, THREE], DType]
"""A `numpy.ndarray` of shape `(2, 4, 3)` and dtype `DType`."""
Array2x4x4: TypeAlias = TypedNDArray[tuple[TWO, FOUR, FOUR], DType]
"""A `numpy.ndarray` of shape `(2, 4, 4)` and dtype `DType`."""
Array2x4xN: TypeAlias = TypedNDArray[tuple[TWO, FOUR, N], DType]
"""A `numpy.ndarray` of shape `(2, 4, N)` and dtype `DType`."""

Array2xNx2: TypeAlias = TypedNDArray[tuple[TWO, N, TWO], DType]
"""A `numpy.ndarray` of shape `(2, N, 2)` and dtype `DType`."""
Array2xNx3: TypeAlias = TypedNDArray[tuple[TWO, N, THREE], DType]
"""A `numpy.ndarray` of shape `(2, N, 3)` and dtype `DType`."""
Array2xNx4: TypeAlias = TypedNDArray[tuple[TWO, N, FOUR], DType]
"""A `numpy.ndarray` of shape `(2, N, 4)` and dtype `DType`."""

Array3x2x2: TypeAlias = TypedNDArray[tuple[THREE, TWO, TWO], DType]
"""A `numpy.ndarray` of shape `(3, 2, 2)` and dtype `DType`."""
Array3x2x3: TypeAlias = TypedNDArray[tuple[THREE, TWO, THREE], DType]
"""A `numpy.ndarray` of shape `(3, 2, 3)` and dtype `DType`."""
Array3x2x4: TypeAlias = TypedNDArray[tuple[THREE, TWO, FOUR], DType]
"""A `numpy.ndarray` of shape `(3, 2, 4)` and dtype `DType`."""
Array3x2xN: TypeAlias = TypedNDArray[tuple[THREE, TWO, N], DType]
"""A `numpy.ndarray` of shape `(3, 2, N)` and dtype `DType`."""

Array3x3x2: TypeAlias = TypedNDArray[tuple[THREE, THREE, TWO], DType]
"""A `numpy.ndarray` of shape `(3, 3, 2)` and dtype `DType`."""
Array3x3x3: TypeAlias = TypedNDArray[tuple[THREE, THREE, THREE], DType]
"""A `numpy.ndarray` of shape `(3, 3, 3)` and dtype `DType`."""
Array3x3x4: TypeAlias = TypedNDArray[tuple[THREE, THREE, FOUR], DType]
"""A `numpy.ndarray` of shape `(3, 3, 4)` and dtype `DType`."""
Array3x3xN: TypeAlias = TypedNDArray[tuple[THREE, THREE, N], DType]
"""A `numpy.ndarray` of shape `(3, 3, N)` and dtype `DType`."""

Array3x4x2: TypeAlias = TypedNDArray[tuple[THREE, FOUR, TWO], DType]
"""A `numpy.ndarray` of shape `(3, 4, 2)` and dtype `DType`."""
Array3x4x3: TypeAlias = TypedNDArray[tuple[THREE, FOUR, THREE], DType]
"""A `numpy.ndarray` of shape `(3, 4, 3)` and dtype `DType`."""
Array3x4x4: TypeAlias = TypedNDArray[tuple[THREE, FOUR, FOUR], DType]
"""A `numpy.ndarray` of shape `(3, 4, 4)` and dtype `DType`."""
Array3x4xN: TypeAlias = TypedNDArray[tuple[THREE, FOUR, N], DType]
"""A `numpy.ndarray` of shape `(3, 4, N)` and dtype `DType`."""

Array3xNx2: TypeAlias = TypedNDArray[tuple[THREE, N, TWO], DType]
"""A `numpy.ndarray` of shape `(3, N, 2)` and dtype `DType`."""
Array3xNx3: TypeAlias = TypedNDArray[tuple[THREE, N, THREE], DType]
"""A `numpy.ndarray` of shape `(3, N, 3)` and dtype `DType`."""
Array3xNx4: TypeAlias = TypedNDArray[tuple[THREE, N, FOUR], DType]
"""A `numpy.ndarray` of shape `(3, N, 4)` and dtype `DType`."""

Array4x2x2: TypeAlias = TypedNDArray[tuple[FOUR, TWO, TWO], DType]
"""A `numpy.ndarray` of shape `(4, 2, 2)` and dtype `DType`."""
Array4x2x3: TypeAlias = TypedNDArray[tuple[FOUR, TWO, THREE], DType]
"""A `numpy.ndarray` of shape `(4, 2, 3)` and dtype `DType`."""
Array4x2x4: TypeAlias = TypedNDArray[tuple[FOUR, TWO, FOUR], DType]
"""A `numpy.ndarray` of shape `(4, 2, 4)` and dtype `DType`."""
Array4x2xN: TypeAlias = TypedNDArray[tuple[FOUR, TWO, N], DType]
"""A `numpy.ndarray` of shape `(4, 2, N)` and dtype `DType`."""

Array4x3x2: TypeAlias = TypedNDArray[tuple[FOUR, THREE, TWO], DType]
"""A `numpy.ndarray` of shape `(4, 3, 2)` and dtype `DType`."""
Array4x3x3: TypeAlias = TypedNDArray[tuple[FOUR, THREE, THREE], DType]
"""A `numpy.ndarray` of shape `(4, 3, 3)` and dtype `DType`."""
Array4x3x4: TypeAlias = TypedNDArray[tuple[FOUR, THREE, FOUR], DType]
"""A `numpy.ndarray` of shape `(4, 3, 4)` and dtype `DType`."""
Array4x3xN: TypeAlias = TypedNDArray[tuple[FOUR, THREE, N], DType]
"""A `numpy.ndarray` of shape `(4, 3, N)` and dtype `DType`."""

Array4x4x2: TypeAlias = TypedNDArray[tuple[FOUR, FOUR, TWO], DType]
"""A `numpy.ndarray` of shape `(4, 4, 2)` and dtype `DType`."""
Array4x4x3: TypeAlias = TypedNDArray[tuple[FOUR, FOUR, THREE], DType]
"""A `numpy.ndarray` of shape `(4, 4, 3)` and dtype `DType`."""
Array4x4x4: TypeAlias = TypedNDArray[tuple[FOUR, FOUR, FOUR], DType]
"""A `numpy.ndarray` of shape `(4, 4, 4)` and dtype `DType`."""
Array4x4xN: TypeAlias = TypedNDArray[tuple[FOUR, FOUR, N], DType]
"""A `numpy.ndarray` of shape `(4, 4, N)` and dtype `DType`."""

Array4xNx2: TypeAlias = TypedNDArray[tuple[FOUR, N, TWO], DType]
"""A `numpy.ndarray` of shape `(4, N, 2)` and dtype `DType`."""
Array4xNx3: TypeAlias = TypedNDArray[tuple[FOUR, N, THREE], DType]
"""A `numpy.ndarray` of shape `(4, N, 3)` and dtype `DType`."""
Array4xNx4: TypeAlias = TypedNDArray[tuple[FOUR, N, FOUR], DType]
"""A `numpy.ndarray` of shape `(4, N, 4)` and dtype `DType`."""

ArrayNx2x2: TypeAlias = TypedNDArray[tuple[N, TWO, TWO], DType]
"""A `numpy.ndarray` of shape `(N, 2, 2)` and dtype `DType`."""
ArrayNx2x3: TypeAlias = TypedNDArray[tuple[N, TWO, THREE], DType]
"""A `numpy.ndarray` of shape `(N, 2, 3)` and dtype `DType`."""
ArrayNx2x4: TypeAlias = TypedNDArray[tuple[N, TWO, FOUR], DType]
"""A `numpy.ndarray` of shape `(N, 2, 4)` and dtype `DType`."""
ArrayNx3x2: TypeAlias = TypedNDArray[tuple[N, THREE, TWO], DType]
"""A `numpy.ndarray` of shape `(N, 3, 2)` and dtype `DType`."""
ArrayNx3x3: TypeAlias = TypedNDArray[tuple[N, THREE, THREE], DType]
"""A `numpy.ndarray` of shape `(N, 3, 3)` and dtype `DType`."""
ArrayNx3x4: TypeAlias = TypedNDArray[tuple[N, THREE, FOUR], DType]
"""A `numpy.ndarray` of shape `(N, 3, 4)` and dtype `DType`."""
ArrayNx4x2: TypeAlias = TypedNDArray[tuple[N, FOUR, TWO], DType]
"""A `numpy.ndarray` of shape `(N, 4, 2)` and dtype `DType`."""
ArrayNx4x3: TypeAlias = TypedNDArray[tuple[N, FOUR, THREE], DType]
"""A `numpy.ndarray` of shape `(N, 4, 3)` and dtype `DType`."""
ArrayNx4x4: TypeAlias = TypedNDArray[tuple[N, FOUR, FOUR], DType]
"""A `numpy.ndarray` of shape `(N, 4, 4)` and dtype `DType`."""
