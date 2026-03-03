# Using `typed-numpy`

`typed-numpy` provides runtime shape validation for NumPy arrays with static type hints.
Define your array shapes using TypeVars and get both IDE autocomplete and runtime checks.

<details>

<summary>Imports</summary>

The following imports are assumed wherever required.

```python
from typing import Generic, Literal, TypeAlias, TypeVar, reveal_type

import numpy as np
import numpy.typing as npt

from typingkit._typed import TypedNDArray
from typingkit._typed.context import enforce_shapes
```

</details>

## Basic usage

```python
# Shape variables are just regular TypeVar's
N = TypeVar("N", bound=int, default=int)
M = TypeVar("M", bound=int, default=int)

# Create aliases for common array types
Vector = TypedNDArray[tuple[N]]
Matrix = TypedNDArray[tuple[M, N]]

v1 = Vector([1, 2, 3])  # Passes
reveal_type(v1)  # TypedNDArray[tuple[N], dtype[Any]]

v2 = Vector([4, 5, 6, 7])
# Also passes, since N is not bound, so only rank checks happen

v3 = TypedNDArray[tuple[int]]([[8, 9]])
# Fails at runtime: expected 1D array but passed 2D
# Raises: RankError: Rank mismatch: expected 1, got 2
```

`TypedNDArray` subclasses `numpy.ndarray`, so instances of `TypedNDArray` work wherever NumPy numpy arrays are expected.

## Progressive type binding

The real power comes from progressively binding TypeVars to create specialised types:

```python
DimSpace = TypeVar("DimSpace", bound=int, default=int)
"""TypeVar denoting dimension of the space"""

NumPoints = TypeVar("NumPoints", bound=int, default=int)
"""TypeVar denoting number of points"""

TwoD: TypeAlias = Literal[2]
ThreeD: TypeAlias = Literal[3]

# Generic aliases with unbound TypeVars
Point: TypeAlias = TypedNDArray[tuple[DimSpace]]
Vector: TypeAlias = TypedNDArray[tuple[DimSpace]]
RotationMatrix: TypeAlias = TypedNDArray[tuple[DimSpace, DimSpace]]
Points: TypeAlias = TypedNDArray[tuple[NumPoints, DimSpace]]

# Partially bound alias: NumPoints is still generic, but second dim is fixed to 2
ArrayNx2: TypeAlias = TypedNDArray[tuple[NumPoints, TwoD]]

# Usage:
arr1 = ArrayNx2([[1, 2], [3, 4], [5, 6]])  # 3x2 array => Passes
reveal_type(arr1)  # "TypedNDArray[tuple[int, Literal[2]], dtype[Any]]"

arr2 = ArrayNx2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3 array => Fails
# Raises: ShapeError: Shape mismatch: expected (None, 2), got (3, 3)
```

### Binding TypeVars in classes

You can bind TypeVars progressively through class hierarchies:

```python
class Curve(Generic[NumPoints, DimSpace]):
    def __init__(self, points: npt.ArrayLike) -> None:
        # Bind both TypeVars using bracket notation
        self.points = Points[NumPoints, DimSpace](points)
        # self.points: TypedNDArray[tuple[NumPoints@Curve, DimSpace@Curve], dtype[Any]]

class Curve2D(Curve[NumPoints, TwoD]):
    """DimSpace is now bound to Literal[2]"""
    pass
    # self.points: TypedNDArray[tuple[NumPoints@Curve, Literal[2]], dtype[Any]]


class Curve3D(Curve[NumPoints, ThreeD]):
    """DimSpace is now bound to Literal[3]"""
    pass
    # self.points: TypedNDArray[tuple[NumPoints@Curve, Literal[3]], dtype[Any]]


# Any Nx2 array-like passes in Curve2D:
curve_1 = Curve2D([[1, 2], [3, 4], [5, 6]])  # Passes
curve_2 = Curve2D(np.random.random((5, 2)))  # Passes

# Shape mismatches fail at runtime:
curve_3 = Curve2D([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# ShapeError: 3x3 doesn't match Nx2

curve_4 = Curve2D(np.random.random((5, 3)))
# ShapeError: 5x3 doesn't match Nx2

reveal_type(curve_4.points)  # "TypedNDArray[tuple[int, Literal[2]], dtype[Any]]"
```

**Note:** The runtime type is always `TypedNDArray`, but static type checkers (mypy, Pylance) show the specialized generic types.

## Common patterns

### Fixed shapes with Literal types

```python
Image256x256 = TypedNDArray[tuple[Literal[256], Literal[256], Literal[3]]]

img = Image256x256(np.random.random((256, 256, 3)))  # Passes
```

### Matching dimensions across arrays

```python
# Both vectors must have the same dimension N
@enforce_shapes
def dot_product(a: Vector[N], b: Vector[N]) -> float:
    return float(np.dot(a, b))
```

### Matrix-vector multiplication with shape constraints

```python
# N must match between matrix columns and vector length
@enforce_shapes
def mat_vec_multiply(
    matrix: Matrix[M, N],
    vector: Vector[N]
) -> Vector[M]:
    return Vector[M](matrix @ vector)
```
