"""
Shape factory for TypedNDArray
=======
"""
# src/typed_numpy/_typed/factory.py

# pyright: reportPrivateUsage = false

from types import GenericAlias
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from typed_numpy._typed.ndarray import TypedNDArray, _AnyShape

_ShapeT = TypeVar("_ShapeT", bound=_AnyShape)
_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_NewScalarT = TypeVar("_NewScalarT", bound=np.generic)


class _ShapeBuilder(Generic[_ShapeT, _ScalarT]):
    __slots__ = ("_shape_spec", "_dtype_spec")

    def __init__(self, shape_spec: GenericAlias, dtype_spec: GenericAlias):
        self._shape_spec = shape_spec
        self._dtype_spec = dtype_spec

    def dtype(self, dtype: type[_NewScalarT]) -> "_ShapeBuilder[_ShapeT, _NewScalarT]":
        return _ShapeBuilder(self._shape_spec, GenericAlias(np.dtype, (dtype,)))

    def __call__(
        self, object: npt.ArrayLike
    ) -> TypedNDArray[_ShapeT, np.dtype[_ScalarT]]:
        dtype = self._dtype_spec.__args__[0]
        dtype = None if dtype is Any else dtype
        return TypedNDArray[self._shape_spec](object, dtype)  # type: ignore

    def __repr__(self) -> str:
        return f"TypedNDArrayFactory[{self._shape_spec.__args__}, {self._dtype_spec.__args__[0]}]"

    ## Convenience helpers

    @property
    def float64(self) -> "_ShapeBuilder[_ShapeT, np.float64]":
        return self.dtype(np.float64)

    @property
    def float32(self) -> "_ShapeBuilder[_ShapeT, np.float32]":
        return self.dtype(np.float32)

    @property
    def int64(self) -> "_ShapeBuilder[_ShapeT, np.int64]":
        return self.dtype(np.int64)


class _ShapeFactory:
    def __getitem__(self, dims: _ShapeT) -> _ShapeBuilder[_ShapeT, Any]:
        if not isinstance(dims, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
            dims = (dims,)

        shape_spec = GenericAlias(tuple, dims)
        dtype_spec = GenericAlias(np.dtype, (Any,))
        return _ShapeBuilder(shape_spec, dtype_spec)

    # Prefer [...] rather than (...), but this is also provided
    def __call__(self, dims: _ShapeT) -> _ShapeBuilder[_ShapeT, Any]:
        return self.__getitem__(dims)


Shaped = _ShapeFactory()
