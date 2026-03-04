"""
Typed NumPy
=======
"""
# src/typingkit/numpy/__init__.py

import numpy

from typingkit.numpy._typed import TypedNDArray, enforce_shapes

__all__ = [
    "numpy",
    "TypedNDArray",
    "enforce_shapes",
]
