"""
Typed NumPy core
=======
"""
# src/typingkit/numpy/_typed/__init__.py

from typingkit.numpy._typed.context import enforce_shapes
from typingkit.numpy._typed.ndarray import TypedNDArray

__all__ = [
    "TypedNDArray",
    "enforce_shapes",
]
