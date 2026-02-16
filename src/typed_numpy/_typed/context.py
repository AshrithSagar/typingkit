"""
Context binding
=======
Manages TypeVar binding contexts for shape validation.
"""
# src/typed_numpy/_typed/context.py

# pyright: reportPrivateUsage = false

import inspect
from contextvars import ContextVar
from functools import wraps
from typing import (
    Any,
    Callable,
    Concatenate,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from typed_numpy._typed.ndarray import DimensionError, _NDShape

# Context variables

_class_typevar_context = ContextVar[dict[int, dict[TypeVar, int]]](
    "_class_typevar_context", default=dict[int, dict[TypeVar, int]]()
)
_method_typevar_context = ContextVar[dict[TypeVar, int]](
    "_method_typevar_context", default=dict[TypeVar, int]()
)
_active_class_context = ContextVar[dict[TypeVar, int]](
    "_active_class_context", default=dict[TypeVar, int]()
)

T = TypeVar("T")
P = ParamSpec("P")  # ParamSpec for function parameters
R = TypeVar("R")  # TypeVar for return type


def _extract_shape_typevars(annotation: Any) -> list[tuple[int, TypeVar]]:
    """Extract TypeVars from a TypedNDArray annotation with their dimension indices."""

    # _NDShape
    if isinstance(annotation, _NDShape):
        return [
            (idx, dim)
            for idx, dim in enumerate(get_args(annotation.shape_spec))
            if isinstance(dim, TypeVar)
        ]

    # GenericAlias
    origin = get_origin(annotation)
    if origin is None:
        return []

    args = get_args(annotation)
    if not args:
        return []

    shape_spec = args[0]
    if get_origin(shape_spec) is tuple:
        shape_dims = get_args(shape_spec)
        return [
            (idx, dim) for idx, dim in enumerate(shape_dims) if isinstance(dim, TypeVar)
        ]

    return []


def _is_class_level_typevar(typevar: TypeVar, owner_cls: type) -> bool:
    """Check if a TypeVar is bound at class level vs method level."""
    cls_params = getattr(owner_cls, "__parameters__", ())
    return typevar in cls_params


def _get_instance_class_context(instance: Any) -> dict[TypeVar, int]:
    """Get or create the class-level TypeVar binding context for an instance."""
    ctx = _class_typevar_context.get()
    instance_id = id(instance)
    if instance_id not in ctx:
        ctx = ctx.copy()
        ctx[instance_id] = {}
        _class_typevar_context.set(ctx)
    return ctx[instance_id]


def enforce_shapes(
    func: Callable[Concatenate[T, P], R],
) -> Callable[Concatenate[T, P], R]:
    """
    Decorator to automatically validate TypeVar shape bindings.
    - Class-level TypeVars (from Generic[T]) are bound per-instance, persist across calls
    - Method-level TypeVars are validated per-call only, local to single invocation
    - Validates both parameter and return types
    """

    @wraps(func)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        hints = get_type_hints(func, include_extras=True)
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        owner_cls = self.__class__
        class_context = _get_instance_class_context(self)
        method_context = dict[TypeVar, int]()

        # Validate inputs
        for param_name, param_value in bound_args.arguments.items():
            if param_name == "self":
                continue
            if param_name not in hints:
                continue

            annotation = hints[param_name]
            typevars = _extract_shape_typevars(annotation)
            if not typevars or not hasattr(param_value, "shape"):
                continue

            actual_shape = param_value.shape
            for dim_idx, typevar in typevars:
                if dim_idx >= len(actual_shape):
                    continue

                actual_dim = actual_shape[dim_idx]
                is_class_level = _is_class_level_typevar(typevar, owner_cls)
                if is_class_level:
                    if typevar in class_context:
                        expected_dim = class_context[typevar]
                        if actual_dim != expected_dim:
                            raise DimensionError(
                                f"In {func.__name__}(...), parameter `{param_name}`'s "
                                f"dimension {dim_idx} [{typevar}] "
                                f"expected {expected_dim} (class-level binding), "
                                f"got {actual_dim}"
                            )
                    else:
                        # First binding for this instance
                        class_context[typevar] = actual_dim
                else:
                    if typevar in method_context:
                        expected_dim = method_context[typevar]
                        if actual_dim != expected_dim:
                            raise DimensionError(
                                f"In {func.__name__}(...), parameter `{param_name}`'s "
                                f"dimension {dim_idx} [{typevar}] "
                                f"expected {expected_dim} (method-level binding), "
                                f"got {actual_dim}"
                            )
                    else:
                        # First binding in this call
                        method_context[typevar] = actual_dim

        # Execute function with active contexts
        method_token = _method_typevar_context.set(method_context)
        class_token = _active_class_context.set(class_context)
        try:
            result = func(self, *args, **kwargs)
        finally:
            _method_typevar_context.reset(method_token)
            _active_class_context.reset(class_token)

        # Validate return
        if "return" in hints and result is not None:
            return_annotation = hints["return"]
            typevars = _extract_shape_typevars(return_annotation)
            if typevars and hasattr(result, "shape"):
                actual_shape = getattr(result, "shape")
                for dim_idx, typevar in typevars:
                    if dim_idx >= len(actual_shape):
                        continue

                    actual_dim = actual_shape[dim_idx]
                    is_class_level = _is_class_level_typevar(typevar, owner_cls)
                    context = class_context if is_class_level else method_context
                    if typevar in context:
                        expected_dim = context[typevar]
                        if actual_dim != expected_dim:
                            level = "class" if is_class_level else "method"
                            raise DimensionError(
                                f"In {func.__name__}(...) return value: "
                                f"dimension {dim_idx} ({typevar.__name__}): "
                                f"expected {expected_dim} ({level}-level binding), "
                                f"got {actual_dim}"
                            )

        return result

    return wrapper
