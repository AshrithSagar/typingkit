"""
Context binding
=======
Manages TypeVar binding contexts for shape validation.
"""
# src/typingkit/numpy/_typed/context.py

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

from typingkit.numpy._typed.ndarray import (
    DimensionError,
    _validate_shape,
    _validate_shape_against_contexts,
)

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


def _extract_shape_dims(annotation: Any) -> tuple[Any, ...] | None:
    """Return shape dimension spec tuple[...] from TypedNDArray annotation."""

    origin = get_origin(annotation)
    if origin is None:
        return None

    args = get_args(annotation)
    if not args:
        return None

    shape_spec = args[0]
    if get_origin(shape_spec) is not tuple:
        return None

    return get_args(shape_spec)


def _validate_and_bind(
    *,
    shape_dims: tuple[Any, ...],
    actual_shape: tuple[int, ...],
    owner_cls: type,
    func_name: str,
    param_name: str | None,
    class_context: dict[TypeVar, int],
    method_context: dict[TypeVar, int],
) -> None:
    """
    Performs:
    1. Structural validation (_validate_shape)
    2. TypeVar binding + consistency checks
    """

    # Structural validation (Literal, int, rank, repeated TypeVar)
    _validate_shape(shape_dims, actual_shape)

    for dim_idx, dim in enumerate(shape_dims):
        if not isinstance(dim, TypeVar):
            continue
        if dim_idx >= len(actual_shape):
            continue

        actual_dim = actual_shape[dim_idx]
        is_class_level = _is_class_level_typevar(dim, owner_cls)
        context = class_context if is_class_level else method_context

        if dim in context:
            expected_dim = context[dim]
            if actual_dim != expected_dim:
                level = "class" if is_class_level else "method"
                location = (
                    f"parameter `{param_name}`"
                    if param_name is not None
                    else "return value"
                )
                raise DimensionError(
                    f"In {func_name}(...), {location} "
                    f"dimension {dim_idx} [{dim}] "
                    f"expected {expected_dim} ({level}-level binding), "
                    f"got {actual_dim}"
                )
        else:
            context[dim] = actual_dim


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

        # Validate arguments
        for param_name, param_value in bound_args.arguments.items():
            if param_name == "self":
                continue
            if param_name not in hints:
                continue
            if not hasattr(param_value, "shape"):
                continue

            shape_dims = _extract_shape_dims(hints[param_name])
            if shape_dims is None:
                continue

            _validate_and_bind(
                shape_dims=shape_dims,
                actual_shape=param_value.shape,
                owner_cls=owner_cls,
                func_name=func.__name__,
                param_name=param_name,
                class_context=class_context,
                method_context=method_context,
            )

        # Execute function with active contexts
        method_token = _method_typevar_context.set(method_context)
        class_token = _active_class_context.set(class_context)
        try:
            result = func(self, *args, **kwargs)
        finally:
            _method_typevar_context.reset(method_token)
            _active_class_context.reset(class_token)

        # Validate return
        if "return" in hints and result is not None and hasattr(result, "shape"):
            shape_dims = _extract_shape_dims(hints["return"])
            actual_shape = getattr(result, "shape")
            if shape_dims is not None:
                _validate_and_bind(
                    shape_dims=shape_dims,
                    actual_shape=actual_shape,
                    owner_cls=owner_cls,
                    func_name=func.__name__,
                    param_name=None,
                    class_context=class_context,
                    method_context=method_context,
                )

                # context-aware validation (cross-call)
                _validate_shape_against_contexts(shape_dims, actual_shape)

        return result

    return wrapper
