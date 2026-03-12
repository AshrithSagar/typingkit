"""
RuntimeOptions
==============

Per-class, inheritable, optionally context-scoped configuration for
``RuntimeGeneric`` subclasses.

Public API
----------
RuntimeOptions          – base frozen-dataclass config attached to a class
RuntimeOptionsProxy     – ContextVar-backed scoped override (context manager)

Attaching options to a class
-----------------------------
Pass an ``options=`` keyword to the class statement — handled by
``RuntimeGeneric.__init_subclass__``::

    class MyClass(RuntimeGeneric[A, B],
                    options=ListOptions(validate_a=False)):
        ...

Options are inherited through the MRO; a subclass that does not pass
``options=`` inherits its nearest parent's options unchanged.

Subclassing RuntimeOptions
--------------------------
Add domain-specific flags by subclassing::

    @dataclass(frozen=True)
    class MyClassRuntimeOptions(RuntimeOptions):
        validate_a: bool = True
        validate_b: bool = True

Temporary overrides
--------------------
Use ``RuntimeOptions.scoped(...)`` as a context manager::

    with RuntimeOptions.scoped(validate=False):
        MyClass[Literal[3], Literal[4]](...)

Scoped options take precedence over class-level options for any
``__runtime_generic_post_init__`` call made within the ``with`` block,
across all ``RuntimeGeneric`` subclasses.  The override is thread- and
async-safe (backed by a ``ContextVar``).
"""
# src/typingkit/core/_options.py

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "RuntimeOptions",
]

# ── Scoped override ───────────────────────────────────────────────────────────

_scoped_options: ContextVar[RuntimeOptions | None] = ContextVar(
    "_scoped_options", default=None
)


# ── Options dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeOptions:
    """
    Immutable per-class configuration for ``RuntimeGeneric``.

    Fields
    ------
    validate : bool
        Master switch.  When ``False``, ``__runtime_generic_post_init__`` is
        skipped entirely — equivalent to plain ``Generic`` behaviour.  Takes
        effect both at the alias-call level (``_RuntimeGenericAlias.__call__``)
        and inside the post-init guard, so subclasses never see the call.

    propagate : bool
        Whether to walk ``__runtime_generic_iter_children__`` and propagate
        resolved types into child ``RuntimeGeneric`` instances.  Setting this
        to ``False`` validates `self` but stops at the boundary — useful when
        children manage their own lifecycle (e.g. lazy containers).
    """

    validate: bool = True
    propagate: bool = True

    # ── Merging ───────────────────────────────────────────────────────────────

    def merged(self, **overrides: Any) -> RuntimeOptions:
        """Return a new options object with `overrides` applied."""
        return replace(self, **overrides)

    # ── Scoped context manager ────────────────────────────────────────────────

    @staticmethod
    def scoped(**overrides: Any) -> _RuntimeOptionsScopedCtx:
        """
        Context manager for temporary option overrides.

        Merges `overrides` into the currently-active scoped options (or the
        sentinel ``RuntimeOptions()`` if none is active), so nested scopes
        compose correctly::

            with RuntimeOptions.scoped(validate=False):
                with RuntimeOptions.scoped(propagate=False):
                    ...  # both validate=False and propagate=False
        """
        return _RuntimeOptionsScopedCtx(overrides)

    # ── Internal resolution ───────────────────────────────────────────────────

    @staticmethod
    def resolve(class_options: RuntimeOptions) -> RuntimeOptions:
        """
        Return the effective options for a call, merging scoped overrides.

        Scoped options (from ``RuntimeOptions.scoped(...)``) always win over
        class-level options.  Only the fields present in *class_options*'s
        type are considered from the scoped override, so domain-specific
        fields on a subclass are not accidentally clobbered.
        """
        scoped = _scoped_options.get()
        if scoped is None:
            return class_options

        # Apply only the base RuntimeOptions fields from the scoped override;
        # domain-specific fields on class_options are preserved.
        base_fields = {"validate", "propagate"}
        overrides: dict[str, Any] = {}
        for field in base_fields:
            scoped_val = getattr(scoped, field, None)
            class_val = getattr(class_options, field, None)
            if scoped_val is not None and scoped_val != class_val:
                # Only override if the scoped value actually differs — this
                # lets a narrower scoped() call not accidentally reset flags
                # set by an outer scope.
                overrides[field] = scoped_val
        if not overrides:
            return class_options
        return replace(class_options, **overrides)


# ── Context manager implementation ────────────────────────────────────────────


class _RuntimeOptionsScopedCtx:
    """Returned by ``RuntimeOptions.scoped()``; not part of the public API."""

    __slots__ = ("_overrides", "_token")

    def __init__(self, overrides: dict[str, Any]) -> None:
        self._overrides = overrides
        self._token = None

    def __enter__(self) -> RuntimeOptions:
        current = _scoped_options.get() or RuntimeOptions()
        merged = replace(current, **self._overrides)
        self._token = _scoped_options.set(merged)  # type: ignore[assignment]
        return merged

    def __exit__(self, *_: Any) -> None:
        if self._token is not None:
            _scoped_options.reset(self._token)
            self._token = None

    # Support async with as well
    async def __aenter__(self) -> RuntimeOptions:
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        self.__exit__(*args)


# ── Global default ────────────────────────────────────────────────────────────

global_default_runtime_options: RuntimeOptions = RuntimeOptions()


def set_global_default_runtime_options(options: RuntimeOptions) -> None:
    global global_default_runtime_options
    global_default_runtime_options = options


def reset_global_default_runtime_options() -> None:
    global global_default_runtime_options
    global_default_runtime_options = RuntimeOptions()
