"""
typingkit mypy plugin
=====================
Teaches mypy to reduce DimExpr type-level expressions at check time.
When args are symbolic (TypeVars), it leaves the expression as-is.
When args are partially concrete, it reduces what it can.

Reduction rules
---------------
1. All args concrete Literals         -> evaluate, return Literal[result]
2. Any arg is a TypeVar               -> return symbolic (unchanged)
3. Nested DimExpr with concrete args  -> reduce inner first, then outer
4. Literal[a, b] (multi-value)        -> distribute, return Literal[a+b, ...]
   i.e. Add[Literal[1,2], Literal[3]] -> Literal[4, 5]
"""
# src/typingkit/mypy_plugin.py

from __future__ import annotations

import math
from typing import Callable

from mypy.nodes import TypeInfo
from mypy.plugin import AnalyzeTypeContext, Plugin
from mypy.types import (
    AnyType,
    Instance,
    LiteralType,
    ProperType,
    RawExpressionType,
    Type,
    TypeOfAny,
    UnboundType,
    UnionType,
    get_proper_type,
)

## Constants

_DIMEXPR_FULLNAME = "typingkit.numpy._typed.dimexpr.DimExpr"


def _log_eval(a: int, b: int) -> int:
    x = math.log(a, b)
    xi = round(x)
    if math.isclose(x, xi, rel_tol=0, abs_tol=1e-12):
        return xi
    raise TypeError(f"Log({a}, {b}) is not an integer")


# Map fully-qualified DimExpr subclass names to their evaluation callables.
# Each callable takes concrete int values and returns an int.
# Mirrors the `expr` classmethods in dimexpr.py — kept here so the plugin
# has zero runtime dependency on typingkit itself.
_DIMEXPR_EVALUATORS: dict[str, Callable[..., int]] = {
    "typingkit.numpy._typed.dimexpr.Neg": lambda a: -a,  # pyright: ignore[reportUnknownLambdaType]
    "typingkit.numpy._typed.dimexpr.Add": lambda a, b: a + b,  # pyright: ignore[reportUnknownLambdaType]
    "typingkit.numpy._typed.dimexpr.Sub": lambda a, b: a - b,  # pyright: ignore[reportUnknownLambdaType]
    "typingkit.numpy._typed.dimexpr.Mul": lambda a, b: a * b,  # pyright: ignore[reportUnknownLambdaType]
    "typingkit.numpy._typed.dimexpr.Pow": lambda a, b: a**b,  # pyright: ignore[reportUnknownLambdaType]
    "typingkit.numpy._typed.dimexpr.Squared": lambda a: a * a,  # pyright: ignore[reportUnknownLambdaType]
    "typingkit.numpy._typed.dimexpr.Cubed": lambda a: a**3,  # pyright: ignore[reportUnknownLambdaType]
    "typingkit.numpy._typed.dimexpr.Log": _log_eval,
    "typingkit.numpy._typed.dimexpr.Log2": lambda a: _log_eval(a, 2),  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    "typingkit.numpy._typed.dimexpr.Log10": lambda a: _log_eval(a, 10),  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
}


## Helpers


def _is_dimexpr_subclass(info: TypeInfo) -> bool:
    """Return True if ``info`` has DimExpr anywhere in its MRO."""
    for base in info.mro:
        if base.fullname == _DIMEXPR_FULLNAME:
            return True
    return False


def _extract_literal_ints(tp: ProperType) -> list[int] | None:
    """
    Extract all int values from a type that is either:
      - LiteralType(int)          -> [value]
      - UnionType of LiteralType  -> [v1, v2, ...]  (Literal[1, 2] desugars to this)

    Returns None if the type contains anything non-concrete (TypeVar, etc.).
    """
    if isinstance(tp, LiteralType):
        if isinstance(tp.value, int):
            return [tp.value]
        return None  # e.g. Literal["hello"] — not an int

    if isinstance(tp, UnionType):
        collected: list[int] = []
        for item in tp.items:
            vals = _extract_literal_ints(get_proper_type(item))
            if vals is None:
                return None  # any non-concrete branch -> whole thing symbolic
            collected.extend(vals)
        return collected

    return None  # TypeVar, Instance, AnyType, etc.


def _make_int_fallback(ctx: AnalyzeTypeContext) -> Instance | None:
    """
    Build a properly registered Instance for builtins.int.

    In get_type_analyze_hook, ctx.api is TypeAnalyser, not SemanticAnalyser.
    TypeAnalyser wraps SemanticAnalyser at ctx.api.api.
    lookup_typeinfo lives on SemanticAnalyser, so we go one level deeper.

    Falls back through three strategies:
      1. ctx.api.api.lookup_typeinfo  (SemanticAnalyser — preferred)
      2. ctx.api.named_type           (TypeAnalyser — may produce synthetic)
      3. None                         (caller returns ctx.type unchanged)
    """
    # Strategy 1: SemanticAnalyser.lookup_typeinfo — properly registered
    semanal = getattr(ctx.api, "api", None)
    if semanal is not None:
        lookup = getattr(semanal, "lookup_typeinfo", None)
        if lookup is not None:
            int_info = lookup("builtins.int")
            if int_info is not None:
                return Instance(int_info, [])

    # Strategy 2: TypeAnalyser.named_type — may work in some mypy versions
    named = getattr(ctx.api, "named_type", None)
    if named is not None:
        try:
            result = named("builtins.int")
            if isinstance(result, Instance):
                return result
        except Exception:
            pass

    return None


def _has_raw_expression(tp: Type) -> bool:
    """
    Recursively check if ``tp`` or any of its args contains a RawExpressionType.

    Our top-level guard only checks ctx.type.args directly, but RawExpressionType
    can be nested inside Instance args (e.g. Add[Mul[RawExpr, ...], ...]).
    This catches all depths.
    """
    if isinstance(tp, RawExpressionType):
        return True
    args = getattr(tp, "args", None)
    if args:
        return any(_has_raw_expression(a) for a in args)
    # UnionType stores items, not args
    items = getattr(tp, "items", None)
    if items:
        return any(_has_raw_expression(i) for i in items)
    return False


def _make_union(types: list[Type]) -> Type:
    """Collapse a list of types into a Union, or return directly if singleton."""
    if len(types) == 1:
        return types[0]
    return UnionType.make_union(types)


## Core reduction


def _try_reduce(
    fullname: str,
    args: list[Type],
    ctx: AnalyzeTypeContext,
    fallback: Instance,
) -> Type | None:
    """
    Attempt to reduce ``fullname[*args]`` to a concrete ``Literal[N]``.

    Returns the reduced type, or None if reduction is not possible.

    ``fallback`` is a pre-built Instance(builtins.int, []) passed through
    to avoid rebuilding it on every recursive call.
    """
    evaluator = _DIMEXPR_EVALUATORS.get(fullname)
    if evaluator is None:
        return None

    # Recursively reduce nested DimExpr args first
    reduced_args: list[Type] = []
    for arg in args:
        proper = get_proper_type(arg)
        if isinstance(proper, Instance) and _is_dimexpr_subclass(proper.type):
            inner = _try_reduce(proper.type.fullname, list(proper.args), ctx, fallback)
            reduced_args.append(inner if inner is not None else arg)
        else:
            reduced_args.append(arg)

    # Extract concrete int values — each arg may be multi-valued (Literal[1,2])
    per_arg_values: list[list[int]] = []
    for arg in reduced_args:
        vals = _extract_literal_ints(get_proper_type(arg))
        if vals is None:
            return None  # symbolic — cannot reduce
        per_arg_values.append(vals)

    # Cartesian product, evaluate each combination
    result_values: list[int] = []
    for combo in _cartesian(per_arg_values):
        try:
            result = evaluator(*combo)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        result_values.append(result)

    # Deduplicate preserving insertion order
    seen: set[int] = set()
    unique: list[int] = []
    for v in result_values:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    literal_types: list[Type] = [
        LiteralType(value=v, fallback=fallback) for v in unique
    ]
    return _make_union(literal_types)


def _cartesian(lists: list[list[int]]) -> list[tuple[int, ...]]:
    """Simple cartesian product without itertools dependency."""
    result: list[tuple[int, ...]] = [()]
    for lst in lists:
        result = [existing + (val,) for existing in result for val in lst]
    return result


## Hook


def _make_dimexpr_hook(
    fullname: str,
) -> Callable[[AnalyzeTypeContext], Type]:
    """
    Return a hook that reduces a specific DimExpr subclass.
    Closed over ``fullname`` so each subclass gets its own hook.
    """

    def hook(ctx: AnalyzeTypeContext) -> Type:
        def _safe_instance(args: list[Type]) -> Type | None:
            """
            Build Instance(fullname_info, args) — the only safe return type.

            Strategy 1: ctx.api.api.lookup_fully_qualified_or_none gives us the
              SymbolTableNode; .node is the TypeInfo we need.
            Strategy 2: ctx.api.named_type(fullname) gives a bare Instance
              (ignore args, but at least it's serializable as a fallback).
            """
            # Strategy 1: get TypeInfo via symbol table lookup
            semanal = getattr(ctx.api, "api", None)
            if semanal is not None:
                for method_name in (
                    "lookup_fully_qualified_or_none",
                    "lookup_fully_qualified",
                ):
                    lookup = getattr(semanal, method_name, None)
                    if lookup is not None:
                        try:
                            node = lookup(fullname)
                            if node is not None:
                                info = getattr(node, "node", None)
                                if isinstance(info, TypeInfo):
                                    return Instance(info, args)
                        except Exception:
                            pass

            # Strategy 2: named_type on TypeAnalyser — returns bare Instance
            # (can't pass args, but at least won't crash serialization)
            named = getattr(ctx.api, "named_type", None)
            if named is not None:
                try:
                    result = named(fullname)
                    if isinstance(result, Instance):
                        # Replace args with our analyzed ones if arity matches
                        if len(result.args) == len(args):
                            return Instance(result.type, args)
                        return result
                except Exception:
                    pass

            return None

        # INVARIANT: never return ctx.type (an UnboundType) — mypy crashes
        # in the indirection detector and serializer if UnboundType leaks
        # into the type map. Every return path must yield a proper Type.

        # Non-unbound types shouldn't reach here, but be safe.
        if not isinstance(ctx.type, UnboundType):  # pyright: ignore[reportUnnecessaryIsInstance]
            result = _safe_instance(list(getattr(ctx.type, "args", [])))
            return result if result is not None else ctx.type

        raw_args = ctx.type.args

        # No args: bare reference like "Add" in base-class position.
        # Return a bare Instance so mypy can serialize it.
        if not raw_args:
            result = _safe_instance([])
            return result if result is not None else AnyType(TypeOfAny.special_form)

        # Analyze each arg: converts UnboundType/RawExpressionType →
        # LiteralType / Instance / TypeVarType.
        analyzed_args: list[Type] = [ctx.api.anal_type(a) for a in raw_args]  # type: ignore[attr-defined]

        # After analysis, args should be clean. If not, return a bare Instance
        # rather than letting unresolvable types leak into the type map.
        if any(_has_raw_expression(a) for a in analyzed_args):
            result = _safe_instance([])
            return result if result is not None else AnyType(TypeOfAny.special_form)

        # Build int fallback for LiteralType construction.
        fallback = _make_int_fallback(ctx)
        if fallback is None:
            result = _safe_instance(analyzed_args)
            return result if result is not None else AnyType(TypeOfAny.special_form)

        # Attempt full reduction (all args concrete).
        reduced = _try_reduce(fullname, analyzed_args, ctx, fallback)
        if reduced is not None:
            return reduced

        # Partial reduction: outer is symbolic, but reduce any concrete inner
        # sub-expressions (e.g. Add[N, Mul[2,3]] → Add[N, Literal[6]]).
        partially_reduced: list[Type] = []
        for arg in analyzed_args:
            proper = get_proper_type(arg)
            if isinstance(proper, Instance) and _is_dimexpr_subclass(proper.type):
                inner = _try_reduce(
                    proper.type.fullname, list(proper.args), ctx, fallback
                )
                partially_reduced.append(inner if inner is not None else arg)
            else:
                partially_reduced.append(arg)

        # Return proper Instance with (partially) reduced args.
        result = _safe_instance(partially_reduced)
        return result if result is not None else AnyType(TypeOfAny.special_form)

    return hook


## Plugin


class TypingKitPlugin(Plugin):
    """
    Mypy plugin for typingkit.

    Currently handles:
    - DimExpr subclass reduction at type-check time
      (Add, Mul, Neg, Pow, Sub, Cubed, Squared, Log, and user-defined subclasses)
    """

    def get_type_analyze_hook(
        self, fullname: str
    ) -> Callable[[AnalyzeTypeContext], Type] | None:
        # Fast path — known built-in DimExpr subclasses
        if fullname in _DIMEXPR_EVALUATORS:
            return _make_dimexpr_hook(fullname)

        # Slow path — user-defined DimExpr subclasses not in our table
        info = self.lookup_fully_qualified(fullname)
        if info is not None and isinstance(info.node, TypeInfo):
            if _is_dimexpr_subclass(info.node):
                return _make_dimexpr_hook(fullname)

        return None


## Entry point


def plugin(version: str) -> type[TypingKitPlugin]:
    return TypingKitPlugin
