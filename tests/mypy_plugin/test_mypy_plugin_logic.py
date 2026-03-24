"""
Unit tests for the core reduction logic in the plugin.
"""
# tests/mypy_plugin/test_mypy_plugin_logic.py

# mypy: disable-error-code="attr-defined"
# pyright: reportArgumentType = false
# pyright: reportAttributeAccessIssue = false
# pyright: reportImplicitOverride = false
# pyright: reportOptionalMemberAccess = false
# pyright: reportUnannotatedClassAttribute = false

import importlib.util
import pathlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any

# ── Minimal mypy type stubs so we can import the plugin without mypy ──────────

# We create just enough of mypy's type structure to run the pure logic.


@dataclass
class LiteralType:
    value: Any
    fallback: Any = None

    def __repr__(self):
        return f"Literal[{self.value}]"


@dataclass
class UnionType:
    items: list[Any]

    @staticmethod
    def make_union(items: list[Any]):
        if len(items) == 1:
            return items[0]
        return UnionType(items=items)

    def __repr__(self):
        vals = ", ".join(repr(i) for i in self.items)
        return f"Union[{vals}]"


@dataclass
class TypeVarType:
    name: str

    def __repr__(self):
        return self.name


@dataclass
class Instance:
    type: Any  # TypeInfo-like
    args: list[Any] = field(default_factory=list[Any])

    def __repr__(self):
        if self.args:
            return f"{self.type.fullname}[{', '.join(repr(a) for a in self.args)}]"
        return self.type.fullname


@dataclass
class TypeInfo:
    fullname: str
    mro: list[Any] = field(default_factory=list[Any])


class AnyType: ...


def get_proper_type(tp: Any) -> Any:
    return tp


# Patch sys.modules so plugin can import mypy.types etc
mypy_types = types.ModuleType("mypy.types")
mypy_types.LiteralType = LiteralType
mypy_types.UnionType = UnionType
mypy_types.TypeVarType = TypeVarType
mypy_types.Instance = Instance
mypy_types.AnyType = AnyType
mypy_types.TypeOfAny = type("TypeOfAny", (), {})
mypy_types.ProperType = object
mypy_types.Type = object
mypy_types.get_proper_type = get_proper_type

mypy_nodes = types.ModuleType("mypy.nodes")
mypy_nodes.TypeInfo = TypeInfo

mypy_plugin = types.ModuleType("mypy.plugin")
mypy_plugin.Plugin = object
mypy_plugin.AnalyzeTypeContext = object

mypy = types.ModuleType("mypy")

sys.modules["mypy"] = mypy
sys.modules["mypy.types"] = mypy_types
sys.modules["mypy.nodes"] = mypy_nodes
sys.modules["mypy.plugin"] = mypy_plugin

# ── Now import our plugin logic ───────────────────────────────────────────────

# We import just the pure functions, not the Plugin class
# (which needs real mypy to subclass)

# Load plugin source
src = pathlib.Path(__file__).parent.parent.parent / "src/typingkit/mypy_plugin.py"
spec = importlib.util.spec_from_file_location("src/typingkit/mypy_plugin", src)
mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(mod)  # type: ignore[union-attr]

_extract_literal_ints = mod._extract_literal_ints
_cartesian = mod._cartesian
_try_reduce = mod._try_reduce
_make_union = mod._make_union
_DIMEXPR_EVALUATORS = mod._DIMEXPR_EVALUATORS

# ── Minimal ctx mock ──────────────────────────────────────────────────────────


class _ApiWithNamedType:
    def named_type(self, name: str):
        return Instance(type=TypeInfo(fullname=name))


class MockCtx:
    # _make_literal_int receives ctx as 'api' param, then calls api.api.named_type
    # so ctx.api must have .named_type
    api = _ApiWithNamedType()


CTX = MockCtx()

# ── DimExpr TypeInfo mocks ────────────────────────────────────────────────────

DIMEXPR_BASE = TypeInfo(fullname="typingkit.numpy._typed.dimexpr.DimExpr")


def make_dimexpr_info(fullname: str) -> TypeInfo:
    info = TypeInfo(fullname=fullname)
    info.mro = [info, DIMEXPR_BASE]
    return info


ADD_INFO = make_dimexpr_info("typingkit.numpy._typed.dimexpr.Add")
MUL_INFO = make_dimexpr_info("typingkit.numpy._typed.dimexpr.Mul")
NEG_INFO = make_dimexpr_info("typingkit.numpy._typed.dimexpr.Neg")
SUB_INFO = make_dimexpr_info("typingkit.numpy._typed.dimexpr.Sub")
POW_INFO = make_dimexpr_info("typingkit.numpy._typed.dimexpr.Pow")
SQUARED_INFO = make_dimexpr_info("typingkit.numpy._typed.dimexpr.Squared")
CUBED_INFO = make_dimexpr_info("typingkit.numpy._typed.dimexpr.Cubed")
LOG_INFO = make_dimexpr_info("typingkit.numpy._typed.dimexpr.Log")

INT_INFO = TypeInfo(fullname="builtins.int")


def lit(n: int) -> LiteralType:
    return LiteralType(value=n, fallback=Instance(type=INT_INFO))


def multi_lit(*ns: int):
    """Literal[a, b, ...] desugars to UnionType in mypy."""
    if len(ns) == 1:
        return lit(ns[0])
    return UnionType(items=[lit(n) for n in ns])


def inst(info: TypeInfo, *args: Any) -> Instance:
    return Instance(type=info, args=list(args))


def tv(name: str) -> TypeVarType:
    return TypeVarType(name=name)


# ── Tests ─────────────────────────────────────────────────────────────────────

passed = 0
failed = 0


def check(name: str, result: Any, expected: Any):
    global passed, failed
    ok = repr(result) == repr(expected)
    status = "S" if ok else "F"
    print(f"  {status} {name}")
    if not ok:
        print(f"      got:      {result!r}")
        print(f"      expected: {expected!r}")
        failed += 1
    else:
        passed += 1


print("\n── _extract_literal_ints ────────────────────────────────────────")

check("single Literal[3]", _extract_literal_ints(lit(3)), [3])

check("multi Literal[1, 2]", _extract_literal_ints(multi_lit(1, 2)), [1, 2])

check("TypeVar -> None", _extract_literal_ints(tv("N")), None)

check(
    "Union with TypeVar -> None",
    _extract_literal_ints(UnionType(items=[lit(1), tv("N")])),
    None,
)


print("\n── _cartesian ───────────────────────────────────────────────────")

check("single list", _cartesian([[1, 2]]), [(1,), (2,)])

check("two lists", _cartesian([[1, 2], [3, 4]]), [(1, 3), (1, 4), (2, 3), (2, 4)])

check("three lists", _cartesian([[1], [2], [3]]), [(1, 2, 3)])


print("\n── _try_reduce: concrete ────────────────────────────────────────")

check(
    "Add[3, 1] -> 4",
    _try_reduce("typingkit.numpy._typed.dimexpr.Add", [lit(3), lit(1)], CTX),
    lit(4),
)

check(
    "Mul[3, 4] -> 12",
    _try_reduce("typingkit.numpy._typed.dimexpr.Mul", [lit(3), lit(4)], CTX),
    lit(12),
)

check(
    "Sub[10, 3] -> 7",
    _try_reduce("typingkit.numpy._typed.dimexpr.Sub", [lit(10), lit(3)], CTX),
    lit(7),
)

check(
    "Neg[5] -> -5",
    _try_reduce("typingkit.numpy._typed.dimexpr.Neg", [lit(5)], CTX),
    lit(-5),
)

check(
    "Pow[2, 8] -> 256",
    _try_reduce("typingkit.numpy._typed.dimexpr.Pow", [lit(2), lit(8)], CTX),
    lit(256),
)

check(
    "Squared[7] -> 49",
    _try_reduce("typingkit.numpy._typed.dimexpr.Squared", [lit(7)], CTX),
    lit(49),
)

check(
    "Cubed[3] -> 27",
    _try_reduce("typingkit.numpy._typed.dimexpr.Cubed", [lit(3)], CTX),
    lit(27),
)

check(
    "Log[8, 2] -> 3",
    _try_reduce("typingkit.numpy._typed.dimexpr.Log", [lit(8), lit(2)], CTX),
    lit(3),
)


print("\n── _try_reduce: nested ──────────────────────────────────────────")

# Add[Mul[2, 3], 1] -> Add[6, 1] -> 7
nested_mul = inst(MUL_INFO, lit(2), lit(3))
check(
    "Add[Mul[2,3], 1] -> 7",
    _try_reduce("typingkit.numpy._typed.dimexpr.Add", [nested_mul, lit(1)], CTX),
    lit(7),
)

# Neg[Neg[5]] -> Neg[-5] -> 5
inner_neg = inst(NEG_INFO, lit(5))
check(
    "Neg[Neg[5]] -> 5",
    _try_reduce("typingkit.numpy._typed.dimexpr.Neg", [inner_neg], CTX),
    lit(5),
)

# Mul[Add[2,3], Sub[10,4]] -> Mul[5, 6] -> 30
inner_add = inst(ADD_INFO, lit(2), lit(3))
inner_sub = inst(SUB_INFO, lit(10), lit(4))
check(
    "Mul[Add[2,3], Sub[10,4]] -> 30",
    _try_reduce("typingkit.numpy._typed.dimexpr.Mul", [inner_add, inner_sub], CTX),
    lit(30),
)


print("\n── _try_reduce: symbolic ────────────────────────────────────────")

N = tv("N")
M = tv("M")

check(
    "Add[N, 1] -> None (symbolic)",
    _try_reduce("typingkit.numpy._typed.dimexpr.Add", [N, lit(1)], CTX),
    None,
)

check(
    "Mul[N, M] -> None (symbolic)",
    _try_reduce("typingkit.numpy._typed.dimexpr.Mul", [N, M], CTX),
    None,
)


print("\n── _try_reduce: partial reduction ───────────────────────────────")

# Add[N, Mul[2, 3]] -> inner reduces to 6, outer still symbolic -> None
# but the inner arg should be reduced when we eventually try
inner_mul = inst(MUL_INFO, lit(2), lit(3))
check(
    "Add[N, Mul[2,3]] -> None (N still symbolic)",
    _try_reduce("typingkit.numpy._typed.dimexpr.Add", [N, inner_mul], CTX),
    None,
)

# The inner DID reduce though — verify separately
check(
    "Mul[2,3] inner reduces to 6",
    _try_reduce("typingkit.numpy._typed.dimexpr.Mul", [lit(2), lit(3)], CTX),
    lit(6),
)


print("\n── _try_reduce: multi-value Literal distribution ─────────────────")

check(
    "Add[Literal[1,2], Literal[3]] -> Literal[4,5]",
    _try_reduce("typingkit.numpy._typed.dimexpr.Add", [multi_lit(1, 2), lit(3)], CTX),
    multi_lit(4, 5),
)

check(
    "Mul[Literal[2,3], Literal[4]] -> Literal[8,12]",
    _try_reduce("typingkit.numpy._typed.dimexpr.Mul", [multi_lit(2, 3), lit(4)], CTX),
    multi_lit(8, 12),
)


print("\n── _try_reduce: deduplication ────────────────────────────────────")

# Add[Literal[1,2], Literal[2,1]]
# cartesian: (1+2, 1+1, 2+2, 2+1) = (3, 2, 4, 3) -> deduped insertion order: [3, 2, 4]
check(
    "Add[Literal[1,2], Literal[2,1]] -> Literal[3,2,4] (deduped, insertion order)",
    _try_reduce(
        "typingkit.numpy._typed.dimexpr.Add", [multi_lit(1, 2), multi_lit(2, 1)], CTX
    ),
    multi_lit(3, 2, 4),
)


print("\n── _try_reduce: error cases ─────────────────────────────────────")

# Log(5, 2) is not an integer -> should return None gracefully
check(
    "Log[5, 2] -> None (not integer)",
    _try_reduce("typingkit.numpy._typed.dimexpr.Log", [lit(5), lit(2)], CTX),
    None,
)

# Unknown fullname -> None
check(
    "Unknown[1, 2] -> None", _try_reduce("some.unknown.Op", [lit(1), lit(2)], CTX), None
)


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'─' * 50}")
print(f"  {passed} passed  {failed} failed")
if failed == 0:
    print("  All good")
