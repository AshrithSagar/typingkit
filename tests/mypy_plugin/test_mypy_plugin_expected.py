"""
Expected plugin behavior — run with:
    mypy tests/mypy_plugin/test_mypy_plugin_expected.py --config-file pyproject.toml

Each `reveal_type` shows what mypy should report WITH the plugin active.
Without the plugin, all of these would show the opaque DimExpr alias.
"""
# tests/mypy_plugin/test_mypy_plugin_expected.py

# pyright: reportGeneralTypeIssues = false
# pyright: reportUnboundVariable = false
# ruff: noqa: F821

from typing import Literal, TypeVar, reveal_type

from typingkit.numpy._typed.dimexpr import Add, Cubed, Log2, Mul, Neg, Pow, Squared, Sub

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)

# ── Concrete reduction ────────────────────────────────────────────────────────

x1: Add[Literal[3], Literal[1]]
reveal_type(x1)  # Expected: Literal[4]

x2: Mul[Literal[3], Literal[4]]
reveal_type(x2)  # Expected: Literal[12]

x3: Sub[Literal[10], Literal[3]]
reveal_type(x3)  # Expected: Literal[7]

x4: Neg[Literal[5]]
reveal_type(x4)  # Expected: Literal[-5]

x5: Pow[Literal[2], Literal[8]]
reveal_type(x5)  # Expected: Literal[256]

x6: Squared[Literal[7]]
reveal_type(x6)  # Expected: Literal[49]

x7: Cubed[Literal[3]]
reveal_type(x7)  # Expected: Literal[27]

x8: Log2[Literal[8]]
reveal_type(x8)  # Expected: Literal[3]

# ── Nested reduction ──────────────────────────────────────────────────────────

x9: Add[Mul[Literal[2], Literal[3]], Literal[1]]
reveal_type(x9)  # Expected: Literal[7]

x10: Mul[Add[Literal[2], Literal[3]], Sub[Literal[10], Literal[4]]]
reveal_type(x10)  # Expected: Literal[30]  (5 * 6)

x11: Neg[Neg[Literal[5]]]
reveal_type(x11)  # Expected: Literal[5]


# ── Symbolic — should stay as-is ──────────────────────────────────────────────


def test_symbolic(
    x12: Add[N, Literal[1]], x13: Mul[N, M], x14: Add[N, Add[M, Literal[1]]]
) -> None:
    reveal_type(x12)  # Expected: Add[N, Literal[1]]
    reveal_type(x13)  # Expected: Mul[N, M]
    reveal_type(x14)  # Expected: Add[N, Add[M, Literal[1]]]


# ── Partial — inner concrete, outer symbolic ──────────────────────────────────


def test_partial(x15: Add[N, Mul[Literal[2], Literal[3]]]) -> None:
    reveal_type(x15)  # Expected: Add[N, Literal[6]]


# ── Multi-value Literal distribution ─────────────────────────────────────────

x16: Add[Literal[1, 2], Literal[3]]
reveal_type(x16)  # Expected: Literal[4, 5]

x17: Mul[Literal[2, 3], Literal[4, 5]]
reveal_type(x17)  # Expected: Literal[8, 10, 12, 15]

# ── Deduplication ─────────────────────────────────────────────────────────────

x18: Add[Literal[1, 2], Literal[2, 1]]
reveal_type(x18)  # Expected: Literal[2, 3, 4]  (3 appears twice -> deduplicated)
