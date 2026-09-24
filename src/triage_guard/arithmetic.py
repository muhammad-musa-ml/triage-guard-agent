"""A small, bounded arithmetic evaluator used by the calculator tool."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

MAX_EXPRESSION_LENGTH = 200
MAX_ABSOLUTE_VALUE = 1_000_000_000_000
MAX_EXPONENT = 10

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpression(ValueError):
    """Raised when an expression leaves the supported arithmetic grammar."""


def evaluate_expression(expression: str) -> int | float:
    """Evaluate numeric literals and basic operators without using ``eval``."""

    if not expression.strip():
        raise UnsafeExpression("expression cannot be empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpression("expression is too long")

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression("expression is not valid arithmetic") from exc

    result = _evaluate(parsed.body)
    return int(result) if isinstance(result, float) and result.is_integer() else result


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _bounded(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _evaluate(node.operand)
        return _bounded(_UNARY_OPERATORS[type(node.op)](operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise UnsafeExpression(f"exponents must be between -{MAX_EXPONENT} and {MAX_EXPONENT}")
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise UnsafeExpression("division by zero is not allowed") from exc
        except (OverflowError, ValueError) as exc:
            raise UnsafeExpression("expression result is outside the allowed range") from exc
        return _bounded(result)

    raise UnsafeExpression(f"unsupported expression element: {type(node).__name__}")


def _bounded(value: int | float) -> int | float:
    if type(value) not in (int, float):
        raise UnsafeExpression("expression result must be a real number")
    if not math.isfinite(value) or abs(value) > MAX_ABSOLUTE_VALUE:
        raise UnsafeExpression("expression result is outside the allowed range")
    return value
