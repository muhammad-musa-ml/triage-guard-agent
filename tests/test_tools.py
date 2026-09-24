from __future__ import annotations

import json

import pytest

from triage_guard.arithmetic import UnsafeExpression, evaluate_expression
from triage_guard.orders import OrderError, get_order, submit_refund
from triage_guard.tools import calculate, refund_order, search_order


def test_arithmetic_obeys_precedence() -> None:
    assert evaluate_expression("2 + 3 * 4") == 14
    assert json.loads(calculate.invoke({"expression": "(10 - 2) / 4"}))["result"] == 2


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "[1, 2, 3]",
        "2 ** 100",
        "1 / 0",
        "(-1) ** 0.5",
    ],
)
def test_arithmetic_rejects_unsafe_or_unbounded_input(expression: str) -> None:
    with pytest.raises(UnsafeExpression):
        evaluate_expression(expression)


def test_order_search_is_case_insensitive_and_returns_copy() -> None:
    first = get_order("ord-1001")
    first["status"] = "changed"

    assert get_order("ORD-1001")["status"] == "shipped"
    assert json.loads(search_order.invoke({"order_id": "ord-1001"}))["total"] == 74.95


def test_unknown_order_is_rejected() -> None:
    with pytest.raises(OrderError, match="not found"):
        get_order("ORD-9999")


def test_refund_validation_rejects_amount_above_order_total() -> None:
    with pytest.raises(OrderError, match="cannot exceed"):
        submit_refund("ORD-1002", 50, "Changed my mind")


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), True])
def test_refund_validation_rejects_non_finite_amounts(amount) -> None:
    with pytest.raises(OrderError, match="finite number"):
        submit_refund("ORD-1003", amount, "Invalid amount")


def test_refund_tool_returns_submission_envelope() -> None:
    result = json.loads(
        refund_order.invoke(
            {"order_id": "ORD-1003", "amount": 75, "reason": "Damaged item"}
        )
    )

    assert result == {
        "amount": 75.0,
        "order_id": "ORD-1003",
        "reason": "Damaged item",
        "status": "refund_submitted",
    }
