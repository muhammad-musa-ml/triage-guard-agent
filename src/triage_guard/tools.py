"""Tools exposed to the triage model."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from triage_guard.arithmetic import evaluate_expression
from triage_guard.orders import get_order, submit_refund


@tool
def calculate(expression: str) -> str:
    """Calculate a basic arithmetic expression such as '(19.99 * 2) + 5'."""

    result = evaluate_expression(expression)
    return json.dumps({"expression": expression, "result": result}, sort_keys=True)


@tool
def search_order(order_id: str) -> str:
    """Look up a support order by ID, for example ORD-1001."""

    return json.dumps(get_order(order_id), sort_keys=True)


@tool
def refund_order(order_id: str, amount: float, reason: str) -> str:
    """Submit a refund for an order with an exact amount and customer reason."""

    return json.dumps(submit_refund(order_id, amount, reason), sort_keys=True)


TOOLS: tuple[BaseTool, ...] = (calculate, search_order, refund_order)
TOOLS_BY_NAME: dict[str, BaseTool] = {item.name: item for item in TOOLS}


def error_result(tool_name: str, error: Exception | str) -> str:
    """Return a stable JSON envelope for a failed or blocked tool call."""

    message = str(error)
    payload: dict[str, Any] = {"status": "error", "tool": tool_name, "error": message}
    return json.dumps(payload, sort_keys=True)
