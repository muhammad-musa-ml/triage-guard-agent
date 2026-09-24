"""Deterministic, no-network model used by the CLI architecture demo."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

_ORDER_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
_DOLLAR_AMOUNT_PATTERN = re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)")
_NAMED_AMOUNT_PATTERN = re.compile(
    r"(?:refund|amount)(?:\s+of)?\s+(\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)


class DemoModel:
    """Minimal chat-model protocol implementation for repeatable local demos.

    This is intentionally not presented as an LLM. The production path uses
    ``ChatOpenAI``; this class simply makes graph mechanics runnable without a key.
    """

    def __init__(self) -> None:
        self.tools: tuple[BaseTool, ...] = ()

    def bind_tools(self, tools: Sequence[BaseTool], **_: Any) -> DemoModel:
        self.tools = tuple(tools)
        return self

    def invoke(self, messages: Sequence[BaseMessage], **_: Any) -> AIMessage:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return self._summarize_tool_result(last)
        if not isinstance(last, HumanMessage):
            return AIMessage(content="Please provide a customer-support request.")

        text = str(last.content)
        lowered = text.lower()
        order_match = _ORDER_PATTERN.search(text)

        if "refund" in lowered:
            amount = _extract_amount(text)
            if not order_match or amount is None:
                return AIMessage(
                    content="Please provide an order ID and exact refund amount."
                )
            reason_match = re.search(r"(?:because|reason[:\s]+)\s*(.+)$", text, re.IGNORECASE)
            reason = (
                reason_match.group(1).strip()
                if reason_match
                else "Customer requested a refund in the CLI demo"
            )
            return _tool_request(
                "refund_order",
                {
                    "order_id": order_match.group(0).upper(),
                    "amount": amount,
                    "reason": reason,
                },
            )

        if order_match:
            return _tool_request(
                "search_order", {"order_id": order_match.group(0).upper()}
            )

        if "calculate" in lowered:
            expression = re.split("calculate", text, flags=re.IGNORECASE, maxsplit=1)[1]
            expression = expression.strip().rstrip("?.")
            return _tool_request("calculate", {"expression": expression})

        return AIMessage(
            content=(
                "In demo mode I can look up ORD-1001 through ORD-1003, calculate "
                "an expression, or request a refund."
            )
        )

    @staticmethod
    def _summarize_tool_result(message: ToolMessage) -> AIMessage:
        try:
            result = json.loads(str(message.content))
        except json.JSONDecodeError:
            return AIMessage(content=f"Tool result: {message.content}")

        status = result.get("status")
        if status == "refund_submitted":
            return AIMessage(
                content=(
                    f"Refund submitted for {result['order_id']} in the amount "
                    f"of ${result['amount']:.2f}."
                )
            )
        if status == "refund_declined":
            return AIMessage(
                content=f"The refund for {result['order_id']} was declined by the reviewer."
            )
        if status == "error":
            return AIMessage(content=f"The {result['tool']} tool failed: {result['error']}")
        if "result" in result:
            return AIMessage(content=f"The result is {result['result']}.")
        if "order_id" in result:
            tracking = result.get("tracking") or "not assigned"
            return AIMessage(
                content=(
                    f"{result['order_id']} is {result['status']}; total "
                    f"${result['total']:.2f}; tracking {tracking}."
                )
            )
        return AIMessage(content=f"Tool result: {result}")


def _extract_amount(text: str) -> float | None:
    match = _DOLLAR_AMOUNT_PATTERN.search(text) or _NAMED_AMOUNT_PATTERN.search(text)
    return float(match.group(1)) if match else None


def _tool_request(name: str, arguments: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": arguments, "id": f"demo-{uuid4().hex}"}
        ],
    )
