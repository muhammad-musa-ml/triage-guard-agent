"""Agentic and deterministic LangGraph nodes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from triage_guard.policy import REFUND_APPROVAL_THRESHOLD, is_sensitive_refund
from triage_guard.state import RefundDecision, TriageState
from triage_guard.tools import TOOLS, error_result

SYSTEM_PROMPT = """You are a concise customer-support triage agent.
Use search_order before making claims about an order. Use calculate for arithmetic.
Use refund_order only when the customer supplied an order ID, exact amount, and reason.
Request at most one refund per model response. Never claim a tool action succeeded
until its ToolMessage confirms success. Refund approval policy is enforced by the graph.
"""


class TriageNodes:
    """Node collection sharing one tool-bound model instance."""

    def __init__(self, model: Any, tools: Sequence[BaseTool] = TOOLS) -> None:
        self.tools = tuple(tools)
        self.tools_by_name = {item.name: item for item in self.tools}
        self.model = model.bind_tools(self.tools)

    def model_node(self, state: TriageState) -> dict[str, Any]:
        """Ask the model for a response or a structured tool request."""

        messages = state.get("messages", [])
        response = self.model.invoke([SystemMessage(content=SYSTEM_PROMPT), *messages])
        if not isinstance(response, AIMessage):
            raise TypeError("the bound chat model must return an AIMessage")

        starts_new_turn = bool(messages) and isinstance(messages[-1], HumanMessage)
        turn_calls = 1 if starts_new_turn else state.get("turn_model_calls", 0) + 1
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "turn_model_calls": turn_calls,
            "last_route": "model",
        }

    def tool_node(self, state: TriageState) -> dict[str, Any]:
        """Execute requested tools and append one result per tool call.

        Sensitive refunds are blocked unless the approval node recorded a decision
        for the exact tool-call ID. This is defense in depth in addition to routing.
        """

        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            raise ValueError("tool_node requires a final AIMessage")

        calls = messages[-1].tool_calls
        decisions = {
            item["tool_call_id"]: item for item in state.get("refund_decisions", [])
        }
        results = [self._execute_tool_call(call, decisions) for call in calls]
        return {
            "messages": results,
            "tool_calls": state.get("tool_calls", 0) + len(calls),
            "last_route": "tools",
            "refund_decisions": [],
        }

    def approval_node(self, state: TriageState) -> dict[str, Any]:
        """Pause once for every sensitive refund in the current model response."""

        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            raise ValueError("approval_node requires a final AIMessage")

        decisions: list[RefundDecision] = []
        for call in messages[-1].tool_calls:
            arguments = call.get("args", {})
            if not is_sensitive_refund(call["name"], arguments):
                continue

            raw_decision = interrupt(
                {
                    "kind": "refund_approval",
                    "question": (
                        f"Approve a ${float(arguments['amount']):.2f} refund "
                        f"for {arguments.get('order_id', 'unknown order')}?"
                    ),
                    "tool_call_id": call["id"],
                    "order_id": arguments.get("order_id"),
                    "amount": float(arguments["amount"]),
                    "reason": arguments.get("reason"),
                    "threshold": REFUND_APPROVAL_THRESHOLD,
                }
            )
            approved, reviewer = _normalize_review(raw_decision)
            decisions.append(
                {
                    "tool_call_id": call["id"],
                    "approved": approved,
                    "reviewer": reviewer,
                }
            )

        if not decisions:
            raise ValueError("approval_node received no sensitive refund calls")
        return {"refund_decisions": decisions, "last_route": "approval"}

    def loop_guard_node(self, state: TriageState) -> dict[str, Any]:
        """End a turn cleanly when a model keeps requesting tools."""

        limit = state.get("turn_model_calls", 0)
        message = AIMessage(
            content=(
                "I stopped this request after "
                f"{limit} model steps to avoid an unsafe tool loop. "
                "Please restate the request or ask for a human agent."
            )
        )
        return {"messages": [message], "last_route": "loop_guard"}

    def _execute_tool_call(
        self,
        call: dict[str, Any],
        decisions: dict[str, RefundDecision],
    ) -> ToolMessage:
        name = call["name"]
        call_id = call["id"]
        arguments = call.get("args", {})

        if is_sensitive_refund(name, arguments):
            decision = decisions.get(call_id)
            if decision is None:
                content = error_result(name, "human approval is required before execution")
            elif not decision["approved"]:
                content = json.dumps(
                    {
                        "status": "refund_declined",
                        "order_id": arguments.get("order_id"),
                        "amount": arguments.get("amount"),
                        "reviewer": decision["reviewer"],
                    },
                    sort_keys=True,
                )
            else:
                content = self._invoke(name, arguments)
        else:
            content = self._invoke(name, arguments)

        return ToolMessage(content=content, tool_call_id=call_id, name=name)

    def _invoke(self, name: str, arguments: dict[str, Any]) -> str:
        selected = self.tools_by_name.get(name)
        if selected is None:
            return error_result(name, "unknown tool")
        try:
            result = selected.invoke(arguments)
            return result if isinstance(result, str) else json.dumps(result, sort_keys=True)
        except Exception as exc:  # Tool failures become observations for the model.
            return error_result(name, exc)


def _normalize_review(raw_decision: Any) -> tuple[bool, str]:
    if isinstance(raw_decision, bool):
        return raw_decision, "external-reviewer"
    if isinstance(raw_decision, dict) and isinstance(raw_decision.get("approved"), bool):
        reviewer = str(raw_decision.get("reviewer") or "external-reviewer").strip()
        return raw_decision["approved"], reviewer[:100] or "external-reviewer"
    raise ValueError(
        "refund review must be a boolean or an object containing an 'approved' boolean"
    )
