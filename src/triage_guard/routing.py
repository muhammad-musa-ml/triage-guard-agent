"""Pure conditional-edge routing decisions."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END

from triage_guard.policy import MAX_MODEL_CALLS_PER_TURN, is_sensitive_refund
from triage_guard.state import TriageState

Route = Literal["approval", "tools", "loop_guard", "__end__"]


def route_after_model(state: TriageState) -> Route:
    """Route a model response to tools, the loop guard, or ``END``."""

    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        return END

    if not messages[-1].tool_calls:
        return END

    if state.get("turn_model_calls", 0) >= MAX_MODEL_CALLS_PER_TURN:
        return "loop_guard"

    if any(
        is_sensitive_refund(call["name"], call.get("args", {}))
        for call in messages[-1].tool_calls
    ):
        return "approval"

    return "tools"
