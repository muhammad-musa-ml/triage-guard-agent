"""Shared graph state and policy constants."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage

from triage_guard.policy import MAX_MODEL_CALLS_PER_TURN, REFUND_APPROVAL_THRESHOLD


class RefundDecision(TypedDict):
    """A human decision associated with one model-generated tool call."""

    tool_call_id: str
    approved: bool
    reviewer: str


class TriageState(TypedDict, total=False):
    """State persisted at every LangGraph super-step.

    Messages append through a reducer. Counters and the current turn's approval
    decisions are replaced explicitly by nodes.
    """

    messages: Annotated[list[AnyMessage], operator.add]
    refund_decisions: list[RefundDecision]
    llm_calls: int
    tool_calls: int
    turn_model_calls: int
    last_route: str
    metadata: dict[str, Any]
