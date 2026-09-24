from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from triage_guard.routing import route_after_model
from triage_guard.state import MAX_MODEL_CALLS_PER_TURN


def test_final_model_reply_routes_to_end() -> None:
    assert route_after_model({"messages": [AIMessage(content="Done")]}) == END


def test_tool_request_routes_to_tool_node() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_order", "args": {"order_id": "ORD-1001"}, "id": "call-1"}
        ],
    )

    assert route_after_model({"messages": [message], "turn_model_calls": 1}) == "tools"


def test_sensitive_refund_routes_to_approval() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "refund_order",
                "args": {"order_id": "ORD-1003", "amount": 75, "reason": "Damaged"},
                "id": "refund-1",
            }
        ],
    )

    assert route_after_model({"messages": [message], "turn_model_calls": 1}) == "approval"


def test_refund_at_threshold_does_not_require_approval() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "refund_order",
                "args": {"order_id": "ORD-1001", "amount": 50, "reason": "Late"},
                "id": "refund-50",
            }
        ],
    )

    assert route_after_model({"messages": [message], "turn_model_calls": 1}) == "tools"


def test_cycle_limit_routes_to_clean_guard_node() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_order", "args": {"order_id": "ORD-1001"}, "id": "call-1"}
        ],
    )

    assert (
        route_after_model(
            {"messages": [message], "turn_model_calls": MAX_MODEL_CALLS_PER_TURN}
        )
        == "loop_guard"
    )


def test_non_ai_tail_terminates_defensively() -> None:
    assert route_after_model({"messages": [HumanMessage(content="Hi")]}) == END
