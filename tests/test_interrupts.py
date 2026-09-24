from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from triage_guard.graph import build_graph, open_sqlite_graph


class RefundModel:
    def __init__(self, amount: float = 75) -> None:
        self.amount = amount

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            status = json.loads(messages[-1].content)["status"]
            return AIMessage(content=f"Refund result: {status}")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "refund_order",
                    "args": {
                        "order_id": "ORD-1003" if self.amount > 50 else "ORD-1001",
                        "amount": self.amount,
                        "reason": "Damaged item",
                    },
                    "id": f"refund-{self.amount}",
                }
            ],
        )


def _start_refund(graph, session: str = "refund-session"):
    config = {"configurable": {"thread_id": session}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="Please refund my damaged item")]},
        config=config,
    )
    return result, config


@pytest.mark.parametrize(
    ("approved", "expected_status"),
    [(True, "refund_submitted"), (False, "refund_declined")],
)
def test_sensitive_refund_pauses_and_obeys_review(approved, expected_status) -> None:
    graph = build_graph(RefundModel(), InMemorySaver())
    paused, config = _start_refund(graph)

    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["kind"] == "refund_approval"
    assert payload["amount"] == 75
    assert payload["threshold"] == 50

    result = graph.invoke(
        Command(resume={"approved": approved, "reviewer": "test-reviewer"}),
        config=config,
    )

    tool_message = next(
        message for message in reversed(result["messages"]) if isinstance(message, ToolMessage)
    )
    assert json.loads(tool_message.content)["status"] == expected_status
    assert result["messages"][-1].content == f"Refund result: {expected_status}"


def test_refund_at_exact_threshold_executes_without_interrupt() -> None:
    graph = build_graph(RefundModel(amount=50), InMemorySaver())

    result, _ = _start_refund(graph, session="threshold-session")

    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "Refund result: refund_submitted"


def test_sqlite_interrupt_can_resume_after_graph_is_reopened(tmp_path) -> None:
    database = tmp_path / "interrupt.sqlite"
    config = {"configurable": {"thread_id": "durable-review"}}

    with open_sqlite_graph(database, RefundModel()) as first:
        paused = first.graph.invoke(
            {"messages": [HumanMessage(content="Refund ORD-1003 by $75")]},
            config=config,
        )
        assert paused["__interrupt__"][0].value["amount"] == 75

    with open_sqlite_graph(database, RefundModel()) as second:
        result = second.graph.invoke(Command(resume=True), config=config)

    assert result["messages"][-1].content == "Refund result: refund_submitted"
