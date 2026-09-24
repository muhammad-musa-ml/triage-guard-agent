from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from triage_guard.nodes import TriageNodes


class StubModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.bound_tools = ()
        self.received_messages = []

    def bind_tools(self, tools):
        self.bound_tools = tuple(tools)
        return self

    def invoke(self, messages):
        self.received_messages = messages
        return self.response


def test_model_node_binds_all_tools_and_updates_counters() -> None:
    model = StubModel(AIMessage(content="How can I help?"))
    nodes = TriageNodes(model)

    update = nodes.model_node(
        {"messages": [HumanMessage(content="Hello")], "llm_calls": 4}
    )

    assert {tool.name for tool in model.bound_tools} == {
        "calculate",
        "search_order",
        "refund_order",
    }
    assert update["llm_calls"] == 5
    assert update["turn_model_calls"] == 1
    assert model.received_messages[0].type == "system"


def test_tool_node_executes_requested_tools() -> None:
    model = StubModel(AIMessage(content="unused"))
    nodes = TriageNodes(model)
    request = AIMessage(
        content="",
        tool_calls=[
            {"name": "calculate", "args": {"expression": "6 * 7"}, "id": "calc-1"},
            {"name": "search_order", "args": {"order_id": "ORD-1002"}, "id": "order-1"},
        ],
    )

    update = nodes.tool_node({"messages": [request], "tool_calls": 2})

    assert update["tool_calls"] == 4
    assert all(isinstance(message, ToolMessage) for message in update["messages"])
    assert json.loads(update["messages"][0].content)["result"] == 42
    assert json.loads(update["messages"][1].content)["order_id"] == "ORD-1002"


def test_sensitive_refund_cannot_execute_without_recorded_approval() -> None:
    nodes = TriageNodes(StubModel(AIMessage(content="unused")))
    request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "refund_order",
                "args": {"order_id": "ORD-1003", "amount": 75, "reason": "Damaged"},
                "id": "refund-1",
            }
        ],
    )

    update = nodes.tool_node({"messages": [request]})

    assert json.loads(update["messages"][0].content)["status"] == "error"
    assert "approval" in json.loads(update["messages"][0].content)["error"]
