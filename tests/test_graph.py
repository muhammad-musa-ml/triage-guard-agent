from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from triage_guard.graph import build_graph, open_sqlite_graph


class ToolThenAnswerModel:
    def bind_tools(self, tools):
        self.tools = tuple(tools)
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"Order lookup complete: {messages[-1].content}")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_order",
                    "args": {"order_id": "ORD-1001"},
                    "id": "lookup-1",
                }
            ],
        )


class HistoryModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        conversation = messages[1:]
        return AIMessage(content=f"conversation_messages_before_reply={len(conversation)}")


def test_compiled_graph_cycles_through_tools_and_terminates() -> None:
    graph = build_graph(ToolThenAnswerModel(), InMemorySaver())
    config = {"configurable": {"thread_id": "tool-cycle"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="Where is ORD-1001?")]}, config=config
    )

    assert isinstance(result["messages"][-2], ToolMessage)
    assert "Order lookup complete" in result["messages"][-1].content
    assert result["llm_calls"] == 2
    assert result["tool_calls"] == 1


def test_sqlite_restores_session_after_connection_is_reopened(tmp_path) -> None:
    database = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "persistent-session"}}

    with open_sqlite_graph(database, HistoryModel()) as first:
        first.graph.invoke(
            {"messages": [HumanMessage(content="First turn")]}, config=config
        )

    with open_sqlite_graph(database, HistoryModel()) as second:
        result = second.graph.invoke(
            {"messages": [HumanMessage(content="Second turn")]}, config=config
        )

    assert len(result["messages"]) == 4
    assert result["messages"][-1].content == "conversation_messages_before_reply=3"
    assert result["llm_calls"] == 2
