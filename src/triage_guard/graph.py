"""Graph construction and SQLite resource lifecycle."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from triage_guard.nodes import TriageNodes
from triage_guard.routing import route_after_model
from triage_guard.state import TriageState


def build_graph(model: Any, checkpointer: BaseCheckpointSaver[Any]) -> Any:
    """Build and compile the triage graph with a supplied checkpointer."""

    nodes = TriageNodes(model)
    builder = StateGraph(TriageState)
    builder.add_node("model", nodes.model_node)
    builder.add_node("approval", nodes.approval_node)
    builder.add_node("tools", nodes.tool_node)
    builder.add_node("loop_guard", nodes.loop_guard_node)

    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        route_after_model,
        {
            "approval": "approval",
            "tools": "tools",
            "loop_guard": "loop_guard",
            END: END,
        },
    )
    builder.add_edge("approval", "tools")
    builder.add_edge("tools", "model")
    builder.add_edge("loop_guard", END)
    return builder.compile(checkpointer=checkpointer)


@dataclass(slots=True)
class SQLiteTriageGraph:
    """Own a compiled graph and the SQLite connection backing its saver."""

    graph: Any
    connection: sqlite3.Connection

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_sqlite_graph(database: str | Path, model: Any) -> SQLiteTriageGraph:
    """Open a durable SQLite checkpointer and compile a graph around it."""

    database_text = str(database)
    if database_text != ":memory:":
        path = Path(database).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        database_text = str(path)

    connection = sqlite3.connect(database_text, check_same_thread=False)
    connection.execute("PRAGMA busy_timeout = 5000")
    if database_text != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")

    serializer = JsonPlusSerializer(allowed_msgpack_modules=None)
    checkpointer = SqliteSaver(connection, serde=serializer)
    try:
        graph = build_graph(model, checkpointer)
    except Exception:
        connection.close()
        raise
    return SQLiteTriageGraph(graph=graph, connection=connection)
