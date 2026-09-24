"""Command-line driver for persistent triage sessions and approval reviews."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from triage_guard.demo_model import DemoModel
from triage_guard.graph import open_sqlite_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage-guard",
        description="Run a durable customer-support triage session.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the deterministic no-key model instead of OpenAI.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(".triage_guard/checkpoints.sqlite3"),
        help="SQLite checkpoint path (default: .triage_guard/checkpoints.sqlite3).",
    )
    parser.add_argument(
        "--session",
        default="support-demo",
        help="Persistent LangGraph thread ID (default: support-demo).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model name when --demo is not set.",
    )
    parser.add_argument(
        "--message",
        help="Run one message and exit; approval prompts remain interactive.",
    )
    return parser


def create_model(demo: bool, model_name: str) -> Any:
    if demo:
        return DemoModel()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or run with --demo for a no-key simulation."
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=0)


def run_turn(
    graph: Any,
    text: str,
    config: dict[str, Any],
    input_fn: Callable[[str], str] = input,
) -> dict[str, Any]:
    """Run one user turn, pausing for every surfaced review interrupt."""

    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    while result.get("__interrupt__"):
        pending = result["__interrupt__"][0]
        payload = pending.value
        _print_review(payload)
        approved = _read_approval(input_fn)
        result = graph.invoke(
            Command(
                resume={
                    "approved": approved,
                    "reviewer": os.getenv("USER") or os.getenv("USERNAME") or "cli-reviewer",
                }
            ),
            config=config,
        )
    return result


def _print_review(payload: dict[str, Any]) -> None:
    print("\n--- HUMAN APPROVAL REQUIRED ---")
    print(f"Order:  {payload['order_id']}")
    print(f"Amount: ${payload['amount']:.2f}")
    print(f"Reason:  {payload['reason']}")
    print(f"Policy:  refunds over ${payload['threshold']:.2f} require approval")


def _read_approval(input_fn: Callable[[str], str]) -> bool:
    while True:
        answer = input_fn("Approve refund? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter y to approve or n to reject.")


def _print_answer(result: dict[str, Any]) -> None:
    messages = result.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        print(f"Agent: {messages[-1].content}")


def _print_state(graph: Any, config: dict[str, Any]) -> None:
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    print(
        "State: "
        f"messages={len(values.get('messages', []))}, "
        f"llm_calls={values.get('llm_calls', 0)}, "
        f"tool_calls={values.get('tool_calls', 0)}, "
        f"last_route={values.get('last_route', 'none')}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = create_model(args.demo, args.model)
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        return 2

    config = {"configurable": {"thread_id": args.session}}
    with open_sqlite_graph(args.db, model) as resource:
        print(f"Session: {args.session} | Checkpoints: {args.db}")
        print("Commands: /state shows checkpointed counters; /quit exits.")

        if args.message:
            result = run_turn(resource.graph, args.message, config)
            _print_answer(result)
            return 0

        while True:
            try:
                text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession saved.")
                return 0
            if not text:
                continue
            if text.lower() in {"/quit", "/exit"}:
                print("Session saved.")
                return 0
            if text.lower() == "/state":
                _print_state(resource.graph, config)
                continue

            result = run_turn(resource.graph, text, config)
            _print_answer(result)


if __name__ == "__main__":
    raise SystemExit(main())
