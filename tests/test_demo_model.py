from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, ToolMessage

from triage_guard.demo_model import DemoModel
from triage_guard.tools import TOOLS


def test_demo_model_builds_structured_refund_call() -> None:
    model = DemoModel().bind_tools(TOOLS)

    response = model.invoke(
        [HumanMessage(content="Refund $75 for ORD-1003 because the item is damaged")]
    )

    call = response.tool_calls[0]
    assert call["name"] == "refund_order"
    assert call["args"] == {
        "order_id": "ORD-1003",
        "amount": 75.0,
        "reason": "the item is damaged",
    }


def test_demo_model_summarizes_tool_result() -> None:
    model = DemoModel()
    response = model.invoke(
        [
            ToolMessage(
                content=json.dumps(
                    {"status": "refund_submitted", "order_id": "ORD-1003", "amount": 75}
                ),
                tool_call_id="refund-1",
            )
        ]
    )

    assert response.content == "Refund submitted for ORD-1003 in the amount of $75.00."
