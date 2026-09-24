from __future__ import annotations

import operator
from typing import Annotated, get_args, get_origin, get_type_hints

from triage_guard.state import TriageState


def test_messages_use_append_reducer() -> None:
    hints = get_type_hints(TriageState, include_extras=True)
    message_hint = hints["messages"]

    assert get_origin(message_hint) is Annotated
    value_type, reducer = get_args(message_hint)
    assert get_origin(value_type) is list
    assert reducer is operator.add


def test_state_contains_observability_counters() -> None:
    hints = get_type_hints(TriageState, include_extras=True)

    assert hints["llm_calls"] is int
    assert hints["tool_calls"] is int
    assert hints["turn_model_calls"] is int

