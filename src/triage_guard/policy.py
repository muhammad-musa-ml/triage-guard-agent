"""Dependency-free deterministic guardrail policy."""

from __future__ import annotations

import math
from typing import Any

REFUND_APPROVAL_THRESHOLD = 50.0
MAX_MODEL_CALLS_PER_TURN = 8


def is_sensitive_refund(name: str, arguments: dict[str, Any]) -> bool:
    """Return whether a tool call crosses the mandatory review threshold."""

    if name != "refund_order":
        return False
    amount = arguments.get("amount", 0)
    if isinstance(amount, bool):
        return False
    try:
        normalized = float(amount)
        return math.isfinite(normalized) and normalized > REFUND_APPROVAL_THRESHOLD
    except (TypeError, ValueError):
        return False
