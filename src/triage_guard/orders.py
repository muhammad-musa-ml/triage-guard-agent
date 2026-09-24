"""Deterministic in-memory order records for the repository demo."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer": "Avery Chen",
        "status": "shipped",
        "total": 74.95,
        "items": ["Wireless keyboard"],
        "tracking": "1Z-DEMO-1001",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "customer": "Jordan Reyes",
        "status": "processing",
        "total": 42.50,
        "items": ["USB-C hub"],
        "tracking": None,
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "customer": "Morgan Patel",
        "status": "delivered",
        "total": 129.99,
        "items": ["Noise-cancelling headphones"],
        "tracking": "1Z-DEMO-1003",
    },
}


class OrderError(ValueError):
    """Raised when an order operation fails deterministic validation."""


def get_order(order_id: str) -> dict[str, Any]:
    """Return a defensive copy of a demo order."""

    normalized = order_id.strip().upper()
    if normalized not in ORDERS:
        raise OrderError(f"order {normalized or order_id!r} was not found")
    return deepcopy(ORDERS[normalized])


def submit_refund(order_id: str, amount: float, reason: str) -> dict[str, Any]:
    """Validate and describe a demo refund submission.

    This function intentionally simulates the transaction; it does not contact a
    payment processor or mutate the order fixture.
    """

    order = get_order(order_id)
    if isinstance(amount, bool):
        raise OrderError("refund amount must be a finite number")
    try:
        normalized_amount = round(float(amount), 2)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OrderError("refund amount must be a finite number") from exc
    normalized_reason = reason.strip()

    if not math.isfinite(normalized_amount):
        raise OrderError("refund amount must be a finite number")
    if normalized_amount <= 0:
        raise OrderError("refund amount must be greater than zero")
    if normalized_amount > order["total"]:
        raise OrderError("refund amount cannot exceed the order total")
    if not normalized_reason:
        raise OrderError("refund reason cannot be empty")
    if len(normalized_reason) > 500:
        raise OrderError("refund reason cannot exceed 500 characters")

    return {
        "status": "refund_submitted",
        "order_id": order["order_id"],
        "amount": normalized_amount,
        "reason": normalized_reason,
    }
