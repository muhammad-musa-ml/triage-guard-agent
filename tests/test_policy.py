from triage_guard.policy import is_sensitive_refund


def test_only_refunds_strictly_above_threshold_are_sensitive() -> None:
    assert not is_sensitive_refund("refund_order", {"amount": 50})
    assert is_sensitive_refund("refund_order", {"amount": 50.01})
    assert not is_sensitive_refund("calculate", {"amount": 500})


def test_invalid_amount_is_not_misrouted_as_sensitive() -> None:
    assert not is_sensitive_refund("refund_order", {"amount": "not-a-number"})
    assert not is_sensitive_refund("refund_order", {"amount": True})
    assert not is_sensitive_refund("refund_order", {"amount": float("nan")})
    assert not is_sensitive_refund("refund_order", {"amount": float("inf")})
