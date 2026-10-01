from src.cost_tracker import UsageTracker


def test_cost_formula():
    tracker = UsageTracker()
    tracker.record("bulk", "model", 1_000_000, 500_000, 1.2, 2.0, 4.0)
    usage = tracker.summary()[0]
    assert usage.estimated_cost == 4.0
    assert usage.calls == 1

