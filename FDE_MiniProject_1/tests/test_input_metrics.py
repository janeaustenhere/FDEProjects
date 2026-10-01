import pandas as pd
import pytest

from src.input_metrics import derive_input_metrics, format_rate


def test_other_share_is_derived_from_data():
    metrics = derive_input_metrics(pd.DataFrame({"return_reason": ["Other", "Fit", " other ", ""]}))
    assert metrics.other_reason_rate == pytest.approx(200 / 3)


def test_return_rate_uses_total_orders_denominator():
    metrics = derive_input_metrics(pd.DataFrame({
        "return_id": ["R1", "R2", "R3"],
        "return_reason": ["Other", "Fit", "Other"],
        "total_orders": [10, 10, 10],
    }))
    assert metrics.overall_return_rate == 30
    assert "3 return records" in metrics.return_rate_basis


def test_return_rate_is_unavailable_for_return_only_extract():
    metrics = derive_input_metrics(pd.DataFrame({"return_id": ["R1"], "return_reason": ["Other"]}))
    assert metrics.overall_return_rate is None
    assert format_rate(metrics.overall_return_rate) == "Not available"


def test_return_rate_can_use_is_returned():
    metrics = derive_input_metrics(pd.DataFrame({"is_returned": ["yes", "no", "true", "false"]}))
    assert metrics.overall_return_rate == 50
