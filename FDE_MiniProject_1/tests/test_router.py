from src.models import ClassificationResult
from src.router import should_route


def make_result(**overrides):
    data = dict(return_id="R1", sku_id="S1", category="Top", return_comment="tight", primary_reason="FIT", sub_reason="TOO_TIGHT", normalized_summary="Tight fit.", confidence=.9, needs_human_review=False, model_name="bulk", processing_status="ACCEPTED")
    data.update(overrides)
    return ClassificationResult(**data)


def test_high_confidence_easy_case_is_accepted():
    assert should_route(make_result(), .75)[0] is False


def test_low_confidence_routes():
    assert should_route(make_result(confidence=.5), .75)[0] is True


def test_multi_issue_routes():
    assert should_route(make_result(return_comment="tight aur stitching kharab"), .75)[0] is True


def test_other_known_always_routes_for_verification():
    result = make_result(
        primary_reason="OTHER_KNOWN", sub_reason=None,
        discovered_category="Skin Irritation", discovered_reason="Fabric caused itching.",
    )
    routed, reason = should_route(result, .75)
    assert routed is True
    assert "verification" in reason
