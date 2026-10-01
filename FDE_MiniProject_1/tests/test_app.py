from streamlit.testing.v1 import AppTest
from src.config import ROOT
from src.models import ClassificationResult, PipelineResult


def test_app_and_demo_validation_render_with_environment_configuration():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "Dhaga & Co."
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "Total rows" and metric.value == "200" for metric in app.metric)
    analyse = next(button for button in app.button if button.label == "Analyse all returns")
    assert not analyse.disabled
    assert not app.text_input


def test_dashboard_button_uses_deferred_navigation():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=10).run()
    app.button[0].click().run()
    result = ClassificationResult(
        return_id="R1", sku_id="S1", vendor="Vendor A", category="Top",
        return_comment="", source_return_reason="Size Issue",
        primary_reason="SIZE_INFORMATION", sub_reason="SIZE_INFORMATION_OTHER",
        normalized_summary="Structured reason.", confidence=1,
        needs_human_review=False, model_name="structured-input", processing_status="ACCEPTED",
    )
    app.session_state["pipeline_result"] = PipelineResult(results=[result], usage=[], processing_seconds=.1)
    app.run()
    open_dashboard = next(button for button in app.button if button.label == "Open full Returns Intelligence dashboard")
    open_dashboard.click().run()
    assert not app.exception
    assert app.radio[0].value == "Returns Intelligence"
