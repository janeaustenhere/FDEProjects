from types import SimpleNamespace

from src.cost_tracker import UsageTracker
from src.llm_client import ClientConfig, LLMError, LLMResponseError, LLMUnavailableError, OpenAICompatibleClient
from src.models import ClassificationPayload


def test_langchain_openrouter_client_uses_structured_output_and_tracks_usage(monkeypatch):
    captured = {"model_initializations": 0, "structured_initializations": 0}

    class FakeRunnable:
        def invoke(self, messages):
            captured["messages"] = messages
            raw = SimpleNamespace(
                usage_metadata={"input_tokens": 120, "output_tokens": 30},
                response_metadata={},
            )
            parsed = ClassificationPayload(
                primary_reason="FIT", sub_reason="TOO_TIGHT",
                normalized_summary="Customer reports a tight fit.",
                confidence=.9, needs_human_review=False,
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["options"] = kwargs
            captured["model_initializations"] += 1

        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured["structured_options"] = kwargs
            captured["structured_initializations"] += 1
            return FakeRunnable()

    monkeypatch.setattr("src.llm_client.ChatOpenAI", FakeChatOpenAI)
    tracker = UsageTracker()
    client = OpenAICompatibleClient(
        ClientConfig(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            temperature=.2, http_referer="https://example.test", app_title="Dhaga Test",
        ),
        tracker,
    )
    result = client.structured_completion(
        stage="bulk", model="openai/gpt-4.1-mini", system_prompt="Classify.",
        user_payload={"customer_comment": "tight"}, response_model=ClassificationPayload,
        pricing=(1.0, 2.0),
    )
    second = client.structured_completion(
        stage="bulk", model="openai/gpt-4.1-mini", system_prompt="Classify.",
        user_payload={"customer_comment": "still tight"}, response_model=ClassificationPayload,
        pricing=(1.0, 2.0),
    )

    assert result.primary_reason.value == "FIT"
    assert second.primary_reason.value == "FIT"
    assert captured["model_initializations"] == 1
    assert captured["structured_initializations"] == 1
    assert captured["options"]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["options"]["temperature"] == .2
    assert captured["options"]["default_headers"]["HTTP-Referer"] == "https://example.test"
    assert captured["structured_options"]["method"] == "json_schema"
    usage = tracker.summary()[0]
    assert usage.input_tokens == 240
    assert usage.output_tokens == 60


def test_llm_errors_expose_safe_structured_context():
    error = LLMError("Rejected", stage="bulk", model="provider/model", status_code=401)
    assert error.as_dict() == {
        "error_type": "LLM_ERROR",
        "message": "Rejected",
        "stage": "bulk",
        "model": "provider/model",
        "status_code": 401,
        "retriable": False,
    }
    assert LLMUnavailableError("Timeout").retriable is True
    assert LLMResponseError("Invalid JSON").error_type == "LLM_RESPONSE_INVALID"
