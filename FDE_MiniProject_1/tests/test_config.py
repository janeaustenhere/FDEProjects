import pytest

from src.config import Settings


def test_settings_loads_operational_values_from_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    values = {
        "OPENROUTER_API_KEY": "test-key",
        "OPENROUTER_MODEL": "provider/default",
        "OPENROUTER_BULK_MODEL": "provider/default",
        "OPENROUTER_STRONG_MODEL": "provider/strong",
        "OPENROUTER_EVALUATOR_MODEL": "provider/default",
        "OPENROUTER_TEMPERATURE": "0.25",
        "CONFIDENCE_THRESHOLD": "0.8",
        "MAX_CONCURRENT_REQUESTS": "3",
        "MAX_BATCH_SIZE": "25",
        "EVALUATOR_BATCH_SIZE": "50",
        "MAX_UPLOAD_MB": "7",
        "LLM_TIMEOUT_SECONDS": "12.5",
        "LLM_MAX_ATTEMPTS": "4",
        "LLM_BACKOFF_BASE_SECONDS": "0.5",
        "LLM_STRUCTURED_OUTPUT_METHOD": "function_calling",
        "LLM_STRUCTURED_OUTPUT_STRICT": "true",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings.load(env_file)

    assert settings.bulk_model == "provider/default"
    assert settings.strong_model == "provider/strong"
    assert settings.evaluator_model == "provider/default"
    assert settings.temperature == 0.25
    assert settings.confidence_threshold == 0.8
    assert settings.max_concurrent_requests == 3
    assert settings.max_batch_size == 25
    assert settings.evaluator_batch_size == 50
    assert settings.max_upload_mb == 7
    assert settings.llm_timeout_seconds == 12.5
    assert settings.llm_max_attempts == 4
    assert settings.llm_backoff_base_seconds == 0.5
    assert settings.structured_output_method == "function_calling"
    assert settings.structured_output_strict is True


def test_settings_rejects_invalid_environment_values(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    monkeypatch.setenv("MAX_BATCH_SIZE", "many")

    with pytest.raises(ValueError, match="MAX_BATCH_SIZE must be an integer"):
        Settings.load(env_file)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_threshold": 1.1}, "CONFIDENCE_THRESHOLD"),
        ({"max_concurrent_requests": 0}, "MAX_CONCURRENT_REQUESTS"),
        ({"llm_timeout_seconds": 0}, "LLM_TIMEOUT_SECONDS"),
        ({"structured_output_method": "xml"}, "LLM_STRUCTURED_OUTPUT_METHOD"),
    ],
)
def test_settings_validates_runtime_constraints(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Settings(**kwargs)
