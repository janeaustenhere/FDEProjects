from __future__ import annotations

from .classifier import Classifier
from .config import Settings
from .cost_tracker import UsageTracker
from .evaluator import Evaluator
from .llm_client import ClientConfig, OpenAICompatibleClient
from .pipeline import ReturnsPipeline


def build_pipeline(settings: Settings) -> ReturnsPipeline:
    tracker = UsageTracker()
    client = OpenAICompatibleClient(
        ClientConfig(
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=settings.temperature,
            http_referer=settings.http_referer,
            app_title=settings.app_title,
            provider_name=settings.provider_name,
            timeout_seconds=settings.llm_timeout_seconds,
            max_attempts=settings.llm_max_attempts,
            backoff_base_seconds=settings.llm_backoff_base_seconds,
            structured_output_method=settings.structured_output_method,
            structured_output_strict=settings.structured_output_strict,
        ),
        tracker,
    )
    return ReturnsPipeline(
        Classifier(client, settings, "bulk"),
        Classifier(client, settings, "strong"),
        Evaluator(client, settings),
        tracker,
        settings,
    )
