from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    bulk_model: str = "openai/gpt-4.1-mini"
    strong_model: str = "openai/gpt-4.1"
    evaluator_model: str = "openai/gpt-4.1-mini"
    temperature: float = 0.1
    http_referer: str = ""
    app_title: str = "Dhaga Returns Intelligence"
    provider_name: str = "OpenRouter"
    confidence_threshold: float = 0.75
    max_concurrent_requests: int = 5
    max_concurrent_batches: int = 3
    max_batch_size: int = 50
    evaluator_batch_size: int = 50
    max_upload_mb: int = 10
    llm_timeout_seconds: float = 60.0
    llm_max_attempts: int = 3
    llm_backoff_base_seconds: float = 1.0
    structured_output_method: str = "json_schema"
    structured_output_strict: bool = False
    bulk_input_cost: float = 0.0
    bulk_output_cost: float = 0.0
    strong_input_cost: float = 0.0
    strong_output_cost: float = 0.0
    evaluator_input_cost: float = 0.0
    evaluator_output_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("OPENROUTER_BASE_URL must not be blank")
        for name, value in {
            "OPENROUTER_BULK_MODEL": self.bulk_model,
            "OPENROUTER_STRONG_MODEL": self.strong_model,
            "OPENROUTER_EVALUATOR_MODEL": self.evaluator_model,
        }.items():
            if not value.strip():
                raise ValueError(f"{name} must not be blank")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
        if not 0 <= self.temperature <= 2:
            raise ValueError("OPENROUTER_TEMPERATURE must be between 0 and 2")
        for name, value in {
            "MAX_CONCURRENT_REQUESTS": self.max_concurrent_requests,
            "MAX_CONCURRENT_BATCHES": self.max_concurrent_batches,
            "MAX_BATCH_SIZE": self.max_batch_size,
            "EVALUATOR_BATCH_SIZE": self.evaluator_batch_size,
            "MAX_UPLOAD_MB": self.max_upload_mb,
            "LLM_MAX_ATTEMPTS": self.llm_max_attempts,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than zero")
        if self.llm_backoff_base_seconds < 0:
            raise ValueError("LLM_BACKOFF_BASE_SECONDS must not be negative")
        if self.structured_output_method not in {"json_schema", "json_mode", "function_calling"}:
            raise ValueError("LLM_STRUCTURED_OUTPUT_METHOD is invalid")
        prices = (
            self.bulk_input_cost, self.bulk_output_cost, self.strong_input_cost,
            self.strong_output_cost, self.evaluator_input_cost, self.evaluator_output_cost,
        )
        if any(price < 0 for price in prices):
            raise ValueError("Model prices must not be negative")

    @classmethod
    def load(cls, env_path: Path | None = None) -> "Settings":
        load_dotenv(env_path or ROOT / ".env", override=False)
        default_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("LLM_API_KEY", "")),
            base_url=os.getenv("OPENROUTER_BASE_URL", os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")),
            bulk_model=os.getenv("OPENROUTER_BULK_MODEL", default_model),
            strong_model=os.getenv("OPENROUTER_STRONG_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1")),
            evaluator_model=os.getenv("OPENROUTER_EVALUATOR_MODEL", default_model),
            temperature=_float("OPENROUTER_TEMPERATURE", 0.1),
            http_referer=os.getenv("OPENROUTER_HTTP_REFERER", ""),
            app_title=os.getenv("OPENROUTER_APP_TITLE", "Dhaga Returns Intelligence"),
            provider_name=os.getenv("LLM_PROVIDER_NAME", "OpenRouter"),
            confidence_threshold=_float("CONFIDENCE_THRESHOLD", 0.75),
            max_concurrent_requests=_int("MAX_CONCURRENT_REQUESTS", 5),
            max_concurrent_batches=_int("MAX_CONCURRENT_BATCHES", 3),
            max_batch_size=_int("MAX_BATCH_SIZE", 50),
            evaluator_batch_size=_int("EVALUATOR_BATCH_SIZE", 50),
            max_upload_mb=_int("MAX_UPLOAD_MB", 10),
            llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 60.0),
            llm_max_attempts=_int("LLM_MAX_ATTEMPTS", 3),
            llm_backoff_base_seconds=_float("LLM_BACKOFF_BASE_SECONDS", 1.0),
            structured_output_method=os.getenv("LLM_STRUCTURED_OUTPUT_METHOD", "json_schema"),
            structured_output_strict=_bool("LLM_STRUCTURED_OUTPUT_STRICT", False),
            bulk_input_cost=_float("BULK_MODEL_INPUT_COST_PER_1M", 0),
            bulk_output_cost=_float("BULK_MODEL_OUTPUT_COST_PER_1M", 0),
            strong_input_cost=_float("STRONG_MODEL_INPUT_COST_PER_1M", 0),
            strong_output_cost=_float("STRONG_MODEL_OUTPUT_COST_PER_1M", 0),
            evaluator_input_cost=_float("EVALUATOR_MODEL_INPUT_COST_PER_1M", 0),
            evaluator_output_cost=_float("EVALUATOR_MODEL_OUTPUT_COST_PER_1M", 0),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key.strip())

    def pricing(self, stage: str) -> tuple[float, float]:
        return {
            "bulk": (self.bulk_input_cost, self.bulk_output_cost),
            "strong": (self.strong_input_cost, self.strong_output_cost),
            "evaluator": (self.evaluator_input_cost, self.evaluator_output_cost),
        }[stage]
