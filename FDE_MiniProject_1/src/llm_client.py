from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol, TypeVar

import openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .cost_tracker import UsageTracker

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base error for a model call, with safe operational context."""

    error_type = "LLM_ERROR"

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        retriable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.model = model
        self.status_code = status_code
        self.retriable = retriable

    def as_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "stage": self.stage,
            "model": self.model,
            "status_code": self.status_code,
            "retriable": self.retriable,
        }


class LLMUnavailableError(LLMError):
    """Provider is temporarily unreachable or exhausted bounded retries."""

    error_type = "LLM_UNAVAILABLE"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message, retriable=True, **context)


class LLMResponseError(LLMError):
    """Provider responded, but the result violated the structured contract."""

    error_type = "LLM_RESPONSE_INVALID"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message, retriable=False, **context)


class LLMClient(Protocol):
    def structured_completion(
        self,
        *,
        stage: str,
        model: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[T],
        pricing: tuple[float, float],
    ) -> T: ...


@dataclass(frozen=True)
class ClientConfig:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1
    http_referer: str = ""
    app_title: str = "Dhaga Returns Intelligence"
    provider_name: str = "OpenRouter"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    backoff_base_seconds: float = 1.0
    structured_output_method: str = "json_schema"
    structured_output_strict: bool = False


class OpenAICompatibleClient:
    """LangChain ChatOpenAI adapter configured for an OpenAI-compatible provider."""

    def __init__(self, config: ClientConfig, tracker: UsageTracker) -> None:
        self.config = config
        self.tracker = tracker
        self._runnables: dict[tuple[str, type[BaseModel]], Any] = {}
        self._cache_lock = Lock()

    def _model(self, model: str) -> ChatOpenAI:
        headers = {"X-Title": self.config.app_title}
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        return ChatOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=model,
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
            max_retries=0,
            default_headers=headers,
        )

    def _structured_runnable(self, model: str, response_model: type[T]) -> Any:
        key = (model, response_model)
        with self._cache_lock:
            runnable = self._runnables.get(key)
            if runnable is None:
                runnable = self._model(model).with_structured_output(
                    response_model,
                    method=self.config.structured_output_method,
                    include_raw=True,
                    strict=self.config.structured_output_strict,
                )
                self._runnables[key] = runnable
            return runnable

    @staticmethod
    def _usage(raw: Any) -> tuple[int, int]:
        usage = getattr(raw, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        if input_tokens or output_tokens:
            return input_tokens, output_tokens
        metadata = getattr(raw, "response_metadata", None) or {}
        token_usage = metadata.get("token_usage", {}) or metadata.get("usage", {}) or {}
        return (
            int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0),
            int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0),
        )

    def structured_completion(
        self,
        *,
        stage: str,
        model: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[T],
        pricing: tuple[float, float],
    ) -> T:
        if not self.config.api_key:
            raise LLMUnavailableError(
                "OPENROUTER_API_KEY is not configured",
                stage=stage,
                model=model,
            )

        request_id = str(uuid.uuid4())
        runnable = self._structured_runnable(model, response_model)
        messages = [
            ("system", system_prompt),
            ("human", json.dumps(user_payload, ensure_ascii=False)),
        ]
        last_error: Exception | None = None
        for attempt in range(self.config.max_attempts):
            started = time.perf_counter()
            try:
                outcome = runnable.invoke(messages)
                latency = time.perf_counter() - started
                raw = outcome.get("raw") if isinstance(outcome, dict) else None
                parsed = outcome.get("parsed") if isinstance(outcome, dict) else outcome
                parsing_error = outcome.get("parsing_error") if isinstance(outcome, dict) else None
                input_tokens, output_tokens = self._usage(raw)
                self.tracker.record(
                    stage, model, input_tokens, output_tokens,
                    latency, pricing[0], pricing[1],
                )
                if parsing_error is not None or parsed is None:
                    raise LLMResponseError(
                        f"Invalid structured response from {model}: {parsing_error or 'empty parsed output'}",
                        stage=stage,
                        model=model,
                    )
                result = parsed if isinstance(parsed, response_model) else response_model.model_validate(parsed)
                logger.info("LLM call complete request_id=%s stage=%s model=%s latency=%.3f", request_id, stage, model, latency)
                return result
            except LLMResponseError:
                raise
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
                last_error = exc
                logger.warning(
                    "Transient LLM failure request_id=%s stage=%s attempt=%s error=%s",
                    request_id, stage, attempt + 1, type(exc).__name__,
                )
                if attempt + 1 < self.config.max_attempts:
                    time.sleep(self.config.backoff_base_seconds * (2**attempt))
            except openai.AuthenticationError as exc:
                raise LLMError(
                    f"{self.config.provider_name} rejected the API credential (401)",
                    stage=stage,
                    model=model,
                    status_code=401,
                ) from exc
            except openai.APIStatusError as exc:
                raise LLMError(
                    f"{self.config.provider_name} rejected request ({exc.status_code})",
                    stage=stage,
                    model=model,
                    status_code=exc.status_code,
                ) from exc
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                raise LLMResponseError(
                    f"Invalid structured response from {model}: {exc}",
                    stage=stage,
                    model=model,
                ) from exc
        raise LLMUnavailableError(
            f"Classification service unavailable: {last_error}",
            stage=stage,
            model=model,
        )
