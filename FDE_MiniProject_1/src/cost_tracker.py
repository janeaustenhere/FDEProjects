from __future__ import annotations

from threading import Lock

from .models import ModelUsage


class UsageTracker:
    def __init__(self) -> None:
        self._usage: dict[tuple[str, str], ModelUsage] = {}
        self._lock = Lock()

    def record(self, stage: str, model_name: str, input_tokens: int, output_tokens: int, latency_seconds: float, input_price: float, output_price: float) -> None:
        key = (stage, model_name)
        with self._lock:
            usage = self._usage.setdefault(key, ModelUsage(stage=stage, model_name=model_name))
            usage.calls += 1
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.latency_seconds += latency_seconds
            usage.estimated_cost += (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)

    def summary(self) -> list[ModelUsage]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._usage.values()]

    def reset(self) -> None:
        with self._lock:
            self._usage.clear()
