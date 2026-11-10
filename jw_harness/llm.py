"""
LLM access through openJiuwen with budget accounting.

Every call goes through `LLM.ask`, which
  * limits concurrency (the proxy allows 5 sessions),
  * retries 429/5xx with backoff,
  * accounts spend in competition units: input/10 + output
    (reasoning tokens are part of completion_tokens on this proxy),
  * refuses new calls once the optional budget is exhausted.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field

from openjiuwen.core.common.logging.log_config import configure_log_config

# console noise off, log files to the writable tmpfs (the image root is read-only)
configure_log_config({
    "level": "CRITICAL",  # failed calls are retried and reported by the harness itself
    "output": ["console"],
    "log_path": "/tmp/jiuwen_logs/",
    "interface_output": ["console"],
    "performance_output": ["console"],
})

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig  # noqa: E402

CHEAP = "qwen3.6-35b-a3b-noreason"
SECOND = "qwen3.8-27b-noreason"


class BudgetExhausted(Exception):
    pass


class ProxyUnavailable(Exception):
    """Connection-level failure after retries: the task should be retried later, not failed."""


@dataclass
class Spend:
    units: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, inp: int, out: int) -> float:
        cost = inp / 10 + out
        with self.lock:
            self.units += cost
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
        return cost


class LLM:
    def __init__(self, base_url: str, api_key: str, budget: float | None, concurrency: int = 5):
        self.model = Model(
            ModelClientConfig(client_provider="OpenAI", api_key=api_key, api_base=base_url,
                              timeout=180),
            ModelRequestConfig(model=CHEAP, temperature=0),
        )
        self.budget = budget
        self.spend = Spend()
        self.sem = asyncio.Semaphore(concurrency)
        self.dead = False  # proxy stopped answering (budget is over on the server side)

    def remaining(self) -> float | None:
        return None if self.budget is None else self.budget - self.spend.units

    async def ask(self, system: str, user: str | list, *, model: str = CHEAP,
                  max_tokens: int = 400, temperature: float = 0.0) -> tuple[str, float]:
        """Return (content, cost_units). `user` may be a full message list."""
        if self.dead:
            raise BudgetExhausted("proxy is not answering")
        rem = self.remaining()
        if rem is not None and rem < max_tokens * 0.2:
            raise BudgetExhausted(f"remaining {rem:.0f}")
        msgs = [{"role": "system", "content": system}]
        msgs += user if isinstance(user, list) else [{"role": "user", "content": user}]
        last = None
        for attempt in range(5):
            async with self.sem:
                try:
                    r = await self.model.invoke(msgs, model=model, max_tokens=max_tokens,
                                                temperature=temperature)
                    u = r.usage_metadata
                    cost = self.spend.add(u.input_tokens if u else 0, u.output_tokens if u else 0)
                    return (r.content or ""), cost
                except Exception as e:  # noqa: BLE001
                    last = e
            text = str(last).lower()
            if any(s in text for s in ("429", "too many", "много", "rate", "502", "503", "504",
                                       "timeout", "timed out", "connection")):
                await asyncio.sleep(min(2 ** attempt, 20))
                continue
            if any(s in text for s in ("401", "402", "403", "budget", "limit", "quota", "лимит")):
                self.dead = True
                raise BudgetExhausted(str(last)[:200])
            break
        text = str(last).lower()
        if any(s in text for s in ("connection", "timeout", "timed out", "name resolution", "502", "503", "504")):
            raise ProxyUnavailable(str(last)[:200])
        raise RuntimeError(f"llm failed: {str(last)[:300]}")


def from_env(budget: float | None) -> LLM:
    return LLM(os.environ.get("LLM_BASE_URL", "").rstrip("/") or "https://api.neuraldeep.ru/v1",
               os.environ["LLM_API_KEY"], budget,
               concurrency=int(os.environ.get("LLM_CONCURRENCY", "5")))


def now() -> float:
    return time.monotonic()
