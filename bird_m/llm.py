"""Minimal OpenAI-compatible chat client (Hugging Face Inference Providers router, vLLM, ...).

No third-party dependency. Retries with backoff, thread-pool concurrency, and an append-only JSONL
cache keyed by request hash so an interrupted eval/generation run resumes without paying twice.
The API key is read from HF_TOKEN; in the managed container the proxy injects the real one.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROUTER = "https://router.huggingface.co/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def _key(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass
class ChatClient:
    model: str  # e.g. "Qwen/Qwen3.8-27B:scaleway"
    url: str = ROUTER
    thinking: bool = False
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    retries: int = 4
    timeout: float = 180.0
    cache_path: Path | None = None
    transport: Callable[[str, dict, dict, float], dict] | None = (
        None  # injectable for tests
    )
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self):
        if self.cache_path and Path(self.cache_path).exists():
            for line in Path(self.cache_path).read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self._cache[rec["key"]] = rec["response"]

    # ------------------------------------------------------------ transport
    @staticmethod
    def _http(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            raise LLMError(f"HTTP {e.code}: {body}") from e

    def payload(self, messages: list[dict], **overrides) -> dict:
        p = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        p.update(overrides)
        return p

    def chat(self, messages: list[dict], **overrides) -> dict:
        """Returns {"content", "reasoning", "usage"}; cached by request content."""
        payload = self.payload(messages, **overrides)
        k = _key(payload)
        with self._lock:
            if k in self._cache:
                return self._cache[k]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('HF_TOKEN', 'proxy-injected')}",
        }
        send = self.transport or self._http
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                data = send(self.url, payload, headers, self.timeout)
                if "error" in data:
                    raise LLMError(str(data["error"])[:500])
                msg = data["choices"][0]["message"]
                out = {
                    "content": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content")
                    or msg.get("reasoning")
                    or "",
                    "usage": data.get("usage", {}),
                }
                break
            except LLMError as e:
                last = e
                if any(
                    s in str(e) for s in ("credits", "Pay-as-you go", "401", "402", "403", "Invalid username")
                ):
                    raise  # not transient: do not burn retries
                time.sleep(min(30, 2**attempt))
            except (urllib.error.URLError, TimeoutError, KeyError) as e:
                last = e
                time.sleep(min(30, 2**attempt))
        else:
            raise LLMError(f"failed after {self.retries} attempts: {last}")
        with self._lock:
            self._cache[k] = out
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps({"key": k, "response": out}, ensure_ascii=False)
                        + "\n"
                    )
        return out

    def map(
        self, batch: list[list[dict]], workers: int = 8, **overrides
    ) -> list[dict | Exception]:
        """Run many chats concurrently; failures are returned in place, never raised."""

        def one(msgs):
            try:
                return self.chat(msgs, **overrides)
            except Exception as e:  # noqa: BLE001 - reported to the caller per item
                return e

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, batch))
