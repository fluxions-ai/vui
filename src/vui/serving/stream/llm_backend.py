"""LLM backend abstraction.

One interface, swappable implementations. Add a new provider by writing a
class with `stream` and `complete`, then a branch in `make_backend`.

Backends translate provider quirks (URL path, request shape, streaming
format, thinking-mode flag) into a uniform API:

    backend.stream(messages, ...) -> AsyncIterator[str]   # text chunks
    backend.complete(messages, ...) -> dict                # {content, tool_calls, usage}
    backend.prefill(messages) -> None                      # warm KV (default = complete max_tokens=1)

Pick at startup via env:
    VUI_LLM_BACKEND=ollama|vllm|litellm
    VUI_OLLAMA_URL / VUI_VLLM_URL / VUI_LITELLM_URL
    VUI_OLLAMA_MODEL / VUI_VLLM_MODEL / VUI_LITELLM_MODEL
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.5-4B"

# Sampling defaults — mirrors the qwen3.5:4b ollama modelfile so vLLM and
# ollama produce comparable replies. vLLM has no equivalent of a modelfile;
# without these, its defaults (top_k=-1 i.e. off, top_p=1.0) sample much
# more diversely and the eval scores diverge purely from sampling, not
# from any real quality difference.
DEFAULT_SAMPLING = {
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "presence_penalty": 1.5,
}


class LLMBackend:
    name: str = "abstract"

    def __init__(self, model: str, base_url: str, sampling: dict | None = None):
        self.model = model
        self.base_url = base_url
        self.sampling = {**DEFAULT_SAMPLING, **(sampling or {})}
        self._client: httpx.AsyncClient | None = None

    def _resolve_sampling(
        self,
        *,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        presence_penalty: float | None,
    ) -> dict:
        return {
            "temperature": (
                temperature if temperature is not None else self.sampling["temperature"]
            ),
            "top_k": top_k if top_k is not None else self.sampling["top_k"],
            "top_p": top_p if top_p is not None else self.sampling["top_p"],
            "presence_penalty": (
                presence_penalty
                if presence_penalty is not None
                else self.sampling["presence_penalty"]
            ),
        }

    def _client_inst(self) -> httpx.AsyncClient:
        # Reuse one client per backend so TCP/TLS handshakes are amortised
        # across calls (matters most for spec-prefill firing every few
        # hundred ms during user speech).
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120, connect=10),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def aclose(self):
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | None = None,
        stats: dict | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError
        yield  # pragma: no cover  (makes this an async generator)

    async def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        stats: dict | None = None,
    ) -> dict:
        """Returns {content, tool_calls, usage: {prompt, completion, ctx_used, ctx_max}, done_reason}."""
        raise NotImplementedError

    async def prefill(self, messages: list[dict]) -> None:
        """Warm KV cache. Default: a 1-token completion. Backends can override."""
        await self.complete(messages, max_tokens=1, temperature=0.0)

    async def list_models(self) -> list[str]:
        return [self.model]

    async def set_model(self, name: str) -> None:
        raise NotImplementedError(
            f"{self.name} backend does not support runtime model switch"
        )


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = "http://localhost:11434",
        *,
        think: bool = False,
        num_ctx: int = 8192,
        sampling: dict | None = None,
    ):
        super().__init__(model=model, base_url=base_url, sampling=sampling)
        self.think = think
        self.num_ctx = num_ctx

    def _options(
        self,
        *,
        max_tokens,
        temperature,
        top_k,
        top_p,
        presence_penalty,
        stop,
    ) -> dict:
        opts: dict = {"num_ctx": self.num_ctx}
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        s = self._resolve_sampling(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )
        opts["temperature"] = s["temperature"]
        opts["top_k"] = s["top_k"]
        opts["top_p"] = s["top_p"]
        opts["presence_penalty"] = s["presence_penalty"]
        if stop:
            opts["stop"] = stop
        return opts

    def _body(
        self,
        messages,
        *,
        stream,
        max_tokens,
        temperature,
        top_k,
        top_p,
        presence_penalty,
        stop,
        tools=None,
    ) -> dict:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "30m",
            "think": self.think,
            "options": self._options(
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                presence_penalty=presence_penalty,
                stop=stop,
            ),
        }
        if tools:
            body["tools"] = tools
        return body

    def _record_stats(self, stats: dict | None, prompt_eval: int, eval_count: int):
        if stats is None:
            return
        stats["prompt_tokens"] = prompt_eval
        stats["completion_tokens"] = eval_count
        stats["ctx_used"] = prompt_eval + eval_count
        stats["ctx_max"] = self.num_ctx

    async def stream(
        self,
        messages,
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | None = None,
        stats: dict | None = None,
    ) -> AsyncIterator[str]:
        body = self._body(
            messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        client = self._client_inst()
        async with client.stream(
            "POST", f"{self.base_url}/api/chat", json=body
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tok = d.get("message", {}).get("content", "")
                if tok:
                    yield tok
                if d.get("done"):
                    self._record_stats(
                        stats,
                        d.get("prompt_eval_count", 0),
                        d.get("eval_count", 0),
                    )
                    return

    async def complete(
        self,
        messages,
        *,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        stats: dict | None = None,
    ) -> dict:
        body = self._body(
            messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop=stop,
            tools=tools,
        )
        client = self._client_inst()
        resp = await client.post(f"{self.base_url}/api/chat", json=body)
        resp.raise_for_status()
        d = resp.json()
        msg = d.get("message", {}) or {}
        pe = d.get("prompt_eval_count", 0)
        ec = d.get("eval_count", 0)
        self._record_stats(stats, pe, ec)
        return {
            "content": msg.get("content", "") or "",
            "tool_calls": msg.get("tool_calls") or None,
            "usage": {
                "prompt": pe,
                "completion": ec,
                "ctx_used": pe + ec,
                "ctx_max": self.num_ctx,
            },
            "done_reason": d.get("done_reason"),
        }

    async def list_models(self) -> list[str]:
        client = self._client_inst()
        try:
            r = await client.get(f"{self.base_url}/api/ps", timeout=5)
            return [
                m.get("name", "") for m in r.json().get("models", []) if m.get("name")
            ]
        except Exception:
            return [self.model]

    async def set_model(self, name: str) -> None:
        self.model = name


class VLLMBackend(LLMBackend):
    name = "vllm"

    def __init__(
        self,
        model: str = DEFAULT_VLLM_MODEL,
        base_url: str = "http://localhost:8000",
        *,
        enable_thinking: bool = False,
        max_model_len: int = 8192,
        sampling: dict | None = None,
    ):
        super().__init__(model=model, base_url=base_url, sampling=sampling)
        self.enable_thinking = enable_thinking
        self.max_model_len = max_model_len

    def _body(
        self,
        messages,
        *,
        stream,
        max_tokens,
        temperature,
        top_k,
        top_p,
        presence_penalty,
        stop,
        tools=None,
    ) -> dict:
        s = self._resolve_sampling(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "presence_penalty": s["presence_penalty"],
            # vLLM exposes top_k via extra_body when using OpenAI client; the
            # raw HTTP API accepts it at the top level.
            "top_k": s["top_k"],
            # Qwen3 has chain-of-thought on by default; voice TTFB needs it off.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if stop:
            body["stop"] = stop
        if tools:
            body["tools"] = tools
        return body

    def _record_stats(self, stats: dict | None, usage: dict | None):
        if stats is None or not usage:
            return
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        stats["prompt_tokens"] = pt
        stats["completion_tokens"] = ct
        stats["ctx_used"] = pt + ct
        stats["ctx_max"] = self.max_model_len

    async def stream(
        self,
        messages,
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | None = None,
        stats: dict | None = None,
    ) -> AsyncIterator[str]:
        body = self._body(
            messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        # Ask vLLM to emit a final chunk with usage stats so callers can
        # update ctx fills the same way ollama provides them on `done`.
        body["stream_options"] = {"include_usage": True}
        client = self._client_inst()
        async with client.stream(
            "POST", f"{self.base_url}/v1/chat/completions", json=body
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    return
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = d.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    tok = delta.get("content") or ""
                    if tok:
                        yield tok
                if "usage" in d:
                    self._record_stats(stats, d.get("usage"))

    async def complete(
        self,
        messages,
        *,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        stats: dict | None = None,
    ) -> dict:
        body = self._body(
            messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop=stop,
            tools=tools,
        )
        client = self._client_inst()
        resp = await client.post(f"{self.base_url}/v1/chat/completions", json=body)
        resp.raise_for_status()
        d = resp.json()
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = d.get("usage") or {}
        self._record_stats(stats, usage)
        # vLLM tool_calls have OpenAI shape: each item is
        #   {"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json string>"}}
        # Existing callers expect arguments to be a dict (ollama parses inline);
        # decode the string here so consumers don't care which backend produced it.
        tool_calls = msg.get("tool_calls") or None
        if tool_calls:
            normalised = []
            for tc in tool_calls:
                fn = (tc.get("function") or {}).copy()
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                normalised.append({**tc, "function": fn})
            tool_calls = normalised
        return {
            "content": msg.get("content", "") or "",
            "tool_calls": tool_calls,
            "usage": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "ctx_used": usage.get("prompt_tokens", 0)
                + usage.get("completion_tokens", 0),
                "ctx_max": self.max_model_len,
            },
            "done_reason": choice.get("finish_reason"),
        }

    async def list_models(self) -> list[str]:
        client = self._client_inst()
        try:
            r = await client.get(f"{self.base_url}/v1/models", timeout=5)
            return [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
        except Exception:
            return [self.model]


class LiteLLMBackend(LLMBackend):
    """Backend for LiteLLM proxy — routes to 100+ LLM providers."""

    name = "litellm"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "http://localhost:4000",
        *,
        sampling: dict | None = None,
    ):
        super().__init__(model=model, base_url=base_url, sampling=sampling)

    def _body(
        self,
        messages,
        *,
        stream,
        max_tokens,
        temperature,
        top_k,
        top_p,
        presence_penalty,
        stop,
        tools=None,
    ) -> dict:
        s = self._resolve_sampling(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "presence_penalty": s["presence_penalty"],
            "drop_params": True,
        }
        if stop:
            body["stop"] = stop
        if tools:
            body["tools"] = tools
        return body

    def _record_stats(self, stats: dict | None, usage: dict | None):
        if stats is None or not usage:
            return
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        stats["prompt_tokens"] = pt
        stats["completion_tokens"] = ct
        stats["ctx_used"] = pt + ct
        stats["ctx_max"] = 0

    async def stream(
        self,
        messages,
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | None = None,
        stats: dict | None = None,
    ) -> AsyncIterator[str]:
        body = self._body(
            messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        body["stream_options"] = {"include_usage": True}
        client = self._client_inst()
        async with client.stream(
            "POST", f"{self.base_url}/v1/chat/completions", json=body
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    return
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = d.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    tok = delta.get("content") or ""
                    if tok:
                        yield tok
                if "usage" in d:
                    self._record_stats(stats, d.get("usage"))

    async def complete(
        self,
        messages,
        *,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        stats: dict | None = None,
    ) -> dict:
        body = self._body(
            messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop=stop,
            tools=tools,
        )
        client = self._client_inst()
        resp = await client.post(f"{self.base_url}/v1/chat/completions", json=body)
        resp.raise_for_status()
        d = resp.json()
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = d.get("usage") or {}
        self._record_stats(stats, usage)
        tool_calls = msg.get("tool_calls") or None
        if tool_calls:
            normalised = []
            for tc in tool_calls:
                fn = (tc.get("function") or {}).copy()
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                normalised.append({**tc, "function": fn})
            tool_calls = normalised
        return {
            "content": msg.get("content", "") or "",
            "tool_calls": tool_calls,
            "usage": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "ctx_used": usage.get("prompt_tokens", 0)
                + usage.get("completion_tokens", 0),
                "ctx_max": 0,
            },
            "done_reason": choice.get("finish_reason"),
        }

    async def list_models(self) -> list[str]:
        client = self._client_inst()
        try:
            r = await client.get(f"{self.base_url}/v1/models", timeout=5)
            return [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
        except Exception:
            return [self.model]

    async def set_model(self, name: str) -> None:
        self.model = name


def make_backend(name: str | None = None, model: str | None = None) -> LLMBackend:
    name = (name or os.environ.get("VUI_LLM_BACKEND", "ollama")).lower()
    if name == "vllm":
        return VLLMBackend(
            model=model or os.environ.get("VUI_VLLM_MODEL", DEFAULT_VLLM_MODEL),
            base_url=os.environ.get("VUI_VLLM_URL", "http://localhost:8000"),
        )
    if name == "ollama":
        return OllamaBackend(
            model=model or os.environ.get("VUI_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.environ.get("VUI_OLLAMA_URL", "http://localhost:11434"),
        )
    if name == "litellm":
        return LiteLLMBackend(
            model=model or os.environ.get("VUI_LITELLM_MODEL", "openai/gpt-4o-mini"),
            base_url=os.environ.get("VUI_LITELLM_URL", "http://localhost:4000"),
        )
    raise ValueError(f"unknown VUI_LLM_BACKEND: {name!r}")


_BACKEND: LLMBackend | None = None


def get_backend() -> LLMBackend:
    """Module-level singleton. First call constructs from env."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = make_backend()
    return _BACKEND


def set_backend(backend: LLMBackend) -> None:
    global _BACKEND
    _BACKEND = backend
