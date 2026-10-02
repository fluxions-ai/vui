"""LLM backend abstraction.

One interface, swappable implementations. Add a new provider by writing a
class with `stream` and `complete`, then a branch in `make_backend`.

Backends translate provider quirks (URL path, request shape, streaming
format, thinking-mode flag) into a uniform API:

    backend.stream(messages, ...) -> AsyncIterator[str]   # text chunks
    backend.complete(messages, ...) -> dict                # {content, tool_calls, usage}
    backend.prefill(messages) -> None                      # warm KV (default = complete max_tokens=1)

Pick at startup via env:
    VUI_LLM_BACKEND=ollama|vllm|litellm|openai
    VUI_OLLAMA_URL / VUI_VLLM_URL / VUI_LITELLM_URL / VUI_OPENAI_URL
    VUI_OLLAMA_MODEL / VUI_VLLM_MODEL / VUI_LITELLM_MODEL / VUI_OPENAI_MODEL
    VUI_{VLLM,LITELLM,OPENAI}_API_KEY, VUI_{VLLM,LITELLM,OPENAI}_REASONING_EFFORT
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator

import httpx

DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_VLLM_MODEL = "google/gemma-4-E4B-it"

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

    # Capabilities the UI keys off, so callers don't have to test `name`.
    # Can the served model be swapped at runtime?
    supports_model_switch: bool = False
    # Is there a registry to fetch a model the server doesn't have yet?
    supports_pull: bool = False

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

    async def health(self) -> bool:
        """Cheap liveness probe. Must not load a model or generate tokens."""
        return True

    async def list_models(self) -> list[str]:
        """Models this backend can be switched to."""
        return [self.model]

    async def loaded_models(self) -> list[str]:
        """Models the server currently holds in memory, most-recent first."""
        return []

    async def set_model(self, name: str) -> None:
        raise NotImplementedError(
            f"{self.name} backend does not support runtime model switch"
        )

    async def pull(self, name: str) -> AsyncIterator[dict]:
        """Yield {status, completed, total} progress dicts."""
        raise NotImplementedError(f"{self.name} backend has no model registry")
        yield {}  # pragma: no cover — makes this an async generator


class OllamaBackend(LLMBackend):
    name = "ollama"
    supports_model_switch = True
    supports_pull = True

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

    async def health(self) -> bool:
        try:
            r = await self._client_inst().get(f"{self.base_url}/api/version", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        # /api/tags = installed. What the UI dropdown offers.
        client = self._client_inst()
        try:
            r = await client.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [
                m.get("name", "") for m in r.json().get("models", []) if m.get("name")
            ]
        except Exception:
            # Fall back to the current model rather than an empty dropdown.
            return [self.model]

    async def loaded_models(self) -> list[str]:
        # /api/ps = resident in VRAM right now. Not the same list as above.
        client = self._client_inst()
        try:
            r = await client.get(f"{self.base_url}/api/ps", timeout=5)
            r.raise_for_status()
            return [
                m.get("name", "") for m in r.json().get("models", []) if m.get("name")
            ]
        except Exception:
            return []

    async def set_model(self, name: str) -> None:
        if name == self.model:
            return
        # Free the VRAM the outgoing model holds; best-effort, Ollama will
        # evict on its own eventually.
        try:
            await self._client_inst().post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass
        self.model = name

    async def pull(self, name: str) -> AsyncIterator[dict]:
        client = self._client_inst()
        async with client.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"model": name, "stream": True},
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# Ceiling on stop sequences a strict API takes: OpenAI's limit is 4, Gemini's 5.
STANDARD_MAX_STOP = 4


def _api_root(url: str) -> str:
    """The provider's documented base URL, or a bare host that gets `/v1`.

    `http://host:8000` and `https://openrouter.ai/api` gain `/v1`;
    `https://api.example.com/v1` and Gemini's `.../v1beta/openai` are used as is.
    """
    url = url.rstrip("/")
    return url if re.search(r"/v\d[\w.]*(/openai)?$", url) else url + "/v1"


def _stop_scan(buf: str, stops: list[str]) -> tuple[str, str, bool]:
    """Split streamed text into (emit, hold back, stopped) for client-side stops.

    Holds back only a tail that could still grow into a stop sequence, so text
    that can't is emitted at once.
    """
    cut = min((i for s in stops if (i := buf.find(s)) >= 0), default=-1)
    if cut >= 0:
        return buf[:cut], "", True
    hold = max(
        (k for s in stops for k in range(1, len(s)) if buf.endswith(s[:k])), default=0
    )
    return buf[: len(buf) - hold], buf[len(buf) - hold :], False


class OpenAICompatBackend(LLMBackend):
    """Any server that speaks the OpenAI chat-completions API.

    With `extras`, the body carries vLLM's sampling fields (`top_p`, `top_k`,
    `presence_penalty`, `chat_template_kwargs`) and every stop sequence, which
    vLLM, sglang, llama.cpp and mlx_lm accept. Hosted APIs reject some of them
    with a 400 (Gemini: `top_k`, `presence_penalty`, more than 5 stops), so
    until a request with the extras has succeeded, a 400 or 422 gets one retry
    with standard fields only and at most `STANDARD_MAX_STOP` stops, and the
    backend stays that way. Stops cut from the request are applied here.
    """

    name = "openai"
    # A real vLLM serves one model per process, so list_models() returns a
    # single id and the dropdown is effectively fixed. A router serves many —
    # so allow the switch and validate it against what's served.
    supports_model_switch = True
    supports_pull = False

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        extras: bool = True,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        max_model_len: int = 8192,
        sampling: dict | None = None,
    ):
        super().__init__(model=model, base_url=base_url, sampling=sampling)
        self._root = _api_root(base_url)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._extras = extras
        self._extras_ok = False
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
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
        }
        if self._extras:
            body["top_p"] = s["top_p"]
            body["presence_penalty"] = s["presence_penalty"]
            # vLLM exposes top_k via extra_body when using OpenAI client; the
            # raw HTTP API accepts it at the top level.
            body["top_k"] = s["top_k"]
            # Qwen3 has chain-of-thought on by default; voice TTFB needs it off.
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if stop:
            body["stop"] = stop if self._extras else stop[:STANDARD_MAX_STOP]
        if tools:
            body["tools"] = tools
        if stream:
            # A final chunk with usage stats, so callers can update ctx fills
            # the same way ollama provides them on `done`.
            body["stream_options"] = {"include_usage": True}
        return body

    def _client_stops(self, stop: list[str] | None) -> list[str]:
        """Stop sequences the request couldn't carry, applied to the reply here."""
        return [] if self._extras or not stop else stop[STANDARD_MAX_STOP:]

    def _drop_extras(self, sent_extras: bool, resp: httpx.Response, error: str) -> bool:
        """Switch to standard fields if this 400 may be the extras' fault."""
        if not sent_extras or self._extras_ok or resp.status_code not in (400, 422):
            return False
        # A concurrent request may have switched already; this one retries too.
        if self._extras:
            self._extras = False
            print(
                f"[llm] {self._root} rejected a request with vLLM's sampling fields "
                f"({resp.status_code}: {' '.join(error.split())[:200]}); "
                "using standard OpenAI fields",
                flush=True,
            )
        return True

    async def prefill(self, messages: list[dict]) -> None:
        # Gemini and templates that need a user turn (Qwen3.5) reject a
        # system-only conversation. A placeholder turn after it leaves the
        # system prefix, the part worth caching, unchanged.
        if not any(m.get("role") == "user" for m in messages):
            messages = messages + [{"role": "user", "content": "."}]
        await super().prefill(messages)

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
        client = self._client_inst()
        while True:
            sent_extras = self._extras
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
            client_stops = self._client_stops(stop)
            async with client.stream(
                "POST",
                f"{self._root}/chat/completions",
                json=body,
                headers=self._headers,
            ) as resp:
                if resp.is_error:
                    error = (await resp.aread()).decode(errors="replace")
                    if self._drop_extras(sent_extras, resp, error):
                        continue
                    resp.raise_for_status()
                self._extras_ok = self._extras_ok or sent_extras
                held = ""
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        d = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = d.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        tok = delta.get("content") or ""
                        if tok and client_stops:
                            tok, held, stopped = _stop_scan(held + tok, client_stops)
                            if stopped:
                                if tok:
                                    yield tok
                                return
                        if tok:
                            yield tok
                    if d.get("usage"):
                        self._record_stats(stats, d.get("usage"))
                if held:
                    yield held
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
        client = self._client_inst()
        while True:
            sent_extras = self._extras
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
            client_stops = self._client_stops(stop)
            resp = await client.post(
                f"{self._root}/chat/completions", json=body, headers=self._headers
            )
            if resp.is_error and self._drop_extras(sent_extras, resp, resp.text):
                continue
            resp.raise_for_status()
            self._extras_ok = self._extras_ok or sent_extras
            break
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
        content = msg.get("content", "") or ""
        if client_stops:
            emit, held, _ = _stop_scan(content, client_stops)
            content = emit + held
        return {
            "content": content,
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

    async def health(self) -> bool:
        # /models rather than /health: portable across every
        # OpenAI-compatible server. 401/403 still prove one is listening — but
        # a 404 means whatever is on this port isn't an OpenAI-compatible API,
        # which is a misconfiguration, not a healthy backend.
        try:
            r = await self._client_inst().get(
                f"{self._root}/models", headers=self._headers, timeout=3
            )
            return r.status_code in (200, 401, 403)
        except Exception:
            return False

    async def _served(self) -> list[str] | None:
        try:
            r = await self._client_inst().get(
                f"{self._root}/models", headers=self._headers, timeout=5
            )
            r.raise_for_status()
            return [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
        except Exception:
            return None

    async def list_models(self) -> list[str]:
        return await self._served() or [self.model]

    async def loaded_models(self) -> list[str]:
        # Whatever it serves is loaded — there's no separate resident set.
        return await self.list_models()

    async def set_model(self, name: str) -> None:
        served = await self._served()
        if served and name not in served:
            raise ValueError(
                f"{name!r} is not served by this endpoint (has: {', '.join(served)}). "
                "vLLM serves one model per process — restart it with --model to change."
            )
        self.model = name


class VLLMBackend(OpenAICompatBackend):
    name = "vllm"

    def __init__(
        self,
        model: str = DEFAULT_VLLM_MODEL,
        base_url: str = "http://localhost:8000",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)


class LiteLLMBackend(OpenAICompatBackend):
    """Backend for LiteLLM proxy — routes to 100+ LLM providers.

    Standard fields only: the proxy validates sampling fields against the
    provider behind each route.
    """

    name = "litellm"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "http://localhost:4000",
        **kwargs,
    ):
        kwargs.setdefault("extras", False)
        kwargs.setdefault("max_model_len", 0)
        super().__init__(model, base_url, **kwargs)


def make_backend(name: str | None = None, model: str | None = None) -> LLMBackend:
    name = (name or os.environ.get("VUI_LLM_BACKEND", "ollama")).lower()
    if name == "ollama":
        return OllamaBackend(
            model=model or os.environ.get("VUI_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            # VUI_OLLAMA_URL wins; bare OLLAMA_URL is honoured so the two names
            # that used to diverge now mean the same thing.
            base_url=os.environ.get("VUI_OLLAMA_URL")
            or os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        )
    if name not in ("vllm", "litellm", "openai"):
        raise ValueError(
            f"unknown VUI_LLM_BACKEND: {name!r} "
            "(expected 'ollama', 'vllm', 'litellm' or 'openai')"
        )
    env = f"VUI_{name.upper()}_"
    kwargs: dict = {
        "api_key": os.environ.get(env + "API_KEY") or None,
        "reasoning_effort": os.environ.get(env + "REASONING_EFFORT") or None,
    }
    url = os.environ.get(env + "URL")
    model = model or os.environ.get(env + "MODEL")
    if url:
        kwargs["base_url"] = url
    if model:
        kwargs["model"] = model
    if name == "openai":
        if not url or not model:
            raise ValueError(
                "VUI_LLM_BACKEND=openai needs VUI_OPENAI_URL and VUI_OPENAI_MODEL"
            )
        return OpenAICompatBackend(**kwargs)
    return (VLLMBackend if name == "vllm" else LiteLLMBackend)(**kwargs)


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
