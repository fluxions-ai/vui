"""The OpenAI-compatible backend against servers that accept different fields.

HTTP is mocked: a handler stands in for each server's validation, as observed
on vLLM 0.24, llama.cpp, mlx_lm, Ollama /v1, a LiteLLM proxy and Gemini.
"""

import json

import httpx
import pytest

from vui.serving.stream.llm_backend import (
    OpenAICompatBackend,
    VLLMBackend,
    make_backend,
)

MSGS = [{"role": "user", "content": "hi"}]
STOPS = ["[Results", "[You asked", "\nUser:", "\nYou:", "one sec.", "hold on."]


def _mock(backend, handler):
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return backend


def _strict(seen: list):
    """Gemini-like: rejects top_k and more than 5 stops, answers otherwise."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "top_k" in body or len(body.get("stop", [])) > 5:
            return httpx.Response(400, json={"error": {"message": 'Unknown name "top_k"'}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Sure. hold on. More"}}]}
        )

    return handler


def _sse(*chunks: str) -> httpx.Response:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) for c in chunks
    ]
    return httpx.Response(200, content=("\n\n".join(lines + ["data: [DONE]"])).encode())


@pytest.mark.asyncio
async def test_a_strict_server_gets_one_retry_with_standard_fields():
    seen = []
    backend = _mock(VLLMBackend(), _strict(seen))

    res = await backend.complete(MSGS, stop=STOPS)
    assert len(seen) == 2
    assert "top_k" in seen[0] and "top_k" not in seen[1]
    assert seen[1]["stop"] == STOPS[:4]
    # The cut stops are applied to the reply instead.
    assert res["content"] == "Sure. "

    await backend.complete(MSGS)
    assert len(seen) == 3 and "top_k" not in seen[2]


@pytest.mark.asyncio
async def test_no_fallback_once_the_extras_have_worked():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        return httpx.Response(400, json={"error": {"message": "context too long"}})

    backend = _mock(VLLMBackend(), handler)
    await backend.complete(MSGS)
    with pytest.raises(httpx.HTTPStatusError):
        await backend.complete(MSGS)
    assert len(calls) == 2 and "top_k" in calls[1]


@pytest.mark.asyncio
async def test_stream_applies_the_stops_the_request_could_not_carry():
    def handler(request):
        body = json.loads(request.content)
        if "top_k" in body:
            return httpx.Response(400, json={"error": {"message": "unknown top_k"}})
        assert body["stop"] == STOPS[:4]
        return _sse("Sure, ", "one s", "ec. Here is ", "the data")

    backend = _mock(VLLMBackend(), handler)
    out = [t async for t in backend.stream(MSGS, stop=STOPS)]
    assert "".join(out) == "Sure, "


@pytest.mark.asyncio
async def test_stream_releases_a_held_tail_that_never_became_a_stop():
    backend = _mock(VLLMBackend(extras=False), lambda r: _sse("Okay\n", "Usually yes."))
    out = [t async for t in backend.stream(MSGS, stop=STOPS)]
    assert "".join(out) == "Okay\nUsually yes."


@pytest.mark.asyncio
async def test_the_api_key_goes_on_every_request():
    auth = []

    def handler(request):
        auth.append((request.url.path, request.headers.get("authorization")))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = _mock(
        OpenAICompatBackend("m", "https://openrouter.ai/api/v1", api_key="sk-test"),
        handler,
    )
    await backend.complete(MSGS)
    assert await backend.health()
    assert await backend.list_models() == ["m"]
    assert auth == [
        ("/api/v1/chat/completions", "Bearer sk-test"),
        ("/api/v1/models", "Bearer sk-test"),
        ("/api/v1/models", "Bearer sk-test"),
    ]


@pytest.mark.parametrize(
    ("url", "root"),
    [
        ("http://gpu-box:8000", "http://gpu-box:8000/v1"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
    ],
)
def test_base_url_is_a_bare_host_or_the_documented_api_root(url, root):
    assert OpenAICompatBackend("m", url)._root == root


def test_openai_backend_reads_its_env(monkeypatch):
    monkeypatch.delenv("VUI_OPENAI_URL", raising=False)
    with pytest.raises(ValueError, match="VUI_OPENAI_URL and VUI_OPENAI_MODEL"):
        make_backend("openai")

    monkeypatch.setenv("VUI_OPENAI_URL", "https://api.example.com/v1")
    monkeypatch.setenv("VUI_OPENAI_MODEL", "some-model")
    monkeypatch.setenv("VUI_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VUI_OPENAI_REASONING_EFFORT", "minimal")
    backend = make_backend("openai")
    assert backend.name == "openai"
    assert backend.model == "some-model"
    assert backend._headers == {"Authorization": "Bearer sk-test"}
    body = backend._body(
        MSGS, stream=False, max_tokens=8, temperature=None, top_k=None,
        top_p=None, presence_penalty=None, stop=None,
    )
    assert body["reasoning_effort"] == "minimal"


def test_vllm_backend_reads_an_api_key(monkeypatch):
    monkeypatch.setenv("VUI_VLLM_API_KEY", "token-abc")
    assert make_backend("vllm")._headers == {"Authorization": "Bearer token-abc"}
    monkeypatch.delenv("VUI_VLLM_API_KEY")
    assert make_backend("vllm")._headers == {}


@pytest.mark.asyncio
async def test_prefill_of_a_system_prompt_adds_a_user_turn():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    backend = _mock(VLLMBackend(), handler)
    system = [{"role": "system", "content": "You are Vui."}]
    await backend.prefill(system)
    await backend.prefill(system + MSGS)
    assert seen[0][:1] == system and seen[0][1]["role"] == "user"
    assert seen[1] == system + MSGS


@pytest.mark.asyncio
async def test_concurrent_requests_both_fall_back():
    """The thoughts and conversation LLMs often send together: both 400s retry."""
    import asyncio

    seen, rejected, both_in = [], [], asyncio.Event()

    async def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "top_k" in body:
            rejected.append(body)
            if len(rejected) == 2:
                both_in.set()
            await both_in.wait()
            return httpx.Response(400, json={"error": {"message": "unknown top_k"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = _mock(VLLMBackend(), handler)
    a, b = await asyncio.gather(backend.complete(MSGS), backend.complete(MSGS))
    assert a["content"] == b["content"] == "ok"
    assert len(rejected) == 2 and len(seen) == 4
