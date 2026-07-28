"""Backend-abstraction tests: health, capabilities, model list/switch/pull.

All HTTP is mocked, so these run anywhere — no GPU, no Ollama, no vLLM.
The point is that the management surface behaves sensibly on *both* backends;
it used to be hardcoded to Ollama, which left vLLM with a permanently-red
status pill, an empty model dropdown, and a Pull button that returned 500.
"""

import httpx
import pytest

from vui.serving.stream.llm_backend import OllamaBackend, VLLMBackend, make_backend


def _mock(backend, handler):
    """Point a backend's pooled client at a MockTransport."""
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return backend


def _routes(table, default=404):
    """Build a handler from {path: (status, json)}."""

    def handler(request: httpx.Request) -> httpx.Response:
        entry = table.get(request.url.path)
        if entry is None:
            return httpx.Response(default, json={"error": "not found"})
        status, payload = entry
        return httpx.Response(status, json=payload)

    return handler


# --------------------------------------------------------------- capabilities


def test_capability_flags():
    assert OllamaBackend().supports_model_switch is True
    assert OllamaBackend().supports_pull is True
    # vLLM can switch among ids it already serves, but has no registry.
    assert VLLMBackend().supports_model_switch is True
    assert VLLMBackend().supports_pull is False


def test_unknown_backend_names_the_valid_options():
    with pytest.raises(ValueError, match="expected 'ollama' or 'vllm'"):
        make_backend("bogus")


def test_ollama_url_falls_back_to_bare_env(monkeypatch):
    monkeypatch.delenv("VUI_OLLAMA_URL", raising=False)
    monkeypatch.setenv("OLLAMA_URL", "http://gpu-box.lan:11434")
    assert make_backend("ollama").base_url == "http://gpu-box.lan:11434"

    # VUI_-prefixed wins when both are set.
    monkeypatch.setenv("VUI_OLLAMA_URL", "http://prefixed:11434")
    assert make_backend("ollama").base_url == "http://prefixed:11434"


# --------------------------------------------------------------------- health


@pytest.mark.asyncio
async def test_health_is_backend_dispatched():
    # Ollama answers /api/version; vLLM 404s it. Probing /api/version for both
    # is what used to pin the UI pill to "down" under vLLM.
    ollama = _mock(
        OllamaBackend(), _routes({"/api/version": (200, {"version": "0.32.5"})})
    )
    assert await ollama.health() is True

    vllm = _mock(VLLMBackend(), _routes({"/v1/models": (200, {"data": []})}))
    assert await vllm.health() is True

    # A vLLM-shaped server probed the old way would have failed.
    vllm_no_ollama_ep = _mock(VLLMBackend(), _routes({}))
    assert await vllm_no_ollama_ep.health() is False


@pytest.mark.asyncio
async def test_vllm_health_accepts_auth_gated_endpoint():
    """401 still proves something is listening; only 5xx/connect-error is down."""
    vllm = _mock(VLLMBackend(), _routes({"/v1/models": (401, {"error": "no key"})}))
    assert await vllm.health() is True

    down = _mock(VLLMBackend(), _routes({"/v1/models": (503, {})}))
    assert await down.health() is False


@pytest.mark.asyncio
async def test_health_survives_connection_refused():
    def boom(request):
        raise httpx.ConnectError("refused")

    assert await _mock(OllamaBackend(), boom).health() is False
    assert await _mock(VLLMBackend(), boom).health() is False


# --------------------------------------------------------------- model lists


@pytest.mark.asyncio
async def test_ollama_separates_installed_from_loaded():
    """The dropdown wants /api/tags (installed), not /api/ps (resident)."""
    backend = _mock(
        OllamaBackend(),
        _routes(
            {
                "/api/tags": (200, {"models": [{"name": "a"}, {"name": "b"}]}),
                "/api/ps": (200, {"models": [{"name": "b"}]}),
            }
        ),
    )
    assert await backend.list_models() == ["a", "b"]
    assert await backend.loaded_models() == ["b"]


@pytest.mark.asyncio
async def test_list_models_falls_back_to_current_on_error():
    backend = _mock(OllamaBackend(model="qwen3.5:4b"), _routes({}))
    assert await backend.list_models() == ["qwen3.5:4b"]


@pytest.mark.asyncio
async def test_vllm_lists_served_ids():
    backend = _mock(
        VLLMBackend(),
        _routes({"/v1/models": (200, {"data": [{"id": "Qwen/Qwen3.5-4B"}]})}),
    )
    assert await backend.list_models() == ["Qwen/Qwen3.5-4B"]


# -------------------------------------------------------------- switching


@pytest.mark.asyncio
async def test_ollama_set_model_evicts_the_outgoing_model():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.read()))
        return httpx.Response(200, json={})

    backend = _mock(OllamaBackend(model="old"), handler)
    await backend.set_model("new")

    assert backend.model == "new"
    # Frees the VRAM the previous model held.
    evicts = [b for p, b in seen if p == "/api/generate"]
    assert evicts, f"no evict call, saw: {[p for p, _ in seen]}"
    assert b'"keep_alive":0' in evicts[0]
    assert b'"model":"old"' in evicts[0]


@pytest.mark.asyncio
async def test_ollama_set_model_is_a_noop_for_the_same_model():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    backend = _mock(OllamaBackend(model="same"), handler)
    await backend.set_model("same")
    assert seen == []


@pytest.mark.asyncio
async def test_vllm_rejects_a_model_it_does_not_serve():
    backend = _mock(
        VLLMBackend(model="Qwen/Qwen3.5-4B"),
        _routes({"/v1/models": (200, {"data": [{"id": "Qwen/Qwen3.5-4B"}]})}),
    )
    await backend.set_model("Qwen/Qwen3.5-4B")  # served: fine

    with pytest.raises(ValueError, match="not served by this endpoint"):
        await backend.set_model("some/other-model")
    assert backend.model == "Qwen/Qwen3.5-4B"  # unchanged


# ------------------------------------------------------------------- pulling


@pytest.mark.asyncio
async def test_ollama_pull_streams_progress():
    lines = (
        b'{"status":"pulling","completed":1,"total":10}\n'
        b'{"status":"verifying"}\n'
        b"\n"  # blank lines are skipped
        b'{"status":"success"}\n'
    )

    def handler(request):
        return httpx.Response(200, content=lines)

    backend = _mock(OllamaBackend(), handler)
    got = [ev async for ev in backend.pull("qwen3.5:4b")]
    assert [e["status"] for e in got] == ["pulling", "verifying", "success"]
    assert got[0]["completed"] == 1


@pytest.mark.asyncio
async def test_vllm_pull_raises_rather_than_pretending():
    backend = VLLMBackend()
    with pytest.raises(NotImplementedError, match="no model registry"):
        async for _ in backend.pull("anything"):
            pass
