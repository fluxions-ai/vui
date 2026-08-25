"""Tests for LiteLLM backend."""

from vui.serving.stream.llm_backend import LiteLLMBackend, make_backend


def test_make_backend_litellm():
    backend = make_backend("litellm", model="anthropic/claude-sonnet-4-6")
    assert isinstance(backend, LiteLLMBackend)
    assert backend.name == "litellm"
    assert backend.model == "anthropic/claude-sonnet-4-6"
    assert backend.base_url == "http://localhost:4000"


def test_litellm_body_includes_drop_params():
    backend = LiteLLMBackend(model="openai/gpt-4o-mini")
    body = backend._body(
        [{"role": "user", "content": "hi"}],
        stream=False,
        max_tokens=100,
        temperature=0.5,
        top_k=None,
        top_p=None,
        presence_penalty=None,
        stop=None,
    )
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["max_tokens"] == 100
    assert "presence_penalty" not in body


def test_litellm_body_with_tools():
    backend = LiteLLMBackend()
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    body = backend._body(
        [{"role": "user", "content": "weather?"}],
        stream=False,
        max_tokens=100,
        temperature=None,
        top_k=None,
        top_p=None,
        presence_penalty=None,
        stop=None,
        tools=tools,
    )
    assert body["tools"] == tools


def test_litellm_default_model():
    backend = LiteLLMBackend()
    assert backend.model == "openai/gpt-4o-mini"


def test_litellm_set_model():
    import asyncio

    backend = LiteLLMBackend()
    asyncio.run(backend.set_model("anthropic/claude-haiku-4-5"))
    assert backend.model == "anthropic/claude-haiku-4-5"
