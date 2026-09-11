"""LLM model routes, exercised against fake backends (no GPU, no live server).

These cover the behaviours that were broken before the routes went through the
LLMBackend abstraction: an empty dropdown and a 500 from Pull under vLLM, and a
model switch that only ever moved a label.
"""

import pytest

from vui.serving.stream import model_routes


class FakeBackend:
    name = "fake"
    supports_model_switch = True
    supports_pull = True

    def __init__(self, **kw):
        self.model = kw.get("model", "m1")
        self.models = kw.get("models", ["m1", "m2"])
        self.set_calls = []
        self.prefill_should_fail = False
        for k, v in kw.items():
            setattr(self, k, v)

    async def list_models(self):
        return self.models

    async def set_model(self, name):
        if name not in self.models:
            raise ValueError(f"{name!r} is not served by this endpoint")
        self.set_calls.append(name)
        self.model = name

    async def pull(self, name):
        for ev in ({"status": "pulling", "completed": 5, "total": 10},
                   {"status": "success"}):
            yield ev


class FakeSession:
    soul = "you are a test"
    ws = None


class FakeServer:
    def __init__(self, backend):
        self._backend = backend
        self.session = FakeSession()
        self.logs = []
        self.blocked = []

    @property
    def llm_model(self):
        return self._backend.model

    async def _block_ready(self, k):
        self.blocked.append(("block", k))

    async def _unblock_ready(self, k):
        self.blocked.append(("unblock", k))

    async def _log(self, msg, level="info"):
        self.logs.append((level, msg))


class FakeRequest:
    def __init__(self, payload=None):
        self._payload = payload or {}

    async def json(self):
        return self._payload


@pytest.fixture
def patched(monkeypatch):
    """Install a fake backend + no-op prefill into model_routes."""

    def _install(backend):
        monkeypatch.setattr(model_routes, "get_backend", lambda: backend)

        async def _prefill(_soul):
            if getattr(backend, "prefill_should_fail", False):
                raise RuntimeError("model failed to load")

        monkeypatch.setattr(model_routes, "llm_prefill_system", _prefill)
        return FakeServer(backend)

    return _install


def _body(resp):
    import json

    return json.loads(resp.body.decode())


# ------------------------------------------------------------------ listing


@pytest.mark.asyncio
async def test_models_route_reports_capabilities(patched):
    backend = FakeBackend(models=["a", "b"], model="a")
    srv = patched(backend)
    data = _body(await model_routes.handle_llm_models(srv, FakeRequest()))

    assert data["models"] == ["a", "b"]
    assert data["current"] == "a"
    assert data["backend"] == "fake"
    # The UI keys its Pull/switch controls off these.
    assert data["can_switch"] is True
    assert data["can_pull"] is True


@pytest.mark.asyncio
async def test_models_route_survives_a_broken_backend(patched):
    backend = FakeBackend(model="only-me")

    async def boom():
        raise RuntimeError("down")

    backend.list_models = boom
    srv = patched(backend)
    data = _body(await model_routes.handle_llm_models(srv, FakeRequest()))
    # Never an empty dropdown.
    assert data["models"] == ["only-me"]


# ---------------------------------------------------------------- switching


@pytest.mark.asyncio
async def test_set_model_actually_switches_the_backend(patched):
    """The old route set a label and never told the backend."""
    backend = FakeBackend(models=["m1", "m2"], model="m1")
    srv = patched(backend)
    resp = await model_routes.handle_llm_set_model(srv, FakeRequest({"model": "m2"}))

    assert _body(resp)["ok"] is True
    assert backend.set_calls == ["m2"]
    assert backend.model == "m2"
    assert srv.llm_model == "m2"


@pytest.mark.asyncio
async def test_set_model_reverts_when_the_new_model_wont_load(patched):
    backend = FakeBackend(models=["m1", "m2"], model="m1")
    backend.prefill_should_fail = True
    srv = patched(backend)
    resp = await model_routes.handle_llm_set_model(srv, FakeRequest({"model": "m2"}))

    assert _body(resp)["ok"] is False
    assert backend.model == "m1", "should have reverted"


@pytest.mark.asyncio
async def test_set_model_409s_when_the_backend_cannot_switch(patched):
    backend = FakeBackend()
    backend.supports_model_switch = False
    srv = patched(backend)
    resp = await model_routes.handle_llm_set_model(srv, FakeRequest({"model": "x"}))

    assert resp.status == 409
    assert backend.set_calls == []


@pytest.mark.asyncio
async def test_set_model_400s_on_an_unserved_model(patched):
    backend = FakeBackend(models=["m1"], model="m1")
    srv = patched(backend)
    resp = await model_routes.handle_llm_set_model(srv, FakeRequest({"model": "nope"}))

    assert resp.status == 400
    assert "not served" in _body(resp)["error"]


@pytest.mark.asyncio
async def test_set_model_always_unblocks_ready(patched):
    backend = FakeBackend(models=["m1"], model="m1")
    srv = patched(backend)
    await model_routes.handle_llm_set_model(srv, FakeRequest({"model": "nope"}))
    assert ("unblock", "llm") in srv.blocked


# ------------------------------------------------------------------ pulling


@pytest.mark.asyncio
async def test_pull_409s_when_there_is_no_registry(patched):
    """vLLM has nothing to pull from — that's expected, not a server fault."""
    backend = FakeBackend()
    backend.supports_pull = False
    srv = patched(backend)
    resp = await model_routes.handle_llm_pull(srv, FakeRequest({"model": "x"}))

    assert resp.status == 409
    assert "no model registry" in _body(resp)["error"]


@pytest.mark.asyncio
async def test_pull_switches_to_the_pulled_model(patched):
    backend = FakeBackend(models=["m1", "new"], model="m1")
    srv = patched(backend)
    resp = await model_routes.handle_llm_pull(srv, FakeRequest({"model": "new"}))

    assert _body(resp)["ok"] is True
    assert backend.model == "new"
