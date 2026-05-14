"""Minimal streaming-state machinery, modeled after Moshi/Mimi.

Each `StreamingModule` carries a `_streaming_state` while inside a
`module.streaming(batch_size)` context. State is allocated once on
entry, mutated in-place across calls (so CUDA graph capture sees stable
tensors), and torn down on exit.

`forward()` is overloaded by state presence: same call site works for
offline (one-shot) and streaming (chunked, stateful) modes. No `step()`
API.

Usage:

    class MyConv(StreamingModule):
        def _init_streaming_state(self, B):
            return _State(B, device, prev=torch.zeros(B, C, K-1, ...))

        def forward(self, x):
            state = self._streaming_state
            if state is None:
                return self.conv(F.pad(x, (K-1, 0)))
            x = torch.cat([state.previous, x], dim=-1)
            state.previous.copy_(x[..., -(K-1):])
            return self.conv(x)

    with model.streaming(batch_size=1):
        for chunk in chunks:
            out.append(model(chunk))
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class State:
    batch_size: int
    device: torch.device

    def reset(self) -> None:
        """Reset state to its just-initialized form (e.g. zero buffers)."""


class StreamingModule(nn.Module):
    """nn.Module that may carry a `_streaming_state` inside `streaming(B)`."""

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State | None = None

    @property
    def is_streaming(self) -> bool:
        return self._streaming_state is not None

    def _walk_streaming(self, fn) -> None:
        def _go(module: nn.Module) -> None:
            if isinstance(module, StreamingModule):
                fn(module)
            for child in module.children():
                _go(child)

        _go(self)

    def _init_streaming_state(self, batch_size: int) -> State | None:
        return None

    def streaming(self, batch_size: int) -> ExitStack:
        def _start(module: StreamingModule) -> None:
            assert module._streaming_state is None, "already streaming"
            module._streaming_state = module._init_streaming_state(batch_size)

        def _stop_all() -> None:
            self._walk_streaming(lambda m: setattr(m, "_streaming_state", None))

        stack = ExitStack()
        self._walk_streaming(_start)
        stack.callback(_stop_all)
        return stack

    def reset_streaming(self) -> None:
        def _reset(module: StreamingModule) -> None:
            if module._streaming_state is not None:
                module._streaming_state.reset()

        self._walk_streaming(_reset)


class StreamingContainer(StreamingModule):
    """Parent that holds streaming children but no state of its own.

    Returns a bare `State` so `is_streaming` reflects context activity —
    callers (e.g. `QwenCodecDecoder.forward`) gate the graph fast path on
    `self._streaming_state is not None`.
    """

    def _init_streaming_state(self, batch_size: int) -> State:
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return State(batch_size, device)
