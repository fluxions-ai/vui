# Rootless install — no sudo, no Docker

`./install.sh --native` completes on a Linux box with neither root nor Docker. This document is
both the how-to and the design record: what blocked a rootless install, why each fix was chosen
over the alternatives, and what remains unverified.

**Implemented.** `file:line` references describe the code *as it was before* this work, since
they're there to explain what was wrong; the fixes are in the tree now. See the CHANGELOG entry
for the summary.

The private upstream's `install.sh` differs from this one only in the docker-compose path, so
everything below ports to it unchanged.

---

## Summary

Exactly two things require root on the native path, and one of them is a mistake:

| # | Blocker | Where | Fix |
|---|---|---|---|
| 1 | `sudo apt install ffmpeg` | `install.sh:192-198`, `README.md:115` | The gate tests the wrong artifact. Replace it with a vendored LGPL shared build + a `ctypes` preload. |
| 2 | `curl https://ollama.com/install.sh \| sh` | `install.sh:207-214` | Stop installing Ollama at all. Use vLLM (pure pip) or an Ollama that already exists. |

Everything else is already unprivileged — see [What already works](#what-already-works-rootless)
at the bottom. Ports, filesystem writes, GPU access, and audio need no special permissions
today.

The second fix has a consequence that dominates the rest of this document: if vLLM becomes the
default LLM backend for a rootless install, it has to actually work properly, and today it
doesn't. Its *hot path* is at parity with Ollama, but its *management surface* is entirely
Ollama-hardcoded. See [Part 3](#part-3--making-vllm-first-class).

---

## Part 0 — Dockerless

The native path exists and works; the problem is that it is neither the default nor the
documented route.

**`README.md:61` makes docker-compose "recommended"**, and the rootless native path is filed
under `## Native install (alternative)` at `README.md:105`. A user without root reads the
recommended path, hits `sudo apt install -y nvidia-container-toolkit` at `README.md:71`, and
concludes Vui isn't for them.

**`install.sh:117-132` silently prefers Docker when non-interactive.** With no TTY it logs
"Docker detected (non-interactive) — using docker compose" and takes the Docker path. The
flagship entry point is `curl -fsSL https://install.fluxions.ai | bash`, where stdin *is* the
pipe — so the auto-pick fires on exactly the install everyone is told to run.

That said, the auto-pick is safer than it looks: `docker_usable()` (`install.sh:109-114`) gates
on `docker info`, which fails for a user who isn't in the `docker` group. So a genuinely
unprivileged box already lands on `MODE="native"` without intervention. **The only user who
gets auto-Docker is one who is in the `docker` group — which is root-equivalent anyway** (the
daemon runs as root and will happily bind-mount `/` for you). There is no case where the
auto-pick escalates someone who couldn't already escalate themselves.

Proposed changes:

- Accept `VUI_MODE=native` as an env alias next to the flag parse (`install.sh:55-67`), so the
  `curl … | bash` form can force native without the `-s --` incantation.
- Reframe the README so the native path is presented as a peer, not an "alternative", and add
  the *Running without root* section described in [Docs](#docs).
- Leave `docker_usable()` alone. It is doing the right thing.

Nothing else about the Docker path needs to change — this document does not propose removing
it.

---

## Part 1 — ffmpeg without root

### The gate tests the wrong thing

```sh
# install.sh:192-198
if ! command -v ffmpeg >/dev/null 2>&1; then
    case "$OS" in
        Linux)  die "ffmpeg not found. Install: sudo apt install ffmpeg" ;;
```

Vui never runs the `ffmpeg` **binary**. The one dependency that could have needed it —
`openai-whisper`'s `load_audio()`, which shells out — is never reached: `src/vui/inference.py:48-63`
calls `whisper.pad_or_trim` / `log_mel_spectrogram` / `decode` on an in-memory tensor.

What Vui actually needs is the ffmpeg **shared libraries**. The wheel ships
`libtorchcodec_core{4,5,6,7,8}.so` — one per supported ffmpeg major, 4 through 8 — and at
import it `dlopen`s them in turn, keeping whichever one loads. Each variant carries `NEEDED`
entries for exactly one ffmpeg generation (verified against the 0.11.1 manylinux wheel):

| variant | `NEEDED` |
|---|---|
| `core4` | `libavutil.so.56`, `libavcodec.so.58`, `libavformat.so.58`, `libavdevice.so.58`, `libavfilter.so.7`, `libswscale.so.5`, `libswresample.so.3` |
| `core5` | `…so.57`, `…so.59`, `…so.59`, `…so.59`, `libavfilter.so.8`, `libswscale.so.6`, `libswresample.so.4` |
| `core6` | `…so.58`, `…so.60`, `…so.60`, `…so.60`, `libavfilter.so.9`, `libswscale.so.7`, `libswresample.so.4` |
| `core7` | `…so.59`, `…so.61`, `…so.61`, `…so.61`, `libavfilter.so.10`, `libswscale.so.8`, `libswresample.so.5` |
| `core8` | `…so.60`, `…so.62`, `…so.62`, `…so.62`, `libavfilter.so.11`, `libswscale.so.9`, `libswresample.so.6` |

So the resolution that has to succeed is ordinary `NEEDED` linking of a `dlopen`ed object, not
an explicit runtime `dlopen` of the ffmpeg libs themselves. That matters for the design below:
it is the *same* mechanism `_preload_nvidia_npp` already exploits, not an analogous one.

Note that all seven libraries are genuinely required — including `libavdevice`, which is easy
to assume is optional.

Consumers: `src/vui/serving/stream/prompt_routes.py:105,353`, `tts_worker.py:1093`,
`tts_worker_mlx.py:533`, `src/vui/demo/cli.py:9-10`, `demo.py:786`.

Two practical consequences of the mismatch:

- The gate **passes** on a box with a static `ffmpeg` on `PATH` while torchcodec still fails.
- The PyPI packages that look like the obvious fix — `ffmpeg-binaries`, `static-ffmpeg`,
  `imageio-ffmpeg` — all ship a **static binary only**. None of them satisfy torchcodec.
- PyAV (`av`, already a dependency) *does* bundle `av.libs/libavcodec-e57b519c.so.62` and
  friends, but the sonames are hash-mangled, so a `dlopen("libavcodec.so.62")` will not match
  them. They cannot be reused.

### Fetch a real shared build

BtbN's FFmpeg-Builds publishes shared LGPL builds for both architectures (asset names verified
to exist):

```
ffmpeg-n7.1-latest-linux64-lgpl-shared-7.1.tar.xz
ffmpeg-n7.1-latest-linuxarm64-lgpl-shared-7.1.tar.xz
```

LGPL rather than GPL: no GPL-only codecs, redistribution-safe, and it still ships
`libav{util,codec,format,filter,device}` plus `libsw{scale,resample}`. `.tar.xz` needs no extra
tooling — `tar -xJf` is universal.

Target directory: `${VUI_FFMPEG_DIR:-~/.cache/vui/ffmpeg}`. `~/.cache/vui/` rather than
`~/.vui/` because the repo already draws that line — `~/.vui/` holds state you'd hate to lose
(TLS certs `tls.py:22`, memories `memories.py:24`, tasks `tasks.py:23`), `~/.cache/vui/` holds
re-downloadable artifacts (`hf.py:9`, `mlx/tts/weights.py:11`). `rm -rf ~/.cache/vui` must stay
a safe thing to do.

Extract to `"$FFMPEG_DIR.tmp.$$"` with `--strip-components=1`, then `mv` into place and write a
`.vui-version` stamp. A killed download then leaves nothing that a later probe would mistake
for a working install.

By hand, that is:

```sh
FFMPEG_DIR="${VUI_FFMPEG_DIR:-$HOME/.cache/vui/ffmpeg}"
mkdir -p "$FFMPEG_DIR"
curl -fL --retry 3 \
  https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-lgpl-shared-7.1.tar.xz \
  | tar -xJ --strip-components=1 -C "$FFMPEG_DIR"

uv run python -m vui.ffmpeg_libs        # torchcodec OK (ffmpeg: .../lib)
```

Nothing needs to go on `PATH` or `LD_LIBRARY_PATH` — `vui/__init__.py` preloads from
`$VUI_FFMPEG_DIR` on import. On arm64 swap `linux64` for `linuxarm64`.

### Building ffmpeg from source instead

Worth documenting as the fallback, because it sidesteps both risks that come with a
third-party binary: BtbN's `latest` is a rolling tag whose bytes change on every autobuild, and
its glibc floor is unknown, so on an older host the prebuilt `.so` files simply won't load.
Building locally also means the libs are matched to the machine that runs them.

It needs a compiler and `make`, but no root — everything lands under the same prefix the
preload already looks in. `--enable-shared --disable-static` is the part that matters;
torchcodec needs the `.so` files, and a default build produces only static archives.
`--disable-everything` plus explicit codecs would be leaner, but the full default build is
about 3 minutes on a modern box and avoids guessing which decoders your prompts need.

```sh
FFMPEG_DIR="${VUI_FFMPEG_DIR:-$HOME/.cache/vui/ffmpeg}"
git clone --depth 1 --branch n7.1 https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg-src
cd /tmp/ffmpeg-src

# --disable-programs: libs only, since Vui never runs the ffmpeg binary.
./configure \
    --prefix="$FFMPEG_DIR" \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --disable-programs \
    --enable-pic
make -j"$(nproc)"
make install

uv run python -m vui.ffmpeg_libs
```

Pick a branch in the 4–8 range; torchcodec ships one `libtorchcodec_core*.so` per major and
will bind to whichever it finds. `n7.1` matches the prebuilt default.

Drop `--disable-programs` if you also want the `ffmpeg`/`ffprobe` binaries for your own use —
Vui doesn't need them, and note that running them *does* require
`LD_LIBRARY_PATH="$FFMPEG_DIR/lib"`, since only the Python side gets the preload.

### If you just want the binary on PATH

Not needed by Vui, and worth stating plainly because it is the intuitive move that doesn't
work: a static `ffmpeg` binary in `~/.local/bin` — from `johnvansickle`, or the
`static-ffmpeg` / `imageio-ffmpeg` / `ffmpeg-binaries` PyPI packages — **will not fix
torchcodec**. It satisfies `command -v ffmpeg` (which is exactly why the current gate is
misleading) while torchcodec still fails, because a static binary exposes no `libavcodec.so.*`
to dlopen. Use one of the two shared-library routes above.

### Reach the libs by preload, not `LD_LIBRARY_PATH`

This is the load-bearing design decision.

`LD_LIBRARY_PATH` only covers processes that `install.sh` itself launches. But `install.sh:254`
and `README.md:136` both tell the user to run `python -m vui.serving.stream` by hand
afterwards, and that is how the server gets restarted from then on — plus `demo.py`, `vui.demo.cli`,
the test suite, and any tmux/systemd wrapper. All of those would lose the variable. It is also
a blunt instrument: it prepends to the search path for *every* child process, so a vendored
`libz` or `libssl` in the ffmpeg build can shadow the system copy for something unrelated.

The alternative is to pull the libraries into the process address space with
`ctypes.CDLL(..., mode=RTLD_GLOBAL)` before torchcodec imports. glibc matches a `dlopen()`
request against already-loaded objects by `DT_SONAME`, so loading them by absolute path first
is enough.

**This is not a new mechanism — the repo already does exactly this.**
`src/vui/__init__.py:4-34` has `_preload_nvidia_npp()`, which solves the identical problem for
torchcodec's NVIDIA NPP dependency chain. The proposal is one sibling function.

New module, stdlib-only and deliberately torch-free (it runs from `vui/__init__.py`, i.e.
before torch, and `install.sh` invokes it as a probe):

```python
# src/vui/ffmpeg_libs.py
"""User-local ffmpeg shared libs for torchcodec (rootless installs).

torchcodec ships libtorchcodec_core{4..8}.so — one per supported ffmpeg major —
with no NEEDED entries; at import it dlopen()s libavutil/libavcodec/... *by
soname* and raises if none of the five variants load. On a box with no root
there is no system ffmpeg to find, so install.sh drops an LGPL shared build
under $VUI_FFMPEG_DIR (default ~/.cache/vui/ffmpeg).

Stdlib only, torch-free: imported from vui/__init__.py before anything else.
Nothing here downloads — install.sh does the fetching.
"""

import ctypes
import os
import sys
from pathlib import Path

# Dependency order matters: the vendored build may carry no RUNPATH, so each
# lib's NEEDED entries have to already be in the loader map when it loads.
_LIBS = ("avutil", "swresample", "swscale", "avcodec",
         "avformat", "avfilter", "avdevice")


def lib_dir() -> Path | None:
    """The vendored lib dir, or None if install.sh never fetched one."""
    root = Path(os.environ.get("VUI_FFMPEG_DIR")
                or Path.home() / ".cache" / "vui" / "ffmpeg")
    d = root / "lib"
    return d if any(d.glob("libavcodec.so.*")) else None


def preload() -> bool:
    """CDLL(RTLD_GLOBAL) the vendored libs so torchcodec's dlopen finds them.

    No-op when no vendored copy exists (system ffmpeg, or macOS).
    """
    if sys.platform != "linux":
        return False
    d = lib_dir()
    if d is None:
        return False
    ok = False
    for stem in _LIBS:
        # libfoo.so.61 (the soname), not libfoo.so.61.19.101 or libfoo.so.
        cands = sorted(d.glob(f"lib{stem}.so.[0-9]*"), key=lambda p: len(p.name))
        if not cands:
            continue
        try:
            ctypes.CDLL(str(cands[0]), mode=ctypes.RTLD_GLOBAL)
            ok = True
        except OSError:
            pass
    return ok
```

Two details that matter and are easy to get wrong: the **dependency ordering** of `_LIBS` (do
not assume the vendored build has a RUNPATH), and preferring the bare-soname symlink
`libavcodec.so.61` over the fully-versioned real file, so the SONAME identity is unambiguous.

Give it a `__main__` self-test that calls `preload()` then imports
`torchcodec.decoders.AudioDecoder`, so `install.sh` can use `uv run python -m vui.ffmpeg_libs`
as the honest post-install check.

Then extend `src/vui/__init__.py`, appended after the existing `_preload_nvidia_npp()` call at
line 33-34 so the diff stays additive (order between the two doesn't matter — they are
independent and both precede any torchcodec import):

```python
def _preload_ffmpeg() -> None:
    """Preload a user-local ffmpeg so torchcodec's runtime dlopen resolves.

    Rootless installs have no system ffmpeg; install.sh caches an LGPL shared
    build under ~/.cache/vui/ffmpeg. No-op when there's a system ffmpeg
    (nothing cached) or on macOS.
    """
    try:
        from vui.ffmpeg_libs import preload
    except Exception:  # never let a preload helper break `import vui`
        return
    preload()


_preload_ffmpeg()
del _preload_ffmpeg
```

### Import-order caveat

`vui/__init__.py` only runs if `vui` is imported *before* torchcodec. That holds for every
production path — the serving call sites are function-level imports inside the package,
`src/vui/demo/cli.py` is a submodule so the package `__init__` runs first, and `demo.py`
imports `vui.inference` at line 24 well before its torchcodec import at line 786.

It does **not** hold in three places that import torchcodec at top of file above the vui
imports:

- `docs/python-api.md:13-14` (and the second block around `:58`)
- `tests/test_python_api_doc.py:16-17`
- `tests/spell_experiment.py:13-14`

These need an explicit `import vui` first. Note that isort/ruff will want to put third-party
`torchcodec` above first-party `vui`, so the fix needs a comment or it will be silently
reverted by a formatter run.

### Where the check goes

Delete the gate at `install.sh:192-198` and move the real check to **after `uv sync`**
(`install.sh:219`) — torchcodec has to be installed before it can be tested.

That is an ordering change. Today's early gate exists to fail fast before a multi-GB torch
download, because the remedy required leaving the script to go run apt. With an automatic
rootless remedy there is nothing to fail fast about. Cost is a few seconds for the probe
(`import torchcodec` pulls in torch); there is no cheaper honest test.

### No `--no-sudo` flag

`install.sh` already never calls `sudo`. The word appears in exactly two places: inside a `die`
message string (`install.sh:194`) and inside the piped Ollama installer. So this work *removes
the two places that outsource root* — it does not add a rootless mode.

That framing kills the flag. A flag or a `sudo -n true` probe would create a second code path
that nobody exercises, to no user-visible benefit; and in the `curl … | bash` entry point a
password prompt can't be answered anyway. Both new helpers no-op when the system already
provides the dependency, so a user with root and `apt install ffmpeg` sees **zero** behaviour
change.

Instead, expose env knobs matching the existing `VUI_REF` / `OLLAMA_HOST` / `VUI_TASK_PORT`
convention (`install.sh:18-21`), and remember there are **two** copies of the help text to keep
in sync — the comment block at `install.sh:18-21` and the heredoc at `install.sh:48-51`:

| Var | Default | Purpose |
|---|---|---|
| `VUI_FFMPEG_DIR` | `~/.cache/vui/ffmpeg` | Where the vendored shared build lands |
| `VUI_FFMPEG_VERSION` | `7.1` | Which BtbN `n<ver>` line to fetch |
| `VUI_MODE` | *(unset)* | `native` to skip the Docker auto-pick |

### Two incidental bugs found in the same area

**`~/.local/bin` is only added to `PATH` inside conditionals.** `install.sh:203` and `:247` do
`export PATH="$HOME/.local/bin:$PATH"` — but only inside the "uv wasn't found" and "claude
wasn't found" branches. On a box where `~/.local/bin` isn't in the login PATH, a *second* run
can't see what the first run installed. Hoist one unconditional export to the top of
`run_native()` (~line 190).

**`--dry-run` isn't side-effect-free.** The gate at `install.sh:192-198` can `die` during a dry
run on a box missing ffmpeg, so `--dry-run` fails to do its one job (print the plan, change
nothing). Both new helpers must return early under `DRY_RUN=1` and print `would fetch <url> →
<dir>` via the existing `run()`/`log()` helpers.

### Idempotency and `--upgrade`

- **Second run, no `--upgrade`:** the ffmpeg helper probes torchcodec, which now succeeds via
  the preload, and returns before touching the network. That is the whole idempotency story,
  and it falls out of using the honest check as the guard rather than a marker file.
- **`--upgrade`:** compare `$FFMPEG_DIR/.vui-version` against the pinned value and re-fetch on
  mismatch. The probe alone can't distinguish "working" from "working but stale".
- **Never clobber a system install.** If the probe passes with `lib_dir() is None`, the libs are
  the system's; `--upgrade` must do nothing.

---

## Part 2 — the LLM, without installing Ollama

### What changes

`install.sh:207-214` — the `curl https://ollama.com/install.sh | sh` block — is **deleted**.
Not replaced with a rootless tarball extraction: deleted. The installer stops installing Ollama.

An Ollama that already exists is still fully supported and is the preferred backend when
present. Only the *installation* goes away. This also removes a wrinkle that a tarball approach
would have inherited: as of v0.32.5 the Linux release assets are `ollama-linux-{amd64,arm64}.tar.zst`
only (1.4–1.5 GB), and Ollama's own installer tells you to `sudo apt-get install zstd` when
`zstd` is missing — so the "rootless" tarball path had a sudo dependency of its own hiding in it.

New flag `--llm vllm|ollama`, defaulting by detection:

1. If an Ollama is reachable (localhost, or `OLLAMA_HOST`), use it as-is. Skip the serve loop
   at `install.sh:221-235` — it's already up — and keep the `ollama pull` at `:237-238` only
   because `pull` honours `OLLAMA_HOST` and works against a remote.
2. Otherwise use vLLM: `VUI_LLM_BACKEND=vllm`.
3. If neither is up, **start anyway** and print how to bring one up.

Point 3 matters and is safe. `server.py:935` binds the HTTP port and prints "Server running at
…" *before* `:959` fire-and-forgets `_warmup()`; the event loop never blocks on the LLM. TTS,
ASR, voice prompts, and the UI all work with no LLM at all; every turn just logs a warning.
Once [R1](#part-3--making-vllm-first-class) lands, the pill turns green by itself when a
backend appears. Dying here would be strictly worse than degrading.

Consequent edits:

| Anchor | Change |
|---|---|
| `install.sh:25` | `MODEL` default `qwen3.5:4b` (an Ollama tag) → `Qwen/Qwen3.5-4B` when the backend is vLLM |
| `install.sh:107` | `ollama_up()` → a backend-aware `llm_up()` probing `/v1/models` or `/api/version` |
| `install.sh:207-214` | deleted |
| `install.sh:221-235` | conditional on an already-present Ollama |
| `install.sh:270-272` | exports become `VUI_LLM_BACKEND` / `VUI_VLLM_URL` / `VUI_VLLM_MODEL` |

### Packaging: sidecar, not an extra

**Do not add `[project.optional-dependencies] vllm`. Do not bump `torchcodec`.**

1. **vLLM's torch pins are exact equality.** vLLM 0.26.0 requires `torch==2.11.0`,
   `torchaudio==2.11.0`, `torchvision==0.26.0`, `torchcodec>=0.14` (fetched from PyPI, not
   recalled). Vui pins `torch==2.11.*` (`pyproject.toml:14`). They agree *today*. The day torch
   2.11.1 ships, `uv sync` floats the base env and the extra becomes unsatisfiable — `uv lock`
   then fails for **everyone**, not just vLLM users. An optional LLM backend must not be able
   to break the base lock.
2. **It would force `torchcodec>=0.14` on every install.** torchcodec is on the TTS/audio hot
   path in four places and is documented public API (`docs/python-api.md:13`). Changing how
   audio decodes for every user as a side effect of an optional LLM backend is the wrong trade.
3. **Vui never imports `vllm`.** `VLLMBackend` speaks HTTP to `${base}/v1/chat/completions`. A
   multi-GB CUDA inference engine inside the app venv buys nothing over one in another venv.
4. **No macOS-arm64 story**, and `uv sync --extra vllm` on a Mac or CPU box either fails or
   wastes gigabytes.

The recommended launch is itself rootless, uv-cached, and cannot perturb Vui's lock:

```sh
# terminal 1 — the LLM
uv run --with 'vllm==0.26.0' --python 3.12 \
    python -m vllm.entrypoints.openai.api_server \
      --model google/gemma-4-E4B-it \
      --max-model-len 8192 \
      --max-num-seqs 1 \
      --enforce-eager \
      --gpu-memory-utilization 0.6 \
      --enable-auto-tool-choice --tool-call-parser gemma4 \
      --port 8000

# terminal 2 — Vui
export VUI_LLM_BACKEND=vllm VUI_VLLM_URL=http://localhost:8000 \
       VUI_VLLM_MODEL=google/gemma-4-E4B-it
uv run python -m vui.serving.stream
```

These are not defaults-with-tweaks; each one is load-bearing for a single user sharing one GPU
with the TTS and ASR workers. The values follow the private repo's `serving/local` launcher,
which solves exactly this case.

- **`--max-num-seqs 1`** — allocate KV for *one* request, not for the many concurrent sequences
  vLLM assumes it's serving. This is the single biggest lever on VRAM after the weights.
  Trade-off worth stating: Vui issues two LLM calls per turn (the reply and the thoughts
  stream), so at `1` they serialise rather than overlap. That is the deliberate choice when the
  goal is minimum footprint; raise it to `2` if you have headroom and want them concurrent.
- **`--enforce-eager`** — skip CUDA graph capture. Saves VRAM and a meaningful slice of startup
  time, at some decode throughput.
- **`--gpu-memory-utilization 0.6`** — vLLM defaults to **0.9 and pre-allocates**, so left
  alone it takes the card and the TTS worker OOMs at `start_workers()`. 0.6 suits a ~9 GiB
  model on a 24 GiB card; it is a starting point to size against your own weights + one
  request's KV, not a universal constant.
- **`--enable-auto-tool-choice --tool-call-parser gemma4`** — required *together*. vLLM refuses
  `tool_choice: "auto"` with a **400 on every request that carries tools** unless both are set
  — not merely the ones the model would answer with a call. The parser is model-specific:
  `gemma4` for gemma-4, `hermes` for Qwen3.

> A caveat carried over from the private stack: even with the right parser, a model whose
> native tool syntax doesn't match vLLM's parser will leak calls into `content` instead of
> `message.tool_calls`. The production agent works around this with JSON-mode prompting
> (`response_format=json_object`, thinking off, temp 0) rather than the OpenAI tools API. Vui's
> stream stack still uses the tools API, so if tool routing misbehaves under a new model, this
> is the first thing to check.

### The honest trade-off

vLLM is the right default for a real GPU box and the wrong one for a small machine. It has no
GGUF quantization, so Qwen3.5-4B in bf16 is ≈8 GB VRAM against Ollama's Q4 ≈2.5 GB; cold start
is minutes (multi-GB HF fetch plus CUDA graph capture) rather than seconds; and it serves one
model per process. A modest machine is better off pointing `VUI_VLLM_URL` or `OLLAMA_HOST` at
another host — which the backend already supports and which costs nothing.

Optionally, `--llm-serve` could launch the sidecar from `install.sh`. Deliberately *not*
recommended for the first pass: cold start is 1–5 minutes against the existing wait loop's 10 s
budget (`install.sh:229`, `seq 1 20` × 0.5 s), VRAM co-tenancy can't be safely guessed, and
`cleanup()` at `install.sh:260` only kills `CLAUDE_PID` — killing a shared inference server on
Ctrl-C would be its own surprise.

---

## Part 3 — making vLLM first-class

Part 2 makes vLLM the default when no Ollama exists, so this was required work rather than a
nice-to-have — a backend you can't see the status of isn't one you can recommend.

The good news: the hot path was already backend-agnostic and vLLM at parity there.
`VLLMBackend.stream()` (`llm_backend.py:384`) handles SSE and sets
`stream_options.include_usage` so usage arrives on the final chunk exactly like Ollama's `done`
frame; `:465-481` normalises OpenAI's `function.arguments` JSON *string* into a dict so callers
can't tell the backends apart; `DEFAULT_SAMPLING` (`:35-40`) deliberately mirrors the
qwen3.5:4b Modelfile so evals don't diverge. `llm.py`, `voice_turn.py`, `thoughts.py`, and
`realtime/inbound.py` all correctly go through `get_backend()`.

The bad news: the management and observability surface bypassed the abstraction entirely. Five
defects, all now fixed.

### Changes, in dependency order

| # | Change | Anchor |
|---|---|---|
| R1 | `health()` on the ABC; `probe_llm` uses it — kills the permanently-red pill | `llm_backend.py:124`, `connection.py:609-618` |
| R2 | Eager backend validation + startup banner; better `ValueError` text | `server.py:1022`, `llm_backend.py:516` |
| R3 | Gate `ensure_mlx_model` on the backend | `connection.py:755-760` |
| R4 | Gate the NUM_PARALLEL warning and the `/api/ps` autodetect on ollama | `connection.py:670`, `:762-775` |
| R5 | `srv.ollama_model` → `srv.llm_model` derived from `backend.model`; delete the dead `model=` params | `server.py:340`, `llm.py:104/120/132/153` |
| R6 | Capability flags; `model_routes` through the backend; `set_model()` actually called; pull 409s not 500s | `llm_backend.py:43`, `model_routes.py:18-110` |
| R7 | UI hides Pull / disables the select from `can_pull`/`can_switch`; shows the backend name | `index.html:284-287`, `:1016`, `:1032`, `:1048` |
| R8 | `test_routes._llm_reply` → `backend.complete()` | `test_routes.py:128-157` |
| R9 | Docs: vLLM quickstart; fix two already-false claims | `docs/configuration.md:87`, `:160` |
| R10 | `install.sh --llm vllm` — covered in Part 2 | `install.sh:207-238`, `:270-272` |

### R1 — the pill is permanently red under vLLM

The single most visible wart, and a nice illustration of the general shape of the problem:

```python
# connection.py:609-618
async def probe_llm() -> bool:
    from vui.serving.stream.llm_backend import get_backend

    base_url = get_backend().base_url          # <- backend-aware!
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{base_url}/api/version")   # <- Ollama-only
            return r.status_code == 200
```

It correctly asks the backend for its URL, then hits an endpoint only Ollama serves. vLLM
404s, `_llm_available` goes False, and `#pill-llm` (`index.html:191`, set at `:635`) stays red
forever even though every reply works.

Add `health()` to the ABC next to `list_models` (`llm_backend.py:124`) with an optimistic
default, override per backend — `/api/version` for Ollama, `/v1/models` for vLLM (portable
across every OpenAI-compatible server, and a 401 still proves the server is up). Then
`probe_llm()` collapses to `return await get_backend().health()`.

### R6 — `list_models()` and `set_model()` already exist and are dead code

`llm_backend.py:124` and `:127` define them; `OllamaBackend` overrides at `:305`/`:315`,
`VLLMBackend` at `:495`. **Neither is called anywhere in `src/vui/`.** The routes at
`model_routes.py:18-110` reimplement them against raw Ollama endpoints. So R6 is mostly wiring
up what is already written.

Two traps:

**The route and the backend method disagree about what "models" means.**
`OllamaBackend.list_models()` uses `/api/ps` (models *currently loaded*);
`model_routes.py:21` uses `/api/tags` (models *installed*). Naively pointing the route at the
backend method would shrink the dropdown to only-loaded models — a UX regression. Split into
`list_models()` (tags) and a new `loaded_models()` (ps).

**There is a latent pre-existing bug here, on Ollama too.** `handle_ollama_set_model`
(`model_routes.py:28-57`) sets `srv.ollama_model = model` and never calls `backend.set_model()`.
Since every `llm_*` helper does `del model` (`llm.py:110,126,144,193`) and reads
`get_backend().model` instead, **the dropdown only changes a label — it does not switch the
model.** The subsequent `llm_prefill_system` then re-warms the *old* model, so the switch
appears to succeed. R6 fixes this for both backends.

Add capability flags to the ABC rather than scattering `if backend.name == "ollama"`:

```python
class LLMBackend:
    name: str = "abstract"
    supports_model_switch: bool = False   # Ollama: yes. vLLM: among served ids only.
    supports_pull: bool = False           # Ollama registry only.
```

`supports_model_switch = True` for vLLM is the right call even though vLLM serves one model per
process: `VLLMBackend` doubles as the generic OpenAI-compatible backend
(`docs/configuration.md:230` sells it as sglang / LM Studio / llama.cpp / OpenAI), where a
router endpoint genuinely does serve several. For a real single-model vLLM, `list_models()`
returns one id, the dropdown has one option, and `change` never fires — zero risk.

Pull should return **409, not 500**, when `supports_pull` is False. It's an expected
capability gap, not a server error.

### R4 — the Ollama-shaped bits that should just be gated

`_probe_ollama_num_parallel()` (`connection.py:48-102`) shells out to
`systemctl show ollama` / `pgrep` + `/proc/<pid>/environ` / `launchctl` to read a specific
daemon's OS-level env var, and warns at `:670-678` that Vui fires two concurrent LLM calls per
turn. This is one of the few places a literal `name == "ollama"` check is *correct* rather than
a smell — it's not an API capability, it's knowledge about a particular daemon. vLLM has
continuous batching and no equivalent concern.

The `/api/ps` autodetect at `connection.py:762-775` is swallowed by a bare
`except Exception: pass`, so under vLLM it fails silently — but leaves `srv.ollama_model` at
its default, which **mislabels the model in `debug_dump/conversation_log.jsonl`** via
`_log_conv` at `voice_turn.py:619,632`. (Worth noting: telemetry does *not* record the model —
`src/vui/telemetry.py` has no model field — so the blast radius is the conversation log only.)

> **Sign-off needed.** Making the autodetect actually call `backend.set_model()` is a *real
> behaviour change on Ollama*, because today it is label-only: the model that answers would
> start depending on what happened to be warm. The conservative option is to delete the
> autodetect entirely and let `VUI_OLLAMA_MODEL` be the one knob. Recommend deleting.

### R5 — `srv.ollama_model` → `srv.llm_model`

15 call sites, but 9 are *fake*: they pass the value into `llm_prefill_system` /
`llm_prefill_user` / `llm_next_chunk` / `llm_stream_chunks`, all of which immediately `del
model` (`llm.py:110,126,144,193`). Delete those vestigial parameters in the same commit and
those 9 sites vanish rather than getting renamed.

The preferred shape is a derived property on `StreamServer`, replacing
`self.ollama_model = DEFAULT_OLLAMA_MODEL` at `server.py:340`:

```python
@property
def llm_model(self) -> str:
    return get_backend().model
```

Then the label physically cannot drift from what actually answers — which is the entire class
of bug behind R4 and R6. Every writer disappears. The JSONL key stays `"model"`, so log
consumers are unaffected.

### R9 — two documentation claims that are already false

`docs/configuration.md:87` says:

> With `VUI_LLM_BACKEND=vllm`, the dropdown shows the served model and switching is disabled.

and `:160` repeats "UI dropdown is read-only". Neither is what the code does. Under vLLM the
dropdown renders **empty** (`/api/tags` 404s → `models = []` at `model_routes.py:22-24`) and
Pull returns **HTTP 500**. R6 and R7 make the docs true rather than requiring a doc rewrite.

Separately, `docs/configuration.md:85` claims the dropdown lists `/api/ps`; the route actually
uses `/api/tags`. And `docs/configuration.md:160`'s Apple Silicon note — see the risks section.

### R8 — `test_routes.py`

`_llm_reply` (`test_routes.py:128-157`) hand-rolls `POST {OLLAMA_URL}/api/chat` with an
Ollama-shaped body (`keep_alive`, `think`, `options.num_ctx`) and hard-errors under vLLM. The
docstring at `:129-136` justifies bypassing the *chunker* — and `backend.complete()` is exactly
"the full reply in one call", so routing through the backend honours that intent while removing
the error. All three Ollama-specific body fields are already produced by
`OllamaBackend._body`/`_options`.

One real difference to flag: `complete()` defaults `temperature=0.0` (`llm_backend.py:109`)
whereas the hand-rolled body sends none and lets the Modelfile decide. Passing
`temperature=None` routes through `_resolve_sampling` to `DEFAULT_SAMPLING["temperature"]=1.0`,
which is the Modelfile value — the closest match. **Any recorded benchmark numbers from these
test routes may shift; re-baseline after the change.**

### Also done, having become cheap once the above landed

- **`OLLAMA_URL` / `VUI_OLLAMA_URL` unification.** These used to address genuinely separate code
  paths — `OLLAMA_URL` (`llm.py:33`) for the model-list/pull helpers, `VUI_OLLAMA_URL` for the
  chat path — so setting only one half-worked, with the dropdown listing a different set of
  models than the one answering. Once the routes went through the backend, the only direct
  readers left were `_ollama_running` / `_ollama_model_exists`, both used solely by
  `ensure_mlx_model`. `make_backend()` now takes `VUI_OLLAMA_URL or OLLAMA_URL`, so either
  alone is sufficient and existing setups keep working.
- **`/ollama/*` → `/llm/*` route rename**, with the old paths aliased to the same handlers for
  one release. `index.html` was the only consumer.
- **`tests/test_llm_backend.py` + `tests/test_model_routes.py`** using `httpx.MockTransport` and
  fake backends. Worth noting these caught two real bugs in the work above: vLLM's `health()`
  counted a 404 on `/v1/models` as healthy (so a misconfigured base URL showed green), and a
  non-2xx model list returned `[]` rather than falling back, which would have left the dropdown
  empty — the exact symptom the change set out to fix.

### Still deferred

- `VUI_VLLM_MAX_MODEL_LEN` — `VLLMBackend.max_model_len` defaults to 8192 and feeds `ctx_max`,
  so the UI context bars are wrong if vLLM was started with a different `--max-model-len`.
- The MLX finding in risk 9 below.

---

## Docs

- **`README.md:59`** — the bootstrap blurb lists "installs deps (uv, Ollama, ffmpeg, Claude
  Code CLI)". Ollama comes out; add "— no sudo required".
- **`README.md:112-118`** — the paragraph is nearly right but names the wrong artifact.
  Rewrite: the dependency is the ffmpeg *shared libraries*, which torchcodec dlopens at
  runtime; the `ffmpeg` binary is not used, so `static-ffmpeg` / `imageio-ffmpeg` /
  `ffmpeg-binaries` do **not** satisfy it. `sudo apt install ffmpeg` stays as the one-liner for
  people who do have root.
- **`README.md`, new `### Running without root`** before `### TTS demo on its own`: everything
  lands in `$HOME`; ports are all >1024; GPU needs no group membership; audio is WebRTC in the
  browser; the disk budget and the redirect knobs (`HF_HOME`, `UV_CACHE_DIR`, `VUI_FFMPEG_DIR`)
  for quota'd homes.
- **`docs/configuration.md`** — fix `:85`, `:87`, `:160`; add `VUI_FFMPEG_DIR` to the env table.
- **`cpu/README.md:20`** — `gcc … -lopenblas` needs `libopenblas-dev`. Document, don't fix; the
  whole `cpu/` tree is an optional side path that nothing in `install.sh` touches. Rootless
  alternative: the `scipy-openblas64` wheel ships both the shared object and headers via
  `get_include_dir()` / `get_lib_dir()` — but the library is named `libscipy_openblas`, so
  `-lopenblas` won't find it; you need `-l:libscipy_openblas.so` plus an rpath.
  **Unverified — not compiled.**
- **`docs/mobile.md:33`** — `tailscale serve --https=443` needs root, and the usual workaround
  (`tailscale set --operator=$USER`) itself needs root once. The genuinely rootless option is
  the cloudflared path documented just above it (static binary, drop in `~/.local/bin`), with
  the existing caveat that its TCP-only tunnel can't carry WebRTC media off-LAN. State the
  trade-off rather than implying tailscale is free.
- **`CHANGELOG.md`** — an `## [Unreleased]` entry: the ffmpeg gate tested the wrong artifact;
  the native install no longer requires root.

---

## What already works rootless

Audited and confirmed — no changes needed to any of this.

| Area | Finding |
|---|---|
| **Ports** | 8080 HTTP (`server.py:933`), 8443 HTTPS (`:945`), 8642 Claude task server (`claude_server.py:435`), 11434 Ollama, plus ephemeral UDP for WebRTC ICE. All >1024. |
| **Filesystem** | Nothing under `/usr`, `/etc`, `/var`, `/opt` is ever written. State in `~/.vui/`, caches in `~/.cache/`, logs in `/tmp`. Two guarded *reads* of system paths in `src/vui/geo.py:20-21` for timezone detection. |
| **GPU** | `/dev/nvidia*` are `crw-rw-rw-`; no `render`/`video` group needed. The entire CUDA userspace comes from pip wheels (`nvidia-cublas-cu12`, `pyproject.toml:47`, plus the NPP preload at `src/vui/__init__.py:4-32`). |
| **Audio** | The server needs no audio device at all — everything is WebRTC and the browser owns the mic and speaker. Zero hits for `sounddevice`/`pyaudio`/`/dev/snd` under `src/vui/serving/`. Local-device use is confined to `cpu/` and to `src/vui/demo/cli.py:117-119` (which shells out to SoX `play`). |
| **uv, Claude CLI** | Both install to `~/.local/bin` (`install.sh:200-205`, `:241-248`). |
| **Native wheels** | `av`, `soundfile`, `onnxruntime`, and `ctranslate2` all bundle their native dependencies. `torchcodec` is the sole exception — see Part 1. |

---

## Risks and unverified items

Listed rather than buried, because several of these would change the plan if they turn out
wrong.

1. ~~The dlopen-by-soname assumption is untested.~~ **Verified.** On a box whose system ffmpeg
   is libavcodec **.60**, pointing `VUI_FFMPEG_DIR` at an extracted n7.1 build and importing
   `vui` then `torchcodec` maps *only* the vendored libs:

   ```
   /tmp/ffmpeg-test/lib/libavcodec.so.61.19.101
   /tmp/ffmpeg-test/lib/libavdevice.so.61.3.100      (+ avfilter, avformat,
   /tmp/ffmpeg-test/lib/libavutil.so.59.39.100        avutil, swresample, swscale)
   ```

   torchcodec selected `core7` and bound to the vendored `.61` in preference to the system
   `.60`, and an `AudioEncoder`→`AudioFile`→`AudioDecoder` roundtrip succeeds. With
   `VUI_FFMPEG_DIR` unset or empty, `preload()` returns False and torchcodec falls back to the
   system `/usr/lib/x86_64-linux-gnu/libavcodec.so.60` as before — so the change is inert on
   machines that already work. Not yet run on a box with *no* system ffmpeg at all.
2. **BtbN's `latest` is a rolling tag.** The asset *name* is stable but its bytes change on
   every autobuild. Recommend a `VUI_FFMPEG_RELEASE` knob (default `latest`, pinnable to a
   dated `autobuild-YYYY-MM-DD` tag) so a broken upstream build is an env-var fix, not a patch
   release.
3. **The glibc floor of the BtbN builds is unknown.** On an older host `ctypes.CDLL` fails with
   `GLIBC_2.xx not found`, which would currently surface as a silent no-op inside `preload()`
   followed by a confusing torchcodec error. The post-fetch re-probe must capture stderr and
   print the actual `CDLL` error alongside the remedies.
4. ~~Exactly which libav\* torchcodec needs is unverified.~~ **Resolved** — read off the wheel's
   `NEEDED` entries; see the table in Part 1. All seven, `libavdevice` included.
5. ~~The tool-parser name for vLLM is unverified.~~ **Resolved** — `gemma4` for
   `google/gemma-4-E4B-it`, taken from the private repo's `deploy/fleet.yaml` (`tool_parser:
   gemma4`), which is what production serves. `hermes` is the Qwen3 answer. Note the failure
   mode is worse than "silent degradation": without both flags vLLM 400s every tools-bearing
   request.
6. **vLLM/TTS VRAM co-tenancy on one GPU is unmeasured for gemma-4-E4B specifically.** The
   `0.6` / `--max-num-seqs 1` / `--enforce-eager` combination is lifted from the private repo's
   `serving/local/launcher.py`, where the 0.6 is documented against Qwen3.5-4B (~8.6 GiB) on a
   24 GiB 4090. Re-measure for a different model or card.
7. **`VLLMBackend._body` sends `top_k` at top level and `chat_template_kwargs`**
   (`llm_backend.py:364-366`). vLLM accepts both; **OpenAI proper rejects unknown fields with a
   400**, so `docs/configuration.md:230`'s claim that "OpenAI itself" works is probably false.
   Not blocking for vLLM.
8. **Whether vLLM's `/v1/models` returning 200 implies the engine is ready** — believed yes (it
   binds the port after engine init) but not verified. If not, the pill goes green early and the
   first turn errors; the poll loop recovers.
9. **Apple Silicon: the MLX model is built and then never used.** `llm.py:39` sets
   `DEFAULT_OLLAMA_MODEL = MLX_MODEL_NAME`, but `make_backend()` (`llm_backend.py:513`) defaults
   to `qwen3.5:4b` and every `llm_*` helper reads `get_backend().model`. So the multi-GB
   download and int4 quantize at `connection.py:757` produce a model the request path never
   references, and the README's "~1.9× faster decode, recommended" claim looks untrue on a stock
   Mac install. **Derived statically — needs a Mac to confirm.** Orthogonal to this work but
   found during it.
10. **macOS is unchanged and still needs Homebrew for ffmpeg** — BtbN publishes no mac builds.
    Homebrew in a user prefix is itself rootless, so this is not a new gap, but `--native` on a
    Mac with no brew still dies. Out of scope.
11. **Disk, not root, becomes the top failure mode** on the shared/HPC boxes this targets:
    ~5 GB of torch/CUDA wheels plus HF checkpoints plus (for vLLM) the model, all in `$HOME`.
    Hence the `df` preflight and the redirect knobs.
