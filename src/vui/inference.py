import io
import re
import threading
import time

import inflect
import julius
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from vui.model import Vui
from vui.sampling import multinomial, sample_top_k, sample_top_p, sample_top_p_top_k
from vui.vad import detect_voice_activity as vad

_prefill_cache = {}
_model_lock = threading.Lock()


def ensure_spaces_around_tags(text: str):
    # Add space before '[' if not preceded by space, '<', or '['
    text = re.sub(
        r"(?<![<\[\s])(\[)",
        lambda m: (
            f"\n{m.group(1)}"
            if m.start() > 0 and text[m.start() - 1] == "\n"
            else f" {m.group(1)}"
        ),
        text,
    )
    # Add space after ']' if not preceded by digit+']' and not followed by space, '>', or ']'
    text = re.sub(
        r"(?<!\d\])(\])(?![>\]\s])",
        lambda m: (
            f"{m.group(1)}\n"
            if m.end() < len(text) and text[m.end()] == "\n"
            else f"{m.group(1)} "
        ),
        text,
    )
    text = text.strip()
    return text


REPLACE = [
    ("—", ","),
    ("'", "'"),
    (":", ","),
    (";", ","),
]

engine = None
wm = None


def asr(chunk, model=None, prefix=None):
    import whisper

    global wm
    if model is not None:
        wm = model
    elif wm is None:
        wm = whisper.load_model("turbo", "cuda")

    chunk = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(chunk, n_mels=wm.dims.n_mels).to(wm.device)
    options = whisper.DecodingOptions(
        language="en", without_timestamps=True, prefix=prefix
    )
    result = whisper.decode(wm, mel[None], options)
    return result[0].text


def replace_numbers_with_words(text):
    global engine

    if engine is None:
        engine = inflect.engine()

    # Function to convert a number match to words
    def number_to_words(match):
        number = match.group()
        return engine.number_to_words(number) + " "

    # Replace digits with their word equivalents
    return re.sub(r"\d+", number_to_words, text)


valid_non_speech = ["breath", "sigh", "laugh", "tut", "hesitate"]
valid_non_speech = [f"[{v}]" for v in valid_non_speech]


def remove_all_invalid_non_speech(txt):
    """
    Remove all non-speech markers that are not in the valid_non_speech list.
    Only keeps valid non-speech markers like [breath], [sigh], etc.
    """
    # Find all text within square brackets
    bracket_pattern = r"\[([^\]]+)\]"
    brackets = re.findall(bracket_pattern, txt)

    # For each bracketed text, check if it's in our valid list
    for bracket in brackets:
        bracket_with_brackets = f"[{bracket}]"
        if bracket_with_brackets not in valid_non_speech and bracket != "pause":
            # If not valid, remove it from the text
            txt = txt.replace(bracket_with_brackets, "")

    return txt


def simple_clean(text):
    text = re.sub(r"(\d+)am", r"\1 AM", text)
    text = re.sub(r"(\d+)pm", r"\1 PM", text)
    text = replace_numbers_with_words(text)
    text = ensure_spaces_around_tags(text)
    text = remove_all_invalid_non_speech(text)

    text = text.replace('"', "")
    text = text.replace("”", "")
    text = text.replace("“", "")
    text = text.replace("’", "'")
    text = text.replace("%", " percent")
    text = text.replace("*", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(";", "")
    text = text.replace("–", " ")
    text = text.replace("—", "")
    text = text.replace(":", "")
    text = text.replace("…", "...")
    text = text.replace("s...", "s")

    # replace repeating \n with just one \n
    text = re.sub(r"\n+", "\n", text)
    ntxt = re.sub(r" +", " ", text)

    # Ensure that ntxt ends with . or ?
    ntxt = ntxt.strip()
    if not ntxt.endswith(".") or ntxt.endswith("?"):
        ntxt += "."
    ntxt += " [pause]"
    return ntxt


def _capture_decode_graph(self: Vui, B: int, Q: int, device):
    """Capture a CUDA graph for single-token decode: embeddings -> decoder -> audio heads."""
    codebook_size = self.config.model.codebook_size + 8

    # Static input buffers
    static_codes = torch.zeros(B, Q, 1, dtype=torch.int64, device=device)
    static_input_pos = torch.zeros(1, dtype=torch.long, device=device)

    # Static output buffer
    static_logits = torch.empty(B, Q, codebook_size, device=device, dtype=torch.bfloat16)

    def _decode_step():
        emb = sum(self.audio_embeddings[q](static_codes[:, q]) for q in range(Q)) / Q
        out = self.decoder(emb, static_input_pos)
        for q in range(Q):
            static_logits[:, q] = self.audio_heads[q](out[:, -1])

    # Warmup runs on a side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _decode_step()
    torch.cuda.current_stream().wait_stream(s)

    # Save KV cache (warmup corrupted it)
    saved = [(b.attn.kv_cache.k_cache.clone(), b.attn.kv_cache.v_cache.clone())
             for b in self.decoder.blocks]

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode_step()

    # Restore KV cache
    for block, (k, v) in zip(self.decoder.blocks, saved):
        block.attn.kv_cache.k_cache.copy_(k)
        block.attn.kv_cache.v_cache.copy_(v)

    return graph, static_codes, static_input_pos, static_logits


def precompute_text(self: Vui, text: str):
    text = remove_all_invalid_non_speech(text)
    text = simple_clean(text)

    if _prefill_cache.get("text") == text:
        return

    if not _model_lock.acquire(blocking=False):
        return

    try:
        with (
            torch.inference_mode(),
            torch.autocast("cuda", torch.bfloat16, True),
            sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]),
        ):
            device = self.device
            Q = self.config.model.n_quantizers
            self.decoder.allocate_inference_cache(1, device, torch.bfloat16)

            encoded = self.tokenizer([text], padding="longest", return_tensors="pt")
            input_ids = encoded.input_ids.to(device)
            text_embeddings = self.token_emb(input_ids)

            T = text_embeddings.size(1)
            input_pos = torch.arange(0, T, device=device)
            out = self.decoder(text_embeddings, input_pos)

            logits = torch.stack(
                [self.audio_heads[q](out[:, -1]) for q in range(Q)], dim=1
            )

            kv_state = []
            for block in self.decoder.blocks:
                kv = block.attn.kv_cache
                kv_state.append((kv.k_cache.clone(), kv.v_cache.clone()))

            self.decoder.deallocate_kv_cache()

        _prefill_cache.clear()
        _prefill_cache.update({"text": text, "kv_state": kv_state, "T": T, "logits": logits})
        print(f"[prefill] cached T={T} for '{text[:40]}...'")
    finally:
        _model_lock.release()


def generate(
    self: Vui,
    text: str,
    prompt_codes: Tensor | None = None,
    temperature: float = 0.5,
    top_k: int | None = 150,
    top_p: float | None = None,
    max_gen_len: int = int(120 * 21.53),
    use_cuda_graph: bool = True,
    yield_every: int | None = None,
):
    gen = _generate_impl(
        self, text, prompt_codes, temperature, top_k, top_p,
        max_gen_len, use_cuda_graph, yield_every,
    )
    if yield_every is not None:
        return gen
    result = None
    for result in gen:
        pass
    return result


def _generate_impl(
    self: Vui,
    text: str,
    prompt_codes: Tensor | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    max_gen_len: int,
    use_cuda_graph: bool,
    yield_every: int | None,
):
    text = simple_clean(text)
    with (
        _model_lock,
        torch.inference_mode(),
        torch.autocast("cuda", torch.bfloat16, True),
        sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]),
    ):
        t1 = time.perf_counter()
        batch_size = 1
        device = self.device
        self.decoder.allocate_inference_cache(batch_size, device, torch.bfloat16)

        B = batch_size
        Q = self.config.model.n_quantizers

        if prompt_codes is None:
            prompt_codes = torch.zeros(
                (batch_size, Q, 0), dtype=torch.int64, device=device
            )
        else:
            prompt_codes = prompt_codes[:, :Q].repeat(batch_size, 1, 1)

        start_offset = prompt_codes.size(-1)

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        unknown_token = -1
        special_token_id = self.config.model.special_token_id

        codes = torch.full(
            (B, Q, max_gen_len), unknown_token, dtype=torch.int64, device=device
        )

        codes[:, :, :start_offset] = prompt_codes

        sequence, indexes, mask = pattern.build_pattern_sequence(
            codes, special_token_id
        )
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        prev_offset = 0
        S = sequence.size(-1)

        do_prefill = True
        eos = self.config.model.audio_eos_id
        graph = None

        cached = _prefill_cache if _prefill_cache.get("text") == text else None

        for offset in range(start_offset_sequence, S):
            curr_sequence = sequence[..., prev_offset:offset]

            if do_prefill:
                if cached:
                    # Restore text KV from precomputed cache
                    for block, (k, v) in zip(self.decoder.blocks, cached["kv_state"]):
                        block.attn.kv_cache.k_cache.copy_(k)
                        block.attn.kv_cache.v_cache.copy_(v)
                    T = cached["T"]

                    audio_embeddings = (
                        sum(self.audio_embeddings[q](curr_sequence[:, q]) for q in range(Q))
                        / Q
                    )
                    if audio_embeddings.size(1) > 0:
                        input_pos = torch.arange(T, T + audio_embeddings.size(1), device=device)
                        out = self.decoder(audio_embeddings, input_pos)
                        T += audio_embeddings.size(1)
                        logits = torch.stack(
                            [self.audio_heads[q](out[:, -1]) for q in range(Q)], dim=1
                        )
                    else:
                        logits = cached["logits"]

                    print(f"[prefill] cache hit, skipped text forward T={cached['T']}")
                else:
                    encoded = self.tokenizer(
                        [text], padding="longest", return_tensors="pt",
                    )
                    input_ids = encoded.input_ids.to(device)
                    text_embeddings = self.token_emb(input_ids)

                    audio_embeddings = (
                        sum(self.audio_embeddings[q](curr_sequence[:, q]) for q in range(Q))
                        / Q
                    )
                    embeddings = torch.cat((text_embeddings, audio_embeddings), dim=1)
                    T = embeddings.size(1)
                    input_pos = torch.arange(0, T, device=device)

                    out = self.decoder(embeddings, input_pos)
                    logits = torch.stack(
                        [self.audio_heads[q](out[:, -1]) for q in range(Q)], dim=1
                    )

                do_prefill = False

                if use_cuda_graph:
                    graph, static_codes, static_input_pos, static_logits = (
                        _capture_decode_graph(self, B, Q, device)
                    )
            else:
                if graph is not None:
                    static_codes.copy_(curr_sequence)
                    static_input_pos.fill_(T)
                    graph.replay()
                    logits = static_logits
                else:
                    audio_embeddings = (
                        sum(self.audio_embeddings[q](curr_sequence[:, q]) for q in range(Q))
                        / Q
                    )
                    input_pos = torch.tensor([T], device=device)
                    out = self.decoder(audio_embeddings, input_pos)
                    logits = torch.stack(
                        [self.audio_heads[q](out[:, -1]) for q in range(Q)], dim=1
                    )
                T += 1

            if offset == start_offset_sequence + 1:
                print("TTFB", time.perf_counter() - t1)

            repetition_penalty = 1.4
            history_window = 12

            for q in range(Q):
                history_start = max(0, offset - history_window)
                token_history = sequence[0, q, history_start:offset]

                unique_tokens = torch.unique(token_history)
                unique_tokens = unique_tokens[unique_tokens != special_token_id]
                unique_tokens = unique_tokens[unique_tokens != eos]
                unique_tokens = unique_tokens[unique_tokens != unknown_token]

                if len(unique_tokens) > 0:
                    logits[0, q, unique_tokens] = (
                        logits[0, q, unique_tokens] / repetition_penalty
                    )

            if offset < 24.53 * 4:
                logits[..., eos] = -float("inf")

            probs = F.softmax(logits / temperature, dim=-1)

            if top_p is not None and top_k is not None:
                next_codes = sample_top_p_top_k(probs, top_p, top_k)
            elif top_p is not None and top_p > 0:
                next_codes = sample_top_p(probs, top_p)
            elif top_k is not None and top_k > 0:
                next_codes = sample_top_k(probs, top_k)
            else:
                next_codes = multinomial(probs, num_samples=1)

            next_codes = next_codes.repeat(batch_size, 1, 1)

            if (probs[..., eos] > 0.95).any():
                print("breaking at", offset)
                break

            valid_mask = mask[..., offset : offset + 1].expand(B, -1, -1)
            next_codes[~valid_mask] = special_token_id

            sequence[..., offset : offset + 1] = torch.where(
                sequence[..., offset : offset + 1] == unknown_token,
                next_codes,
                sequence[..., offset : offset + 1],
            )

            prev_offset = offset

            first_yield = yield_every // 2
            steps_done = offset - start_offset_sequence
            should_yield = (steps_done == first_yield) or (steps_done > first_yield and (steps_done - first_yield) % yield_every == 0)
            if yield_every and should_yield:
                out_codes, _, _ = pattern.revert_pattern_sequence(sequence, special_token=unknown_token)
                out_codes = out_codes[..., prompt_codes.shape[-1]:offset]
                yield out_codes[[0]]

        out_codes, _, _ = pattern.revert_pattern_sequence(
            sequence, special_token=unknown_token
        )
        out_codes = out_codes[..., prompt_codes.shape[-1] : offset]
        yield out_codes[[0]]


@torch.inference_mode()
def render(
    self: Vui,
    text: str,
    prompt_codes: Tensor | None = None,
    temperature: float = 0.5,
    top_k: int | None = 100,
    top_p: float | None = None,
    max_secs: int = 100,
    use_cuda_graph: bool = True,
):
    """
    Render audio from text. Uses generate for text < 1000 characters,
    otherwise breaks text into sections and uses chunking with context.
    """
    text = remove_all_invalid_non_speech(text)
    text = simple_clean(text)
    SR = self.codec.config.sample_rate
    HZ = self.codec.hz
    max_gen_len = int(HZ * max_secs)

    if len(text) < 1000:
        codes = generate(
            self, text, prompt_codes, temperature, top_k, top_p, max_gen_len,
            use_cuda_graph=use_cuda_graph,
        )
        codes = codes[..., :-10]
        audio = self.codec.from_indices(codes)
        paudio = julius.resample_frac(audio[0], 22050, 16000)
        results = vad(paudio)

        if len(results):
            # Cut the audio based on VAD results, add 200ms silence at end
            s, e = results[0][0], results[-1][1]
            return audio[..., int(s * SR) : int((e + 0.2) * SR)].cpu()

        raise Exception("Failed to render")

    # Otherwise we have to do some clever chaining!

    orig_codes = prompt_codes

    lines = text.split("\n")
    audios = []
    prev_codes = prompt_codes
    prev_text = ""

    for i, line in enumerate(lines):
        run = True
        while run:
            current_text = prev_text + "\n" + line if prev_text else line
            current_text = current_text.strip()
            current_text = current_text.replace("...", "")
            current_text = current_text + " [pause]"

            # Calculate max length based on text length
            maxlen = int(HZ * int(60 * len(current_text) / 500))

            try:
                print("rendering", current_text)
                codes = generate(
                    self,
                    current_text,
                    prompt_codes=prev_codes,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_gen_len=maxlen,
                    use_cuda_graph=use_cuda_graph,
                )

                codes = codes[..., :-10]
                audio = self.codec.from_indices(codes)
                # Resample for VAD
                paudio = julius.resample_frac(audio[0], 22050, 16000)

                results = vad(paudio)
                run = len(results) == 0

                if len(results):
                    prev_text = line
                    # Cut the audio based on VAD results, add 200ms silence at end
                    s, e = results[0][0], results[0][1]
                    codes = codes[..., int(s * HZ) : int(e * HZ)]
                    prev_codes = codes
                    audio = audio[..., int(s * SR) : int((e + 0.2) * SR)].cpu()
                    audios.append(audio)
                else:
                    prev_codes = orig_codes
                    prev_text = ""
            except KeyboardInterrupt:
                break
            except RuntimeError as e:
                prev_codes = orig_codes
                prev_text = ""
                print(e)

    return torch.cat(audios, dim=-1)


def _numpy_to_mp3(audio: np.ndarray, sr: int) -> bytes:
    from pydub import AudioSegment
    audio_int16 = (audio * 32767).astype(np.int16)
    seg = AudioSegment(audio_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate="128k")
    return buf.getvalue()


def stream_render(
    self: Vui,
    text: str,
    prompt_codes: Tensor | None = None,
    temperature: float = 0.5,
    top_k: int | None = 100,
    top_p: float | None = None,
    max_secs: int = 100,
    yield_every: int = 44,
):
    text = remove_all_invalid_non_speech(text)
    text = simple_clean(text)
    SR = self.codec.config.sample_rate
    max_gen_len = int(self.codec.hz * max_secs)

    t0 = time.perf_counter()
    gen = generate(
        self, text, prompt_codes, temperature, top_k, top_p,
        max_gen_len, use_cuda_graph=True, yield_every=yield_every,
    )

    OVERLAP = 5
    SAMPLES_PER_CODE = 1024
    DECODE_LEN = yield_every + OVERLAP
    prev_code_len = 0
    chunk_idx = 0
    total_audio = 0.0

    # CUDA graph for fixed-shape codec decode
    codec_graph = None
    static_codec_input = None
    static_codec_output = None

    for codes in gen:
        t_chunk = time.perf_counter()
        codes = codes[..., :-10] if codes.shape[-1] > 10 else codes
        code_len = codes.shape[-1]

        decode_start = max(0, prev_code_len - OVERLAP)
        chunk_codes = codes[..., decode_start:]
        actual_len = chunk_codes.shape[-1]

        if actual_len < DECODE_LEN:
            chunk_codes = F.pad(chunk_codes, (0, DECODE_LEN - actual_len))

        with torch.inference_mode():
            if codec_graph is None:
                static_codec_input = chunk_codes.clone()
                self.codec.from_indices(static_codec_input)
                codec_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(codec_graph):
                    static_codec_output = self.codec.from_indices(static_codec_input)
            static_codec_input.copy_(chunk_codes)
            codec_graph.replay()
            audio = static_codec_output

        valid_samples = actual_len * SAMPLES_PER_CODE
        skip = (prev_code_len - decode_start) * SAMPLES_PER_CODE if prev_code_len > 0 else 0
        new_audio = audio[0, 0, skip:valid_samples].float().cpu().numpy()
        prev_code_len = code_len

        if len(new_audio) > 0:
            t_vocode = time.perf_counter() - t_chunk
            chunk_dur = len(new_audio) / SR
            total_audio += chunk_dur
            elapsed = time.perf_counter() - t0
            if chunk_idx == 0:
                print(f"[stream] TTFB={elapsed:.2f}s chunk={chunk_dur:.2f}s vocode={t_vocode*1000:.0f}ms")
            else:
                print(f"[stream] chunk {chunk_idx}: {chunk_dur:.2f}s vocode={t_vocode*1000:.0f}ms elapsed={elapsed:.2f}s total_audio={total_audio:.2f}s")
            chunk_idx += 1
            yield (SR, new_audio)
            time.sleep(0.01)
