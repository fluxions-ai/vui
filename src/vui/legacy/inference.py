"""Inference for the original-release (pre-1.0) checkpoints.

Ported from the pre-1.0 tree with three changes: device-agnostic autocast
(bfloat16 on CUDA, model dtype elsewhere — so it runs on Apple Silicon / CPU),
text cleaning reused from vui.inference, and VAD trimming made optional
(the old pyannote dependency is gone; without it the full render is returned).
"""

import time

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from vui.inference import remove_all_invalid_non_speech, simple_clean
from vui.legacy.model import Vui
from vui.legacy.sampling import (
    multinomial,
    sample_top_k,
    sample_top_p,
    sample_top_p_top_k,
)


def _autocast():
    if torch.cuda.is_available():
        return torch.autocast("cuda", torch.bfloat16, True)
    from contextlib import nullcontext

    return nullcontext()


def _kv_dtype(model: Vui) -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else model.dtype


def _try_vad(audio_16k: Tensor) -> list:
    """Old pipeline trimmed with pyannote VAD; treat it as optional."""
    try:
        from vui.legacy.vad import detect_voice_activity

        return detect_voice_activity(audio_16k)
    except ImportError:
        return []


@torch.inference_mode()
def generate(
    self: Vui,
    text: str,
    prompt_codes: Tensor | None = None,
    temperature: float = 0.5,
    top_k: int | None = 150,
    top_p: float | None = None,
    max_gen_len: int = int(120 * 21.53),
    decoder=None,
):
    dec = decoder if decoder is not None else self.decoder
    text = simple_clean(text)
    if decoder is None and self.device.type == "mps":
        raise RuntimeError(
            "The legacy model produces corrupted audio on MPS (and is slower "
            "than CPU there). Keep it on CPU — already faster than real-time "
            "— or CUDA."
        )
    with _autocast(), sdpa_kernel([SDPBackend.MATH]):
        t1 = time.perf_counter()
        batch_size = 1
        device = self.device
        dec.allocate_inference_cache(batch_size, device, _kv_dtype(self))

        encoded = self.tokenizer([text], padding="longest", return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        text_embeddings = self.token_emb(input_ids)

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

        for offset in range(start_offset_sequence, S):
            curr_sequence = sequence[..., prev_offset:offset]
            audio_embeddings = (
                sum([self.audio_embeddings[q](curr_sequence[:, q]) for q in range(Q)])
                / Q
            )

            if do_prefill:
                embeddings = torch.cat((text_embeddings, audio_embeddings), dim=1)
                T = embeddings.size(1)
                input_pos = torch.arange(0, T, device=device)
                do_prefill = False
            else:
                embeddings = audio_embeddings
                input_pos = torch.tensor([T], device=device)
                T += 1

            out = dec(embeddings, input_pos)

            if offset == 15:
                print("TTFB", time.perf_counter() - t1)

            logits = torch.stack(
                [self.audio_heads[q](out[:, -1]) for q in range(Q)], dim=1
            )

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

        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(
            sequence, special_token=unknown_token
        )
        out_codes = out_codes[..., prompt_codes.shape[-1] : offset]
        return out_codes[[0]]


@torch.inference_mode()
def render(
    self: Vui,
    text: str,
    prompt_codes: Tensor | None = None,
    temperature: float = 0.5,
    top_k: int | None = 100,
    top_p: float | None = None,
    max_secs: int = 100,
    decoder=None,
):
    """Render audio from text. Returns (1, 1, S) float audio at the codec
    sample rate (22050). Long texts (>1000 chars) are chunked line-by-line
    with rolling code context, as in the original release."""
    text = remove_all_invalid_non_speech(text)
    text = simple_clean(text)
    SR = self.codec.config.sample_rate
    HZ = self.codec.hz
    max_gen_len = int(HZ * max_secs)

    if len(text) < 1000:
        codes = generate(
            self, text, prompt_codes, temperature, top_k, top_p, max_gen_len,
            decoder=decoder,
        )
        codes = codes[..., :-10]
        audio = self.codec.from_indices(codes)
        paudio = torchaudio.functional.resample(audio[0], SR, 16000)
        results = _try_vad(paudio)
        if results:
            s, e = results[0][0], results[-1][1]
            return audio[..., int(s * SR) : int((e + 0.2) * SR)].cpu()
        return audio.cpu()

    lines = text.split("\n")
    audios = []
    prev_codes = prompt_codes
    orig_codes = prompt_codes
    prev_text = ""

    for line in lines:
        run = True
        while run:
            current_text = prev_text + "\n" + line if prev_text else line
            current_text = current_text.strip().replace("...", "") + " [pause]"
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
                    decoder=decoder,
                )
                codes = codes[..., :-10]
                audio = self.codec.from_indices(codes)
                paudio = torchaudio.functional.resample(audio[0], SR, 16000)
                results = _try_vad(paudio)
                run = False

                if results:
                    prev_text = line
                    s, e = results[0][0], results[0][1]
                    codes = codes[..., int(s * HZ) : int(e * HZ)]
                    prev_codes = codes
                    audios.append(audio[..., int(s * SR) : int((e + 0.2) * SR)].cpu())
                else:
                    prev_text = line
                    prev_codes = codes
                    audios.append(audio.cpu())
            except KeyboardInterrupt:
                break
            except RuntimeError as e:
                prev_codes = orig_codes
                prev_text = ""
                print(e)

    return torch.cat(audios, dim=-1)
