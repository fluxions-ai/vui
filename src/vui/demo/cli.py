"""One-shot CLI renderer: text -> audio file with streaming playback."""

import datetime
import subprocess
import time
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from vui.inference import InferenceState, asr, render_audio_stream, simple_clean
from vui.qwen_codec import SAMPLE_RATE as QWEN_SR
from vui.qwen_codec import QwenCodecDecoder, QwenCodecEncoder


def run(
    checkpoint_path: str,
    prompt_file: str = "prompts/harry.wav",
    text: str | None = None,
    **overrides,
):
    if text is None:
        raise SystemExit("--render requires text: python demo.py --render [ckpt] \"text\"")
    from vui.model import Vui

    torch.set_float32_matmul_precision("high")

    print(f"Loading model from {checkpoint_path}...")
    model = Vui.from_pretrained_inf(checkpoint_path).cuda()

    codec_dec = QwenCodecDecoder.from_pretrained().cuda().float().eval()
    Q = model.config.model.n_quantizers

    # --- Settings ---
    settings = {
        "temperature": 0.9,
        "max_secs": 30.0,
        "eos_threshold": 0.8,
        "n_codebooks": Q,
    }
    settings.update(overrides)
    sq = (3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0)

    print("Setting up CUDA graphs...")
    with torch.inference_mode():
        state = InferenceState(
            model, codec_dec, sq_scores=sq, wps_score=0.0, codec_graphs=False
        )

    # --- Load prompt ---
    prompt_codes = None
    prompt_text = None

    def load_prompt(pf: str):
        nonlocal prompt_codes, prompt_text
        codec_enc = QwenCodecEncoder.from_pretrained().cuda().half().eval()
        from julius.resample import resample_frac

        wav = AudioDecoder(pf, sample_rate=16000, num_channels=1).get_all_samples()
        audio_16k = wav.data.squeeze(0)
        audio_24k = resample_frac(audio_16k.unsqueeze(0), 16000, QWEN_SR)
        with torch.inference_mode():
            codes = codec_enc.encode(audio_24k.half().cuda().unsqueeze(0))
            prompt_codes = codes[0, :Q].T.long()  # (T, Q)
        prompt_text = asr(audio_16k)
        del codec_enc
        torch.cuda.empty_cache()
        # Free ASR
        from vui import inference as _inf

        if _inf.wm:
            del _inf.wm
            _inf.wm = None
            torch.cuda.empty_cache()
        print(f"  Prompt: '{prompt_text[:60]}' ({prompt_codes.shape[0]} frames)")

    if Path(prompt_file).exists():
        load_prompt(prompt_file)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    with torch.inference_mode():
        prompt_segs = None
        if prompt_text and prompt_codes is not None:
            prompt_segs = [(prompt_text, prompt_codes)]

        t0 = time.perf_counter()
        audio_chunks = []
        for audio_chunk in render_audio_stream(
            state,
            simple_clean(text),
            prompt_segments=prompt_segs,
            temperature=settings["temperature"],
            max_secs=settings["max_secs"],
            eos_threshold=settings["eos_threshold"],
            sq_scores=sq,
            wps_score=0.0,
        ):
            audio_chunks.append(audio_chunk)
        dt = time.perf_counter() - t0

        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=-1)
            dur = full_audio.shape[-1] / QWEN_SR
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = str(out_dir / f"render_{ts}.wav")
            tmp_file = "/tmp/vui_render.wav"
            encoded = AudioEncoder(
                full_audio.squeeze().cpu().float().unsqueeze(0),
                sample_rate=int(QWEN_SR),
            )
            encoded.to_file(save_file)
            encoded.to_file(tmp_file)
            print(f"  {dur:.1f}s in {dt:.2f}s ({dur/dt:.1f}x RTF) -> {save_file}")
            subprocess.Popen(
                ["play", tmp_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            print("  Generation failed")

    state.teardown()
