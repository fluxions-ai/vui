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
    if not torch.cuda.is_available():
        return run_mlx(checkpoint_path, prompt_file, text, **overrides)
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


def _mlx_prompt(engine, row, prompt_file: str) -> bool:
    """Prefill `row` with a voice prompt. Official voices (maeve/abraham/...)
    use the pre-baked HF safetensors; other wavs are encoded locally with the
    torch codec encoder + a sibling .txt transcript (mlx_whisper if missing)."""
    from vui.engine import Segment
    from vui.mlx.engine import load_official_prompt

    voice = Path(prompt_file).stem
    try:
        text, codes, spk_token, cond_bias = load_official_prompt(voice)
        if cond_bias is not None:
            engine.cond_bias = cond_bias
        row.prefill([Segment(text=text, codes=codes)], spk_emb=spk_token)
        print(f"  Prompt '{voice}': '{text[:60]}' ({codes.shape[0]} frames)")
        return True
    except Exception:
        pass

    if not Path(prompt_file).exists():
        print(f"  No prompt at {prompt_file}; rendering unprompted")
        return False

    from julius.resample import resample_frac

    from vui.qwen_codec import QwenCodecEncoder
    from vui.qwen_spk_enc import QwenSpeakerEncoder

    wav = AudioDecoder(prompt_file, sample_rate=16000, num_channels=1)
    audio_16k = wav.get_all_samples().data.squeeze(0)
    audio_24k = resample_frac(audio_16k.unsqueeze(0), 16000, QWEN_SR)

    txt_path = Path(prompt_file).with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text().strip()
    else:
        import mlx_whisper

        text = mlx_whisper.transcribe(
            audio_16k.numpy(),
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="en",
            verbose=False,
        )["text"].strip()

    enc = QwenCodecEncoder.from_pretrained().cpu().float().eval()
    with torch.inference_mode():
        codes = enc.encode(audio_24k.float().unsqueeze(0))
    codes_tq = codes[0, : engine.Q].T.long().cpu()
    spk_emb = QwenSpeakerEncoder.from_pretrained().embed(
        audio_24k.squeeze(0), sr=int(QWEN_SR)
    )
    row.prefill([Segment(text=text, codes=codes_tq)], spk_emb=spk_emb)
    print(f"  Prompt (encoded): '{text[:60]}' ({codes_tq.shape[0]} frames)")
    return True


def run_mlx(
    checkpoint_path: str,
    prompt_file: str = "prompts/harry.wav",
    text: str | None = None,
    **overrides,
):
    """MLX one-shot render (Apple Silicon / no CUDA) via the Engine API."""
    from vui.engine import Engine, GenConfig

    engine = Engine(checkpoint_path)
    cfg = GenConfig(
        temperature=overrides.get("temperature", 0.9),
        max_secs=overrides.get("max_secs", 30.0),
        eos_threshold=overrides.get("eos_threshold", 0.45),
        n_codebooks=overrides.get("n_codebooks", 0),
    )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    with engine.new_row() as row:
        _mlx_prompt(engine, row, prompt_file)
        t0 = time.perf_counter()
        _codes, full_audio = row.render(simple_clean(text), cfg)
        dt = time.perf_counter() - t0

    if full_audio.shape[-1] == 0:
        print("  Generation failed")
        return

    dur = full_audio.shape[-1] / QWEN_SR
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = str(out_dir / f"render_{ts}.wav")
    AudioEncoder(full_audio.squeeze(0), sample_rate=int(QWEN_SR)).to_file(save_file)
    print(f"  {dur:.1f}s in {dt:.2f}s ({dur/dt:.1f}x RTF) -> {save_file}")
    subprocess.Popen(
        ["play", save_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
