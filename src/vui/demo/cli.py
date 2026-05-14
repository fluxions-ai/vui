"""Interactive CLI for TTS rendering with streaming playback."""

import datetime
import readline
import subprocess
import time
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from vui.inference import InferenceState, asr, render_audio_stream, simple_clean
from vui.qwen_codec import SAMPLE_RATE as QWEN_SR
from vui.qwen_codec import QwenCodecDecoder, QwenCodecEncoder

SAMPLE_TEXTS = {
    "Podcast intro (2-speaker)": """Welcome to the podcast where we explore how technology is shaping the world around us. I'm your host, Alex.
And I'm Jamie [laugh] today, we're diving into a topic that's transforming customer service, voice technology for agents.
That's right. We're talking about the AI-driven tools that are making those long, frustrating customer service calls a little more bearable, for both the customer and the agents.""",
    "Excited voicemail": """Um, hey Sarah, so I just left the meeting with the rabbit focus group and they are absolutely loving the new heritage carrots! Like, I've never seen such enthusiastic thumping in my life! The purple ones are testing through the roof, apparently the flavor profile is just amazing, and they're willing to pay a premium for them! We need to triple production on those immediately and maybe consider a subscription model. Anyway, gotta go, but let's touch base tomorrow about scaling this before the Easter rush hits!""",
    "Short frustrated": """What an absolute joke, like I'm really not enjoying this situation where I'm just forced to say things.""",
    "Neural networks lecture": """Right, so today we're going to be looking at how neural networks actually learn. And I know that sounds complicated, but bear with me, because it's actually quite intuitive once you get the core idea. Think of it like a child learning to recognise a cat. At first, they might call every four-legged animal a cat. But over time, with enough examples, they get better and better at telling the difference.""",
    "Risotto monologue": """The thing about cooking a perfect risotto is patience. You cannot rush it. You add the stock one ladle at a time, and you stir, and you wait. And I know that sounds tedious, but that slow process is what creates that beautiful creamy texture. No cream needed, just time and attention.""",
    "Breaking news": """Breaking news this evening. A major storm system is moving across the south of England, bringing winds of up to seventy miles per hour and heavy rainfall. The Met Office has issued an amber weather warning for the entire region. Residents are advised to avoid unnecessary travel and secure any loose outdoor furniture. We'll have more updates throughout the evening as the situation develops.""",
    "Assistant: flight delay": """Your flight to Edinburgh has been delayed by forty-five minutes. The new departure time is fourteen twenty from gate B sixteen. I've already updated your calendar and sent a message to your contact letting them know you'll be arriving later than planned.""",
    "Assistant: build failure": """The build failed on the main branch. There are two test failures in the authentication module, both related to the token refresh logic that was changed in the last commit. I can show you the stack traces if you'd like, or I can attempt to identify the root cause.""",
    "Hesitant discussion": """So I th- I think uh... we do... um, we need to think about what we're doing here? Or do you not think that's a good idea?
I um, I've really thought about uh... what we need to do. And I think the answer is... well it's not simple, but basically we just need to... to start again? From scratch.""",
}


def run(
    checkpoint_path: str,
    prompt_file: str = "prompts/good_prompt3.wav",
    text: str | None = None,
    **overrides,
):
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

    # --- Tab completion ---
    examples = list(SAMPLE_TEXTS.keys())
    commands = ["/set", "/prompt", "/examples", "/settings", "/help", "/quit"]

    def completer(text, state_idx):
        line = readline.get_line_buffer().strip()
        if line.startswith("/set "):
            opts = [f"{k}=" for k in settings]
            matches = [o for o in opts if o.startswith(text)]
        elif line.startswith("/"):
            matches = [c for c in commands if c.startswith(text)]
        else:
            matches = [
                name for name in examples if name.lower().startswith(text.lower())
            ]
        return matches[state_idx] if state_idx < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims("")
    readline.parse_and_bind("tab: complete")

    def print_settings():
        print("  " + "  ".join(f"{k}={v}" for k, v in settings.items()))

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_idx = 0

    @torch.inference_mode()
    def render(render_text: str) -> str | None:
        nonlocal out_idx
        # Resolve example name
        if render_text in SAMPLE_TEXTS:
            render_text = SAMPLE_TEXTS[render_text]

        prompt_segs = None
        if prompt_text and prompt_codes is not None:
            prompt_segs = [(prompt_text, prompt_codes)]

        t0 = time.perf_counter()
        audio_chunks = []
        for audio_chunk in render_audio_stream(
            state,
            simple_clean(render_text),
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
            save_file = str(out_dir / f"render_{ts}_{out_idx}.wav")
            tmp_file = f"/tmp/vui_render_{out_idx}.wav"
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
            out_idx += 1
            return save_file
        else:
            print("  Generation failed")
            return None

    # Non-interactive: render text and exit
    if text is not None:
        render(text)
        state.teardown()
        return

    # Interactive mode
    print("\n=== CLI Render Mode (streaming) ===")
    print("  Type text or tab-complete an example name.")
    print("  /examples  list examples    /settings  show settings")
    print("  /set k=v   change setting   /prompt f  load prompt wav")
    print("  /help      show this        /quit      exit")
    print_settings()

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line == "/quit":
            break
        if line == "/help":
            print("  Type text or tab-complete an example name.")
            print("  /examples  list examples    /settings  show settings")
            print("  /set k=v   change setting   /prompt f  load prompt wav")
            continue
        if line == "/examples":
            for name in examples:
                preview = SAMPLE_TEXTS[name][:70].replace("\n", " ")
                print(f"  {name}: {preview}...")
            continue
        if line == "/settings":
            print_settings()
            continue
        if line.startswith("/set "):
            for pair in line[5:].split():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k in settings:
                        try:
                            settings[k] = type(settings[k])(v)
                            print(f"  {k}={settings[k]}")
                        except ValueError:
                            print(f"  Bad value: {v}")
                    else:
                        print(f"  Unknown: {k}")
            continue
        if line.startswith("/prompt "):
            pf = line[8:].strip()
            if Path(pf).exists():
                load_prompt(pf)
            else:
                print(f"  Not found: {pf}")
            continue
        if line.startswith("/"):
            print(f"  Unknown command: {line}")
            continue

        render(line)

    state.teardown()
