# Vui - 100M Parameter On-Device Conversational Text-to-Speech

Vui is a lightweight, open-source text-to-speech model with only 100M parameters, designed for natural conversational speech synthesis. Built on a Llama-style transformer architecture, it generates expressive multi-speaker dialogue with breaths, laughter, hesitations, and other non-verbal sounds.

Trained on 40,000 hours of real audio conversations. Runs on consumer GPUs.

**[Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/fluxions/vui-space)**

## Features

- **100M parameters** - small enough for on-device and edge deployment
- **Conversational speech** - trained on real conversations, not studio recordings
- **Non-verbal sounds** - generates `[breath]`, `[laugh]`, `[sigh]`, `[hesitate]`, `[tut]` naturally
- **Multi-speaker** - COHOST model handles two-speaker dialogues
- **Voice cloning** - clone from audio samples with the base model
- **Streaming** - real-time streaming synthesis with CUDA graph acceleration
- **Custom audio codec** - Fluac, a modified DAC with FSQ that reduces token rate from 86Hz to 21.5Hz (4x reduction)

## Quick Start

```python
from vui.model import Vui
from vui.inference import render
from torchcodec.encoders import AudioEncoder

model = Vui.from_pretrained(Vui.ABRAHAM).cuda().eval()

audio = render(model, """
So [breath] the thing about this is, it's not what you'd expect, right?
Um, it's actually [hesitate] completely different.
""")

encoder = AudioEncoder(audio[0], sample_rate=22050)
encoder.to_file("output.wav")
```

## Comparison with Other Small TTS Models

| Model | Params | Conversational | Multi-Speaker | Voice Cloning | Breaths & Non-Verbal | Streaming |
|---|---|---|---|---|---|---|
| **Vui** | **100M** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| Kokoro | 82M | No | No | No | No | No |
| Pocket TTS | 100M | No | No | Yes | No | No |
| KittenTTS | 14-80M | No | No | No | No | No |
| Orpheus | 150M+ | Partial | No | No | Partial | No |

## Models

| Model | Description |
|---|---|
| `Vui.BASE` | Base checkpoint trained on 40k hours of audio conversations |
| `Vui.ABRAHAM` | Single-speaker model with context-aware replies |
| `Vui.COHOST` | Two-speaker model for multi-speaker dialogue |

## Architecture

Vui is a Llama-style causal transformer that predicts audio tokens from text:

- **Text encoder**: ByT5 byte-level tokenizer
- **Decoder**: 6-layer transformer, 512 dim, 8 heads, RMSNorm, SiLU, RoPE
- **Audio codec**: Fluac - a modified [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) using Finite Scalar Quantization (FSQ) with 9 codebooks at 1000 entries each
- **Token rate**: ~21.5 Hz (vs 86 Hz for standard DAC), enabling longer context windows
- **Inference**: KV caching + CUDA graphs for fast autoregressive generation

## Non-Verbal Sound Tags

Vui understands these inline tags for expressive speech:

```
[breath]    - breathing sounds
[laugh]     - laughter
[sigh]      - sighing
[hesitate]  - filled pauses / um / uh
[tut]       - tutting
```

Example:
```
And I'm Jamie um [laugh] today, we're diving into a [hesitate] topic
that's transforming customer service [breath] voice technology for agents.
```

## Installation

*Before running `demo.py`, you must accept model terms for [Voice Activity Detection](https://huggingface.co/pyannote/voice-activity-detection) and [Segmentation](https://huggingface.co/pyannote/segmentation) on Hugging Face.*

### Linux
```sh
uv pip install -e .
```

### Windows
```pwsh
uv venv
.venv\Scripts\activate
uv pip install -e .
uv pip install triton_windows
```

## Demo

```sh
python demo.py
```

Or [try it on Hugging Face Spaces](https://huggingface.co/spaces/fluxions/vui-space).

## Voice Cloning

You can clone voices with the base model. Pass an audio sample and the model will adapt to the speaker's characteristics. Quality varies as the model hasn't been extensively trained for this task.

## FAQ

1. Developed on two 4090s: https://x.com/harrycblum/status/1752698806184063153
2. The model does hallucinate occasionally - this is the best achievable with limited compute resources.
3. VAD slows things down but is needed to remove silence regions.

## Attributions

- [Whisper](https://github.com/openai/whisper) - OpenAI
- [Audiocraft](https://github.com/facebookresearch/audiocraft) - Meta
- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)

## Citation

```bibtex
@software{vui_2025,
  author = {Coultas Blum, Harry},
  month = {01},
  title = {{Vui: 100M Parameter Conversational Text-to-Speech}},
  url = {https://github.com/fluxions-ai/vui},
  version = {1.0.0},
  year = {2025}
}
```
