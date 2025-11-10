# vui

Small 100M Conversational speech models that can run on device

# Installation

## Linux
```sh
uv pip install -e .
```
## Windows
Create and activate a virtual environment
```pwsh
uv venv
.venv\Scripts\activate
```
Install dependencies 
```pwsh
uv pip install -e .
uv pip install triton_windows
```

# Demo

[Try on Gradio](https://huggingface.co/spaces/fluxions/vui-space)

```sh
python demo.py
````

# Models

- Vui.BASE is base checkpoint trained on 40k hours of audio conversations
- Vui.ABRAHAM is a single speaker model that can reply with context awareness.
- Vui.COHOST is checkpoint with two speakers that can talk to each other.

# Voice Cloning

You can clone with the base model quite well but it's not perfect as hasn't seen that much audio / wasn't trained for long

# Research

vui is a llama based transformer that predicts audio tokens.

fluac is a audio tokenizer based on descript-audio-codec which reduces the number of codes per second by 4 from 86hz to 21.53hz

# FAQ

1) Was developed with on two 4090's https://x.com/harrycblum/status/1752698806184063153
2) Hallucinations: yes the model does hallucinate, but this is the best I could do with limited resources! :(
3) VAD does slow things down but needed to help remove areas of silence. We use Silero VAD for voice activity detection - it's lightweight and doesn't require authentication.

# Attributions

- Whisper - https://github.com/openai/whisper
- Audiocraft - https://github.com/facebookresearch/audiocraft
- Descript Audio Codec - https://github.com/descriptinc/descript-audio-codec
- Silero VAD - https://github.com/snakers4/silero-vad

# Citation

```
@software{vui_2025,
  author = {Coultas Blum, Harry},
  month = {01},
  title = {{vui}},
  url = {https://github.com/fluxions-ai/vui},
  version = {1.0.0},
  year = {2025}
}
```
