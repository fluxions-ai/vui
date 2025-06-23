# vui

Small Conversational speech models that can run on device

# Installation

```sh
uv pip install -e .
```
For help with installation on Windows, please check this [guide](https://github.com/fluxions-ai/vui/issues/4#issuecomment-2971853482).
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

fluac is a audio tokenizer based on descript-audio-codec which reduces the number of codes per second by 4 from 83.1hz to 21.53hz

# FAQ

1) Was developed with on two 4090's https://x.com/harrycblum/status/1752698806184063153
2) Hallucinations: yes the model does hallucinate, but this is the best I could do with limited resources! :(
3) VAD does slow things down but needed to help remove areas of silence.

# Attributions

- Whisper - https://github.com/openai/whisper
- Audiocraft - https://github.com/facebookresearch/audiocraft
- Descript Audio Codec - https://github.com/descriptinc/descript-audio-codec

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
