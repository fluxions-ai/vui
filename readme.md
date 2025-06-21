# vui

Small Conversational speech models that can run on device

# Run Locally

Create environment:
```sh
uv venv --python=3.12
```
Activate environment:
```sh
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```
Install core dependencies:
```sh
uv pip install -e . --link-mode=copy
```
(Windows only) Install Triton:
```sh
uv pip install triton_windows --link-mode=copy
```
(Optional) Install latest PyTorch (CUDA 12.6):
```sh
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --link-mode=copy
```
Authenticate with Hugging Face:
```sh
huggingface-cli login
```
Finally, run the demo:
```sh
python demo.py
```
*Before running `demo.py`, you must accept model terms for [Voice Activity Detection](https://huggingface.co/pyannote/voice-activity-detection) and [Segmentation](https://huggingface.co/pyannote/segmentation) on Hugging Face.*

# Live Demo
[Try on Gradio](https://huggingface.co/spaces/fluxions/vui-space)



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

