import torchaudio

from vui.inference import render
from vui.model import Vui

model = Vui.from_pretrained().cuda()
waveform = render(
    model,
    "Hey, here is some random stuff, usually something quite long as the shorter the text the less likely the model can cope!",
)
print(waveform.shape)
torchaudio.save("out.wav", waveform[0], 22050)
