import argparse
import time

import torch
from torchcodec.encoders import AudioEncoder

torch.set_float32_matmul_precision("high")

from vui.inference import render
from vui.model import Vui

TEXT = "Hey, here is some random stuff, usually something quite long as the shorter the text the less likely the model can cope!"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--model", default=Vui.ABRAHAM)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    model = Vui.from_pretrained(args.model).cuda().eval()

    if args.benchmark:
        # Warmup
        render(model, "Hello, warmup run.", use_cuda_graph=not args.no_cuda_graph)
        torch.cuda.synchronize()

        sr = model.codec.config.sample_rate
        ttfbs, rtfs, gen_times = [], [], []

        for i in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            waveform = render(model, args.text, use_cuda_graph=not args.no_cuda_graph)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            audio_duration = waveform.shape[-1] / sr
            rtf = audio_duration / elapsed
            gen_times.append(elapsed)
            rtfs.append(rtf)
            print(f"  run {i+1}: {elapsed:.3f}s, {audio_duration:.2f}s audio, RTF={rtf:.1f}x")

        print(f"\n  avg gen:  {sum(gen_times)/len(gen_times):.3f}s")
        print(f"  avg RTF:  {sum(rtfs)/len(rtfs):.1f}x")
    else:
        waveform = render(model, args.text, use_cuda_graph=not args.no_cuda_graph)
        print(waveform.shape)
        encoder = AudioEncoder(waveform[0], sample_rate=22050)
        encoder.to_file("out.wav")


if __name__ == "__main__":
    main()
