import base64
import threading
import time

import gradio as gr
import julius
import numpy as np
import soundfile as sf
import torch

torch.set_float32_matmul_precision("high")

from vui.inference import asr, precompute_text, render, stream_render
from vui.model import Vui


def get_available_models():
    models = {}
    for attr_name in dir(Vui):
        if attr_name.isupper():
            attr_value = getattr(Vui, attr_name)
            if isinstance(attr_value, str) and attr_value.endswith(".pt"):
                models[attr_name] = attr_value
    return models


AVAILABLE_MODELS = get_available_models()
print(f"Available models: {list(AVAILABLE_MODELS.keys())}")

current_model = None
current_model_name = None


def load_and_warm_model(model_name):
    global current_model, current_model_name

    if current_model_name == model_name and current_model is not None:
        print(f"Model {model_name} already loaded and warmed up!")
        return current_model

    print(f"Loading model {model_name}...")
    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    model_path = AVAILABLE_MODELS[model_name]
    model = Vui.from_pretrained_inf(model_path).cuda()

    print(f"Warming up model {model_name}...")
    warmup_text = "Hello, this is a warmup test, just saying some random stuff to make sure everything is working properly."
    try:
        render(model, warmup_text, max_secs=10)
    except Exception:
        pass
    for _ in stream_render(model, warmup_text, max_secs=10):
        pass

    current_model = model
    current_model_name = model_name
    print(f"Model {model_name} loaded and warmed up successfully!")
    return model


SAMPLE_TEXTS = [
    """Welcome to Fluxions, the podcast where... we uh explore how technology is shaping the world around us. I'm your host, Alex.
[breath] And I'm Jamie um [laugh] today, we're diving into a [hesitate] topic that's transforming customer service uh voice technology for agents.
That's right. We're [hesitate] talking about the AI-driven tools that are making those long, frustrating customer service calls a little more bearable, for both the customer and the agents.""",
    """Um, hey Sarah, so I just left the meeting with the, uh, rabbit focus group and they are absolutely loving the new heritage carrots! Like, I've never seen such enthusiastic thumping in my life! The purple ones are testing through the roof - apparently the flavor profile is just amazing - and they're willing to pay a premium for them! We need to, like, triple production on those immediately and maybe consider a subscription model? Anyway, gotta go, but let's touch base tomorrow about scaling this before the Easter rush hits!""",
    """What an absolute joke, like I'm really not enjoying this situation where I'm just forced to say things.""",
    """ So [breath] I don't know if you've been there [breath] but I'm really pissed off.
Oh no! Why, what happened?
Well I went to this cafe hearth, and they gave me the worst toastie I've ever had, it didn't come with salad it was just raw.
Well that's awful what kind of toastie was it?
It was supposed to be a chicken bacon lettuce tomatoe, but it was fucking shite, like really bad and I honestly would have preferred to eat my own shit.
[laugh] well, it must have been awful for you, I'm sorry to hear that, why don't we move on to brighter topics, like the good old weather?""",
    """Right so [breath] the thing about quantum computing is, it's not just faster classical computing, right? It's a completely different paradigm. Um, you're working with qubits that can be in superposition, and when you entangle them [hesitate] that's where the magic happens. But here's what nobody tells you, the error rates are still absolutely brutal.""",
    """Oh my god, you will not believe what just happened to me at the supermarket. So I'm standing in the queue, minding my own business, and this woman just [breath] cuts right in front of me with a trolley full of stuff! And I'm standing there with like, two items. Two! [laugh] So I said excuse me, and she just looked at me like I was the problem. The audacity, honestly.""",
    """Today we're going to be looking at how to make the perfect sourdough bread. Now [breath] the key thing that most people get wrong is the hydration level. You want to be somewhere around seventy five percent for a nice open crumb. Um, and your starter needs to be really active, I'm talking like, doubling in size within four to six hours. If it's not doing that, don't even bother, you'll just end up with a brick.""",
    """And the winner of this year's award goes to [hesitate] oh wow, I can barely read this, um [breath] it goes to the team from Edinburgh! [laugh] I have to say, this is absolutely deserved, they have worked so incredibly hard this year and the results speak for themselves. Congratulations to everyone involved, this is a truly special moment.""",
]

default_model = "ABRAHAM" if "ABRAHAM" in AVAILABLE_MODELS else list(AVAILABLE_MODELS.keys())[0]
model = load_and_warm_model(default_model)

log_lines = [f"Model {default_model} loaded and ready"]


def log(msg):
    log_lines.append(msg)
    return "\n".join(log_lines[-20:])


def get_log():
    return "\n".join(log_lines[-20:])


def text_to_speech(
    text, prompt_audio=None, temperature=0.5, top_k=100, top_p=None, max_duration=120
):
    if not text.strip():
        return None, log("No text provided")

    if current_model is None:
        return None, log("No model loaded")

    print(f"Generating speech for: {text[:50]}... using model {current_model_name}")

    # prompt_codes = None
    # prompt_text = ""
    # if prompt_audio is not None:
    #     sr, audio = prompt_audio
    #     audio = torch.from_numpy(audio).float()
    #     audio = audio / audio.abs().max()
    #     if len(audio.shape) > 1:
    #         audio = audio.mean(1)
    #     codec_sr = current_model.codec.config.sample_rate
    #     max_samples = int(30 * codec_sr)
    #     if len(audio) > max_samples:
    #         audio = audio[:max_samples]
    #     sf.write("prompt_audio.wav", audio.numpy(), sr)
    #     print(audio.shape)
    #     if sr != codec_sr:
    #         audio = julius.resample_frac(audio, sr, codec_sr)
    #     with torch.inference_mode():
    #         audio = audio[None, None]
    #         prompt_codes = current_model.codec.encode(audio.cuda())
    #     prompt_text = asr(julius.resample_frac(audio.flatten(), codec_sr, 16000))
    #     print("PROMPT_TEXT", prompt_text)
    #     print(f"Using audio prompt with shape: {prompt_codes.shape}")

    t1 = time.perf_counter()
    result = render(
        current_model,
        text,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_secs=max_duration,
    )

    waveform = result.cpu()
    sr = current_model.codec.config.sample_rate

    generation_time = time.perf_counter() - t1
    audio_duration = waveform.shape[-1] / sr
    speed_factor = audio_duration / generation_time

    if waveform.shape[-1] > 2000:
        waveform = waveform[..., :-2000]

    audio_array = waveform.flatten().numpy()

    info = f"Generated {audio_duration:.1f}s in {generation_time:.1f}s ({speed_factor:.1f}x RT) [{current_model_name}]"
    print(info)

    return (sr, audio_array), log(info)


def change_model(model_name):
    try:
        log(f"Loading {model_name}...")
        load_and_warm_model(model_name)
        return log(f"Loaded {model_name}")
    except Exception as e:
        return log(f"Error loading {model_name}: {e}")


PLAYER_JS = """
<script>
(function() {
    let ctx = null;
    let nextTime = 0;
    let sources = [];
    let lastGenId = null;
    let lastChunkId = -1;
    let pollLast = '';

    window.vuiPrepare = function() {
        sources.forEach(function(s) { try { s.stop(); } catch(e) {} });
        sources = [];
        lastGenId = null;
        lastChunkId = -1;
        pollLast = '';
        if (!ctx || ctx.state === 'closed') {
            ctx = new AudioContext();
        }
        if (ctx.state === 'suspended') {
            ctx.resume();
        }
        nextTime = ctx.currentTime;
        document.querySelectorAll('audio').forEach(function(a) {
            a.pause(); a.currentTime = 0;
        });
    };

    function playChunk(data) {
        if (!data || data === pollLast) return;
        pollLast = data;

        var i1 = data.indexOf(':');
        var i2 = data.indexOf(':', i1 + 1);
        var i3 = data.indexOf(':', i2 + 1);
        var genId = data.substring(0, i1);
        var chunkId = parseInt(data.substring(i1 + 1, i2));
        var sr = parseInt(data.substring(i2 + 1, i3));
        var b64 = data.substring(i3 + 1);

        if (genId !== lastGenId) {
            lastGenId = genId;
            lastChunkId = -1;
        }
        if (chunkId <= lastChunkId) return;
        lastChunkId = chunkId;

        if (!ctx) return;

        var raw = atob(b64);
        var bytes = new Uint8Array(raw.length);
        for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
        var samples = new Float32Array(bytes.buffer);

        var buffer = ctx.createBuffer(1, samples.length, sr);
        buffer.getChannelData(0).set(samples);

        var source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        sources.push(source);

        var when = Math.max(nextTime, ctx.currentTime);
        source.start(when);
        nextTime = when + buffer.duration;

        source.onended = function() {
            var idx = sources.indexOf(source);
            if (idx >= 0) sources.splice(idx, 1);
        };
    }

    setInterval(function() {
        var el = document.querySelector('#vui-chunk textarea');
        if (el && el.value) playChunk(el.value);
    }, 30);

    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey) && e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            var btn = document.querySelector('#generate-btn');
            if (btn && !btn.disabled) btn.click();
        }
    });

})();
</script>
"""

with gr.Blocks(title="Vui", theme=gr.themes.Soft(), head=PLAYER_JS) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            audio_chunk = gr.Textbox(visible=False, elem_id="vui-chunk")
            audio_output = gr.Audio(label=None, type="numpy", autoplay=False)
            log_output = gr.Textbox(label=None, lines=4, interactive=False, value=get_log())
            # audio_input = gr.Audio(
            #     label="Voice prompt (optional, up to 30s)",
            #     type="numpy",
            #     format="wav",
            #     waveform_options={"sample_rate": 22050},
            # )

        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=default_model,
                label=None,
            )
            text_input = gr.Textbox(
                label=None,
                placeholder="Enter text to convert to speech...",
                lines=5,
                max_lines=10,
            )
            with gr.Row():
                for i, sample in enumerate(SAMPLE_TEXTS):
                    btn = gr.Button(f"Sample {i + 1}", size="sm")
                    btn.click(fn=lambda idx=i: SAMPLE_TEXTS[idx], outputs=text_input)
            with gr.Accordion("Settings", open=False):
                temperature = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Temperature")
                top_k = gr.Slider(1, 200, value=100, step=1, label="Top-K")
                use_top_p = gr.Checkbox(label="Use Top-P", value=False)
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P", visible=False)
                max_duration = gr.Slider(5, 120, value=120, step=5, label="Max Duration (s)")
                use_top_p.change(fn=lambda x: gr.update(visible=x), inputs=use_top_p, outputs=top_p)
            with gr.Row():
                generate_btn = gr.Button(
                    "Generate", variant="primary", size="lg", elem_id="generate-btn"
                )
                full_btn = gr.Button("Full", variant="secondary", size="sm")

    model_dropdown.change(fn=change_model, inputs=model_dropdown, outputs=log_output)

    _precompute_timer = [None]

    def on_text_change(text):
        if _precompute_timer[0]:
            _precompute_timer[0].cancel()
        if not text or not text.strip() or current_model is None:
            return
        def worker():
            precompute_text(current_model, text)
        _precompute_timer[0] = threading.Timer(0.3, worker)
        _precompute_timer[0].daemon = True
        _precompute_timer[0].start()

    text_input.change(fn=on_text_change, inputs=[text_input])

    def generate_wrapper(text, temp, k, use_p, p, duration):
        if not text.strip() or current_model is None:
            return
        top_p_val = p if use_p else None
        t1 = time.perf_counter()
        gen_id = str(int(t1 * 1000))
        all_audio = []
        sr = None
        yield gr.skip(), gr.skip(), None
        for i, (chunk_sr, audio) in enumerate(
            stream_render(
                current_model,
                text,
                temperature=temp,
                top_k=int(k),
                top_p=top_p_val,
                max_secs=int(duration),
            )
        ):
            sr = chunk_sr
            all_audio.append(audio)
            b64 = base64.b64encode(audio.astype(np.float32).tobytes()).decode()
            total_secs = sum(len(a) for a in all_audio) / sr
            elapsed = time.perf_counter() - t1
            progress = f"Streaming [{current_model_name}] {total_secs:.1f}s generated ({elapsed:.1f}s elapsed)"
            yield f"{gen_id}:{i}:{sr}:{b64}", progress, gr.skip()

        elapsed = time.perf_counter() - t1
        if all_audio and sr:
            full_audio = np.concatenate(all_audio)
            total_secs = len(full_audio) / sr
            yield "", log(f"Done: {total_secs:.1f}s in {elapsed:.1f}s ({total_secs/elapsed:.1f}x RT) [{current_model_name}]"), (sr, full_audio)
        else:
            yield "", log("No audio generated"), gr.skip()

    gen_event = generate_btn.click(
        fn=generate_wrapper,
        inputs=[text_input, temperature, top_k, use_top_p, top_p, max_duration],
        outputs=[audio_chunk, log_output, audio_output],
        js="(...a) => { window.vuiPrepare && window.vuiPrepare(); return a; }",
        cancels=[],
    )
    submit_event = text_input.submit(
        fn=generate_wrapper,
        inputs=[text_input, temperature, top_k, use_top_p, top_p, max_duration],
        outputs=[audio_chunk, log_output, audio_output],
        js="(...a) => { window.vuiPrepare && window.vuiPrepare(); return a; }",
        cancels=[],
    )
    gen_event.cancels = [gen_event, submit_event]
    submit_event.cancels = [gen_event, submit_event]

    def full_wrapper(text, temp, k, use_p, p, duration):
        top_p_val = p if use_p else None
        return text_to_speech(text, None, temp, int(k), top_p_val, int(duration))

    full_btn.click(
        fn=full_wrapper,
        inputs=[text_input, temperature, top_k, use_top_p, top_p, max_duration],
        outputs=[audio_output, log_output],
        cancels=[gen_event, submit_event],
    )

    demo.load(fn=lambda: SAMPLE_TEXTS[1], outputs=text_input)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
