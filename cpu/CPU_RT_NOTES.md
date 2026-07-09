# vui CPU realtime — working notes (loop)

Goal: vui TTS at >=1.0x realtime on CPU (Ryzen 7 7840U, 4 threads), using gguf-style
quantization, with moonshine-verified faithful output.

RTF convention here = audio/compute (HIGHER = faster, like the engine prints). 1.0x = realtime.

## Baseline (preset voice abraham.wav, "The quick brown fox jumps over the lazy dog.")
- token-gen (backbone+RQ, fp32 cblas_sgemv): 1523ms, 1.42x rt on its own
- codec decode: 1533ms (0.69x rt on its own), of which:
  - waveform decoder 1320ms (86%)  <- transposed-conv vocoder, NOT a transformer
  - upsample 79ms, transformer 24ms, quantizer 2ms
- END-TO-END: 3056ms / 2200ms audio = 0.72x realtime
- moonshine: EXACT match. quality good.

## Key insight
Both halves run IN SERIES and each is ~1.4x rt alone; series sum = 0.72x rt.
- GGUF only accelerates the transformer (backbone+RQ). The 86% vocoder is conv -> gguf-immune.
- BUT: 2x on backbone (1523->~760ms) => 760+1533=2293ms => ~0.96x rt. Almost realtime on backbone alone.
- Closers to cross 1.0x: (a) any codec win, or (b) OVERLAP gen+codec (pipeline) => wall-clock ~max(760,1533)=1533ms => 1.43x rt.

## Plan
1. [done] baseline + profile + verify
2. [in progress] Q8_0 quantize backbone+RQ GEMV in vui_tts.c
   - matmul()/matmul_bias() at lines 172-180 are the only GEMV helpers (backbone+RQ).
   - Quantize big weight matrices to Q8_0 (32-val blocks, int8 + fp32 scale) at load.
   - Quantized GEMV with VNNI; expect ~1.5-2x on token-gen.
   - target: token-gen >=2.3x rt, end-to-end ~0.96x rt
   - EXACT call sites to route through matmul_q8 (all go through single matmul() at line 172):
     backbone: lines 413-415 (wqkv, offset-indexed), 455 (wo), 459-460 (w1/w3), 466 (w2)
     RQ:       lines 491-493, 517, 521-522, 528, 558
     wqkv is one [3*dim,dim] matrix; sub-matmuls offset by whole rows (dim multiple) -> Q8_0 row-blocked indexing is clean.
   - Q8_0 = {fp32 scale; int8 q[32]} per 32-col block. First cut: int8 weights, fp accumulate (4x weight-bandwidth cut).
     Must BEAT cblas_sgemv (AVX512) -> needs AVX512 int8 GEMV (VNNI _mm512_dpbusd) + OMP over rows, else it's slower. Verify perf AND moonshine before keeping.
   - gate behind -DUSE_Q8 so build stays green / easy A/B.
3. codec: int8 the waveform-decoder convs (im2col+sgemm -> int8 gemm) OR overlap gen/codec
   - target: end-to-end >=1.0x rt
   - NOTE: waveform decoder is bandwidth-bound -> int8 convs should give real win (next priority after backbone Q8).
4. verify EVERY change with moonshine (must stay faithful)

## Q8_0 GEMV kernel — VERIFIED (microbench /tmp/q8_bench.c)
AVX512-VNNI (_mm256_dpbusd_epi32, w+128 unsigned + -128*sum(x) correction), relerr ~0.005.
Speedup vs cblas_sgemv on backbone sizes:
  1 thread: wqkv 1.63x, w1 1.43x, w2 1.24x, wo 1.22x
  4 thread: wqkv 1.23x, w1 1.13x, w2 1.26x, wo 0.85x   <- gains compress at 4thr (bandwidth/latency bound)
=> realistic token-gen gain at 4thr ~1.2x. token-gen 1523ms -> ~1270ms.
=> end-to-end with Q8 backbone only: 1270+1533 = 2803ms / 2200ms = ~0.78x realtime. STILL NOT REALTIME.

## STRATEGIC PIVOT (data-driven)
gguf/Q8 only speeds the backbone (~1.2x), which is HALF the serial time. Caps end-to-end at ~0.78x.
The CODEC (1533ms, 50%) is the wall. Two higher-leverage levers:
  A) OVERLAP gen+codec (producer/consumer threads): wall-clock -> max(gen,codec)=1533ms => 1.43x realtime.
     Hits realtime with ZERO quant, ZERO quality change. Engine already has --stream chunked mode to build on.
  B) int8 the vocoder convs (bandwidth-bound) -> push codec below 1533ms.
NOTE: if we OVERLAP (A), gen is hidden behind codec, so Q8-backbone gives ~0 end-to-end gain (codec-bound).
  => Q8 backbone only matters for non-overlapped or low-latency streaming TTFB. Prioritize CODEC.

## Engine findings (vui_tts.c)
- Non-stream path (the 0.72x baseline): generate ALL frames, then codec_decode once. = sum of halves.
- --stream path (lines 1857-1889): chunked but STILL sequential on one thread (gen chunk, then decode),
  AND re-decodes ALL frames every chunk via codec_decode(all_codes, n_frames) => O(n^2) waste. Bug.
  Yet an incremental per-frame path EXISTS: codec_stream_frame() (line 1141) + codec_stream_init/state.
- THE UNLOCK: producer/consumer threads. Thread A: decode_step -> frame queue. Thread B: codec_stream_frame -> audio.
  Wall-clock -> max(gen, codec) ~= 1533ms => 1.43x realtime. No quant, no quality change.
  decode_step uses backbone/RQ state; codec_stream_frame uses codec_stream state -> likely separable buffers (CHECK for shared scratch in model struct before threading).
- Q8 kernel (/tmp/q8_bench.c) is complementary: matters for low-latency streaming TTFB, not for overlapped throughput (codec-bound).

## OUTPUT FIDELITY ISSUE (checking outputs are good)
Symptom: text drift — first words dropped ("Hey"/"Turn"), hallucinated filler prepended
  ("Another together...", "I don't want to be a puppet...") before the real sentence.
- prompt+speaker ARE applied correctly (cache prefills spk emb + prompt audio + text; C engine loads it).
- temp sweep 0.5/0.65/0.9: does NOT fix it. systemic, not a temperature problem.
- neutral declarative text ("fox") = exact; casual/conversational lines drift more (model trained w/ filler).
- PRIME SUSPECT: sq_proj cond_bias truncation. checkpoint sq_proj.proj.0.weight=[768,448] (7 SQ dims x64)
  but code builds [768,384] (6 dims x64) -> export_full.py:84 m.sq_proj([4,4,4,4.5,4,4,4.5]) silently truncated.
  This SQ "speech-quality" steering is what keeps output clean/faithful. FIX: reconcile config sq dims to
  checkpoint (7), re-export cond_bias, re-verify with moonshine. (model.py:1490-1502, config sinusoidal_cond)
- NOTE for realtime work: fidelity must be fixed BEFORE trusting any speed-vs-quality A/B.

## CHECKPOINT FIX (big) — 0jiksor5_0100000.pt was the WRONG (stale training) checkpoint
- Correct release = vui-nano.safetensors (HF fluxions/vui, 583MB). sq_proj=[768,384] -> matches code, NO truncation warning.
- The .pt had sq_proj=[768,448] (stale 7-dim SQ) -> silently truncated. export_full.py:84 passed 7 vals (would error on release).
- FIXED export_full.py: now bakes the OFFICIAL global cond_bias from prompts/*.safetensors (norm 0.329, identical across voices).
  Re-exported -> vui_nano_full.bin. (SQ_P90=(3.58,3.95,3.90,4.25,3.75,4.03) is the 6-dim fallback; note official cond_bias != P90.)
- Official preloaded voices: prompts/<name>.safetensors carry {audio, codes(T,16), cond_bias, spk_token_emb}.
  /tmp/prep_official.py builds KV cache from EXACT transcript (prompts/<name>.txt, has disfluencies) + pre-encoded codes + official spk token.
  -> prompt_maeve_official.bin. This is the correct "preloaded voice" (no whisper, no re-encode).

## DRIFT IS A C-ENGINE BUG (isolated this iteration)
Symptom persists IDENTICALLY across: stale .pt AND vui-nano; wrong AND official cond_bias; my-encode AND official preloaded prompt; temps 0.5-0.9.
  => NOT data/checkpoint/prompt. It is the C inference. neutral text mostly ok; first word often dropped/garbled
     ("The"/"Turn"->"Or"), casual lines drift, occasional 30s runaway to max_frames.
SUSPECT: first-frame seeding. main() seeds codes_in=calloc(Q)=all-zeros (line 1839), so the FIRST decode_step embeds a
  spurious [0,0,...] audio frame (decode_step line 1677-1680) before predicting frame-0, instead of predicting frame-0
  directly from the post-text hidden state. A spurious leading frame would shift/eat the first word. ALSO check cond_bias
  during decode (README: decode steps = NO cond_bias; prefill_token_emb adds it only in text prefill - that looks right).
NEXT: run PyTorch reference on CPU (slow ok) for 1 sentence -> compare frame-0 seeding + codes. Confirms engine-vs-spec.

## FIXED: first-frame seeding bug (drift root cause) -- outputs now GOOD
Reference (inference.py:1067-1070): frame-0 sampled DIRECTLY from post-text-prefill hidden
  (hidden=out[:,-1]; first_codes=rq_sample(codec_head(hidden),hidden)). NO audio input for frame-0.
C engine bug: seeded codes_in=calloc(Q)=zeros and called decode_step -> embedded a phantom [0,0,...]
  audio frame + extra backbone_forward before frame-0 -> ate/shifted first word, caused drift + runaways.
FIX (vui_tts.c): added predict_frame_from_hidden(); frame-0 now uses model.bb.x (post-text hidden) directly,
  subsequent frames use decode_step (embed prev codes + forward). Also ~1 fewer backbone forward / utterance.
VERIFIED (vui_nano_full.bin + official maeve prompt, temp 0.7, moonshine):
  4/4 exact on prior failing lines (incl casual "Hey..." and the 30s-runaway line).
  Robustness batch 6/6 faithful (2 shown as diff are ONLY ASR digit formatting: "2 cups"/"14%").
  No more runaways. Speed unchanged (~1.0-1.23x token-gen).
=> "outputs are good" requirement now MET. Realtime/gguf work can resume on a faithful baseline.

## REALTIME: overlap is the lever, but per-frame streaming is a DEAD END
- codec path is buffer-isolated from backbone (mallocs own buffers) => gen||codec overlap is THREAD-SAFE. Good.
- codec_stream_frame() (per-frame incremental) was DEAD CODE. Tested behind --codec-stream: 37147ms vs batched 1791ms
  = 20x SLOWER (re-runs waveform decoder per frame, kills batched-conv efficiency). Also wrong length. UNUSABLE for overlap.
- => Overlap MUST chunk-pipeline the BATCHED codec_decode: generate chunk of C frames, worker runs codec_decode on
  [a-L .. b] with lookback L, discard first L frames' samples (avoid causal-conv boundary clicks), overlap with next chunk gen.
  Overhead ~ (1 + numChunks*L/N)x codec work; for C=25,L=10,N=70 ~1.43x codec, overlapped => ~1.1-1.2x realtime. Worth it.
- EOS: NOT an issue. eos head well-calibrated (fires p~1.0 at true end); thr 0.35-0.9 ~same length. Default set to 0.35
  (matches reference engine.py n_threshold). Added --eos-threshold/--min-frames flags. C engine had been stricter (0.5).

## CONCLUSION: the realtime wall is VOCODER ACTIVATION BANDWIDTH (not weights/threads/gguf)
Measured per utterance in the waveform decoder: weights ~116MB vs activations ~2.6GB = 22x.
The 480x upsample creates 720k-timestep tensors (last block) passed 4x through residual units.
Implications (all evidence-backed this session):
- gguf / int8-WEIGHTS: cannot help codec (weights = 4% of its bandwidth). Backbone-only gguf caps end-to-end ~0.78-0.96x.
- overlap (gen||codec): BUILT + verified faithful (moonshine == serial). Capped ~0.88x by shared-DDR5 bandwidth contention;
  thread-partition sweep (OMP/OPENBLAS combos) all 0.80-0.88x. Generation slows 3.5s->4.6s when codec runs alongside.
- => to cross 1.0x realtime on CPU you MUST cut the 2.6GB activation traffic:
    (a) int8 ACTIVATIONS through the whole vocoder (4x; invasive, quality risk, hard to beat OpenBLAS sgemm), OR
    (b) lighter vocoder that avoids 480x transconv blowup: iSTFT/Vocos head or subband (RETRAIN codec) -- the neutts approach, OR
    (c) offload vocoder to the 780M iGPU (no retrain; user said CPU-only).
CURRENT BEST: faithful output @ ~0.85-0.88x realtime (C engine + overlap). That is the CPU ceiling for this
  vocoder architecture with weight-level optimization. Matches literature (Meta/LPCNet/subband all attack the vocoder arch).
DONE this session: outputs GOOD (checkpoint+cond_bias+preloaded prompt+first-frame fix), Q8 VNNI kernel (verified 1.2-1.6x),
  overlap (built+faithful), EOS matched to reference. Flags added: --eos-threshold --min-frames --overlap --codec-stream.

## *** REALTIME ACHIEVED (ONNX codec, NOT gguf) ***
The bottleneck was the codec, and onnxruntime's fused conv kernels solve it:
- ONNX fp32 codec = 3.6x realtime (2x faster than hand-rolled C codec). int8 quant NOT needed (dynamic=0.26x, static=0.78x SLOWER).
- C backbone (Q=12, first-frame-fixed) = 1.55x realtime, faithful.
- SEQUENTIAL pipeline (C backbone emits codes -> ONNX codec batch-decode) = **1.08-1.14x realtime, moonshine EXACT.**
- gguf was a red herring for realtime: backbone was never the wall. Q8/gguf on backbone is now optional HEADROOM (1.55->~1.9x).
Hookup: vui_tts.c `--emit-codes` streams frame codes to stdout; Python feeds ONNX codec (/tmp/codec_q12.onnx, 12-codebook).
Codec export: torch.onnx.export(QwenCodecDecoder.decode, codes(1,12,T), dynamic axis T). ORT intra_op_threads=4.
STREAMING (vui_stream.py): TTFA ~460ms but throughput 0.62x (gen||codec bandwidth contention) -> stutters on long utts.
  Streaming-realtime needs chunk-pipeline gen->codec->play WITHOUT concurrent gen/codec (alternate, overlap only w/ playback),
  or faster backbone (Q8). For batch/file: sequential 1.14x is the answer.

## *** WARM SERVER + DAEMON: validated ***
vui_tts.c `--server`: load model+cache+tokenizer ONCE, loop {READY -> read text line -> reset bb.pos=prompt_pos
  -> prefill gen text -> generate+emit codes (CHUNK flow-control acks on stdin) -> END}. KV reset = set bb.pos to the
  post-prompt-cache position; new prefill/gen overwrites the rest. VALIDATED (/tmp/warm_verify.py):
  load 1.7s once, then 3 warm utterances 1.12-1.28x rt, ALL moonshine-faithful (KV reset correct, not corrupted).
vui_daemon.py: resident process holds warm C --server + ONNX + simple_clean + audio device; unix socket /tmp/vui.sock.
  say.sh = thin client (auto-starts daemon). Each `say` after the first is WARM (no 3.4s reload, no 2.2s torch import)
  => start time drops to ~prebuffer. Daemon reaches "ready" and serves correctly.
HARNESS GOTCHA (cost hours): `pkill -f vui_tts` matches the pkill command's OWN shell cmdline -> kills itself ->
  silent abort. Use `pkill -x vui_tts` (exact name). Also harness shell is set -e: guard every pkill/grep -c with `|| true`.
  And background daemons don't survive across separate tool calls -> validate multi-utterance in ONE process.

## Ruled out
- Thread tuning: WASH. 4/4 near-optimal. 8-core part, SMT doesn't help compute-bound GEMM.
  OMP=8/OB=8 gave codec 1438ms (vs 1539) but gen worse (1717 vs 1536). OB=16 catastrophic (11.7s, bandwidth thrash).
  => waveform decoder is MEMORY-BANDWIDTH bound => int8 (half the bytes) is the right lever, not threads.

## A/B facts already established
- neutts: Q4 GGUF + onnx-int8 = 2.5x rt on this CPU, moonshine-faithful.
- OpenBLAS build of llama.cpp made NO difference vs default wheel (batch-1 decode uses
  native quantized kernels, not BLAS). So for vui, gguf-quant > BLAS.
