"""mlx-whisper streaming ASR backend (Apple Silicon).

Same simulated-streaming approach as fwhisper: buffer audio, re-transcribe
the last N seconds every ~500ms for interim updates with word timestamps,
then a full-accuracy pass on stop.

MLX is NOT thread-safe, so all transcription runs on a single dedicated
worker thread via a task queue. feed() posts work; the worker serialises it.

Models: `mlx-community/whisper-small-mlx` (best speed, ~600ms for 7s),
`mlx-community/whisper-large-v3-turbo` (best accuracy, ~2.4s for 7s).
"""

from __future__ import annotations

import queue
import threading
import time
from multiprocessing import Queue

import numpy as np

from vui.serving.stream.asr.base import ASRBackend, ASRSession

MODELS = {
    "small": "mlx-community/whisper-small-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class MLXWhisperSession(ASRSession):
    def __init__(
        self,
        model_path: str,
        result_queue: Queue,
        interim_every_s: float = 0.5,
        interim_window_s: float = 4.0,
        stable_iters: int = 2,
    ):
        super().__init__(result_queue)
        self._model_path = model_path
        self._buffer = np.zeros(0, dtype=np.float32)
        self._interim_every_n = int(interim_every_s * 16000)
        self._interim_window_n = int(interim_window_s * 16000)
        self._last_interim_i = 0
        self._last_text = ""
        self._lock = threading.Lock()
        self._feed_count = 0
        self._started_at = 0.0
        self._stable_iters = stable_iters
        self._last_words: list[str] = []
        self._last_word_ends: list[float] = []
        self._stable_counts: list[int] = []
        self._committed_words: list[str] = []
        self._committed_end_s: float = 0.0
        # Single worker thread for MLX calls (not thread-safe)
        self._work_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                task = self._work_queue.get(timeout=0.05)
            except Exception:
                continue
            if task is None:
                break
            try:
                task()
            except Exception as e:
                print(f"[ASR] mlx-whisper worker error: {e}")

    def start(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_interim_i = 0
        self._last_text = ""
        self._feed_count = 0
        self._started_at = time.perf_counter()
        self._last_words = []
        self._last_word_ends = []
        self._stable_counts = []
        self._committed_words = []
        self._committed_end_s = 0.0
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print("[ASR] Stream started (mlx-whisper)")

    # See fwhisper._SILENCE_RMS — mlx-whisper has no vad_filter at all, so
    # an energy gate is the only thing standing between mic noise and a
    # Whisper hallucination cascade ("Thank you for watching" / "ciao").
    _SILENCE_RMS = 0.005

    def _transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        return (result.get("text") or "").strip()

    def _transcribe_words(self, audio: np.ndarray) -> list[tuple[str, float]]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                word = w["word"].strip()
                if word:
                    words.append((word, w["end"]))
        return words

    def _run_interim(self):
        with self._lock:
            committed_n = int(self._committed_end_s * 16000)
            start = max(committed_n, len(self._buffer) - self._interim_window_n)
            buf = self._buffer[start:].copy()
        if len(buf) < 16000 * 0.2:
            return
        tail_n = min(len(buf), int(0.5 * 16000))
        tail = buf[-tail_n:]
        rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
        if rms < self._SILENCE_RMS:
            return
        buf_offset_s = start / 16000.0
        word_times = self._transcribe_words(buf)
        if not word_times:
            return

        words = [w for w, _ in word_times]
        full_words = list(self._committed_words) + words
        text = " ".join(full_words)

        if text != self._last_text:
            self._last_text = text
            self.result_queue.put({"type": "partial", "text": text})

        # Stability tracking
        new_counts: list[int] = []
        for i, w in enumerate(words):
            if i < len(self._last_words) and w == self._last_words[i]:
                prev = self._stable_counts[i] if i < len(self._stable_counts) else 0
                new_counts.append(prev + 1)
            else:
                break
        new_counts.extend([0] * (len(words) - len(new_counts)))
        self._stable_counts = new_counts
        self._last_words = words
        self._last_word_ends = [buf_offset_s + t for _, t in word_times]

        stable_n = 0
        for i, c in enumerate(new_counts):
            if c >= self._stable_iters:
                stable_n = i + 1
            else:
                break

        # Commit policy (same as fwhisper)
        buf_len_s = len(self._buffer) / 16000
        interim_window_s = self._interim_window_n / 16000
        commit_up_to = 0
        if buf_len_s > interim_window_s:
            window_start_s = buf_len_s - interim_window_s
            force_commit_threshold_s = window_start_s + 1.0
            for i, end_s in enumerate(self._last_word_ends):
                if end_s < force_commit_threshold_s:
                    commit_up_to = i + 1
                else:
                    break

        TAIL_WORDS = 12
        commit_up_to = max(commit_up_to, stable_n - TAIL_WORDS)
        commit_up_to = max(0, commit_up_to)

        for i in range(commit_up_to, stable_n):
            w = words[i].rstrip("\",')")
            if w.endswith((".", "!", "?")):
                commit_up_to = i + 1

        if commit_up_to > 0:
            new_words = words[:commit_up_to]
            end_s = self._last_word_ends[commit_up_to - 1]
            chunk = " ".join(new_words)
            n_total = len(full_words)

            self._committed_words.extend(new_words)
            full_text = " ".join(self._committed_words)
            start_s = self._committed_end_s
            self._committed_end_s = end_s

            self.result_queue.put(
                {
                    "type": "stable_prefix",
                    "text": full_text,
                    "stable_words": len(self._committed_words),
                    "total_words": n_total,
                    "stable_end_time": end_s,
                }
            )
            self.result_queue.put(
                {
                    "type": "line_completed",
                    "text": full_text,
                    "sentence": chunk,
                    "start_time": start_s,
                    "duration": end_s - start_s,
                }
            )

            words = words[commit_up_to:]
            self._last_words = words
            self._last_word_ends = self._last_word_ends[commit_up_to:]
            self._stable_counts = new_counts[commit_up_to:]

    def feed(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        if sample_rate != 16000:
            import torch
            from julius.resample import resample_frac

            t = torch.from_numpy(audio).float()
            t = resample_frac(t, sample_rate, 16000)
            audio = t.numpy().astype(np.float32)
        else:
            audio = audio.astype(np.float32, copy=False)

        with self._lock:
            self._buffer = np.concatenate([self._buffer, audio])
            need_interim = (
                len(self._buffer) - self._last_interim_i
            ) >= self._interim_every_n
            if need_interim:
                self._last_interim_i = len(self._buffer)
        self._feed_count += 1
        if self._feed_count == 1:
            print(
                f"[ASR] First feed: shape={audio.shape}, dtype={audio.dtype}, "
                f"range=[{audio.min():.4f}, {audio.max():.4f}]"
            )
        if need_interim:
            # Drop stale interims — if the worker hasn't caught up, replace
            # the pending interim with a fresh one rather than queuing N
            while not self._work_queue.empty():
                try:
                    self._work_queue.get_nowait()
                except queue.Empty:
                    break
            self._work_queue.put(self._run_interim)

    def stop(self) -> None:
        # Run final transcription on the worker thread, wait for result
        result_holder: list[str] = []
        done_event = threading.Event()

        def _final():
            with self._lock:
                buf = self._buffer.copy()
            dur = len(buf) / 16000
            t0 = time.perf_counter()
            try:
                text = self._transcribe(buf) if len(buf) > 0 else ""
            except Exception as e:
                print(f"[ASR] mlx-whisper final error: {e}")
                text = self._last_text
            ms = (time.perf_counter() - t0) * 1000
            print(
                f"[ASR] Final (mlx-whisper): '{text}' (buf={dur:.2f}s, took={ms:.0f}ms, "
                f"feeds={self._feed_count})"
            )
            if text:
                self.result_queue.put(
                    {
                        "type": "line_completed",
                        "text": text,
                        "start_time": 0.0,
                        "duration": dur,
                    }
                )
            self.result_queue.put({"type": "final", "text": text})
            result_holder.append(text)
            done_event.set()

        self._work_queue.put(_final)
        self._work_queue.put(None)  # sentinel to stop worker
        done_event.wait(timeout=30)
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=2.0)


class MLXWhisperBackend(ASRBackend):
    def __init__(
        self,
        model: str = "small",
        interim_every_s: float = 0.5,
        interim_window_s: float = 4.0,
        **_,
    ):
        import mlx_whisper

        self._model_path = MODELS.get(model, model)
        self._interim_every_s = interim_every_s
        self._interim_window_s = interim_window_s
        self.name = f"mlx-whisper:{model}"
        t0 = time.perf_counter()
        # Warm: load model weights into memory
        mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32),
            path_or_hf_repo=self._model_path,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        print(f"[ASR] Loaded in {(time.perf_counter()-t0)*1000:.0f}ms ({self.name})")

    def make_session(self, result_queue: Queue) -> ASRSession:
        return MLXWhisperSession(
            self._model_path,
            result_queue,
            interim_every_s=self._interim_every_s,
            interim_window_s=self._interim_window_s,
        )

    def transcribe_once(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import mlx_whisper

        if sample_rate != 16000:
            import torch
            from julius.resample import resample_frac

            t = torch.from_numpy(audio).float()
            t = resample_frac(t, sample_rate, 16000)
            audio = t.numpy().astype(np.float32)

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        return (result.get("text") or "").strip()
