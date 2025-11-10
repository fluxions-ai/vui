import torch

# Global pipeline variable
pipeline = None


@torch.inference_mode()
def detect_voice_activity(waveform, pipe=None):
    """
    Detect voice activity using Silero VAD.

    Args:
        waveform: Audio waveform tensor (16khz expected)
        pipe: Optional pre-loaded model (for consistency with old API)

    Returns:
        List of (start, end) tuples indicating voice segments in seconds
    """
    waveform = waveform.flatten().float()
    global pipeline

    # Load model and utils if not already loaded
    if pipe is not None:
        pipeline = pipe
    elif pipeline is None:
        # Load Silero VAD model and utilities
        # Returns: (model, utils) where utils = (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        pipeline = {'model': model, 'utils': utils}

    # Extract model and get_speech_timestamps function
    if isinstance(pipeline, dict):
        model = pipeline['model']
        get_speech_timestamps = pipeline['utils'][0]
    else:
        # Fallback for backward compatibility if pipe is just a model
        model = pipeline
        _, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        get_speech_timestamps = utils[0]

    # Silero VAD expects 16kHz audio
    sample_rate = 16000

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sample_rate,
        return_seconds=False  # Returns in samples
    )

    # Convert from samples to seconds
    segments = [
        (segment['start'] / sample_rate, segment['end'] / sample_rate)
        for segment in speech_timestamps
    ]

    return segments


def merge_segments(segments, min_gap=0.3):
    """
    Merge segments that are close together.

    Args:
        segments: List of (start, end) tuples
        min_gap: Minimum gap in seconds between segments to keep them separate

    Returns:
        List of merged (start, end) tuples
    """
    if not segments:
        return []

    # Sort segments by start time
    segments = sorted(segments, key=lambda x: x[0])

    merged = [segments[0]]

    for current_start, current_end in segments[1:]:
        last_start, last_end = merged[-1]

        # If current segment is close to the last one, merge them
        if current_start - last_end < min_gap:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged
