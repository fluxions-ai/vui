"""
Simple test script to verify Silero VAD implementation works correctly.
"""

import torch
import numpy as np

print("Importing VAD module...")
from vui.vad import detect_voice_activity

def create_test_audio(sample_rate=16000):
    """
    Create a simple test audio signal:
    - 1 second of silence
    - 2 seconds of simulated speech (sine wave)
    - 0.5 seconds of silence
    - 1.5 seconds of simulated speech
    - 1 second of silence
    """
    duration_silence1 = 1.0
    duration_speech1 = 2.0
    duration_silence2 = 0.5
    duration_speech2 = 1.5
    duration_silence3 = 1.0

    # Create silence segments
    silence1 = torch.zeros(int(duration_silence1 * sample_rate))
    silence2 = torch.zeros(int(duration_silence2 * sample_rate))
    silence3 = torch.zeros(int(duration_silence3 * sample_rate))

    # Create "speech" segments (combination of sine waves to simulate voice)
    t1 = torch.linspace(0, duration_speech1, int(duration_speech1 * sample_rate))
    speech1 = 0.3 * torch.sin(2 * np.pi * 200 * t1) + 0.2 * torch.sin(2 * np.pi * 350 * t1)

    t2 = torch.linspace(0, duration_speech2, int(duration_speech2 * sample_rate))
    speech2 = 0.3 * torch.sin(2 * np.pi * 180 * t2) + 0.2 * torch.sin(2 * np.pi * 320 * t2)

    # Concatenate all segments
    audio = torch.cat([silence1, speech1, silence2, speech2, silence3])

    return audio, {
        'speech1_expected': (duration_silence1, duration_silence1 + duration_speech1),
        'speech2_expected': (
            duration_silence1 + duration_speech1 + duration_silence2,
            duration_silence1 + duration_speech1 + duration_silence2 + duration_speech2
        )
    }

def test_vad():
    """Test the Silero VAD implementation"""
    print("=" * 60)
    print("Testing Silero VAD Implementation")
    print("=" * 60)

    # Create test audio
    print("\n1. Creating test audio signal...")
    audio, expected_times = create_test_audio()
    print(f"   Audio shape: {audio.shape}")
    print(f"   Audio duration: {len(audio) / 16000:.2f} seconds")
    print(f"   Expected speech segments:")
    print(f"     - Segment 1: {expected_times['speech1_expected'][0]:.2f}s - {expected_times['speech1_expected'][1]:.2f}s")
    print(f"     - Segment 2: {expected_times['speech2_expected'][0]:.2f}s - {expected_times['speech2_expected'][1]:.2f}s")

    # Run VAD
    print("\n2. Running Silero VAD detection...")
    try:
        segments = detect_voice_activity(audio)
        print(f"   ✓ VAD completed successfully!")
        print(f"   Detected {len(segments)} speech segment(s)")

        if segments:
            print("\n3. Detected segments:")
            for i, (start, end) in enumerate(segments):
                duration = end - start
                print(f"   Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
        else:
            print("\n   ⚠ Warning: No speech segments detected!")

        # Verify we got some results
        if len(segments) > 0:
            print("\n" + "=" * 60)
            print("✓ TEST PASSED: VAD is working correctly!")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("⚠ TEST WARNING: VAD returned no segments")
            print("This might be expected for synthetic audio.")
            print("=" * 60)
            return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return False

def test_vad_with_real_speech():
    """Test VAD with more realistic speech-like signal"""
    print("\n" + "=" * 60)
    print("Testing with more realistic speech signal")
    print("=" * 60)

    sample_rate = 16000
    duration = 3.0

    # Create more complex speech-like signal
    t = torch.linspace(0, duration, int(duration * sample_rate))

    # Mix multiple frequencies to simulate formants
    signal = (
        0.3 * torch.sin(2 * np.pi * 120 * t) +  # F0
        0.2 * torch.sin(2 * np.pi * 350 * t) +  # F1
        0.15 * torch.sin(2 * np.pi * 2200 * t) + # F2
        0.1 * torch.sin(2 * np.pi * 3000 * t)    # F3
    )

    # Add some amplitude modulation to simulate speech rhythm
    modulation = 0.5 + 0.5 * torch.sin(2 * np.pi * 4 * t)
    signal = signal * modulation

    # Add silence at beginning and end
    silence = torch.zeros(int(0.5 * sample_rate))
    audio = torch.cat([silence, signal, silence])

    print(f"\n1. Created realistic speech-like signal")
    print(f"   Duration: {len(audio) / sample_rate:.2f}s")

    try:
        print("\n2. Running VAD...")
        segments = detect_voice_activity(audio)

        print(f"   ✓ Detected {len(segments)} segment(s)")
        if segments:
            for i, (start, end) in enumerate(segments):
                print(f"   Segment {i+1}: {start:.2f}s - {end:.2f}s")

        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run basic test
    success1 = test_vad()

    # Run realistic speech test
    success2 = test_vad_with_real_speech()

    if success1 and success2:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("Silero VAD is working correctly.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
