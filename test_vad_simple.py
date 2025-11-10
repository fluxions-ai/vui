"""
Simple test to verify the VAD module can be imported and check its structure.
This test works without needing to actually run the model.
"""

import sys
import inspect

print("=" * 60)
print("Testing Silero VAD Module Structure")
print("=" * 60)

try:
    # Test 1: Import the module
    print("\n1. Testing module import...")
    sys.path.insert(0, '/home/user/vui/src')
    from vui import vad
    print("   ✓ Module imported successfully")

    # Test 2: Check detect_voice_activity function exists
    print("\n2. Checking detect_voice_activity function...")
    assert hasattr(vad, 'detect_voice_activity'), "detect_voice_activity function not found"
    print("   ✓ detect_voice_activity function exists")

    # Test 3: Check function signature
    print("\n3. Checking function signature...")
    sig = inspect.signature(vad.detect_voice_activity)
    params = list(sig.parameters.keys())
    print(f"   Parameters: {params}")
    assert 'waveform' in params, "waveform parameter missing"
    assert 'pipe' in params, "pipe parameter missing"
    print("   ✓ Function signature is correct")

    # Test 4: Check merge_segments helper exists
    print("\n4. Checking merge_segments helper function...")
    assert hasattr(vad, 'merge_segments'), "merge_segments function not found"
    print("   ✓ merge_segments function exists")

    # Test 5: Verify no pyannote imports
    print("\n5. Verifying pyannote removal...")
    with open('/home/user/vui/src/vui/vad.py', 'r') as f:
        content = f.read()
        assert 'pyannote' not in content.lower(), "pyannote still referenced in code"
        assert 'silero' in content.lower() or 'torch.hub' in content.lower(), "Silero VAD not found in code"
    print("   ✓ pyannote successfully removed")
    print("   ✓ Silero VAD implementation found")

    # Test 6: Check imports in the vad.py file
    print("\n6. Checking imports...")
    import_lines = [line for line in content.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
    print(f"   Found {len(import_lines)} import statements:")
    for line in import_lines:
        print(f"     - {line.strip()}")
    assert any('torch' in line for line in import_lines), "torch import missing"
    print("   ✓ Required imports present")

    print("\n" + "=" * 60)
    print("✓ ALL STRUCTURE TESTS PASSED!")
    print("=" * 60)
    print("\nThe Silero VAD module is correctly structured.")
    print("To fully test functionality, run the code with PyTorch installed.")
    print("\nKey changes verified:")
    print("  • pyannote imports removed")
    print("  • Silero VAD implementation added")
    print("  • Function signatures maintained for compatibility")
    print("=" * 60)

except AssertionError as e:
    print(f"\n✗ TEST FAILED: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
