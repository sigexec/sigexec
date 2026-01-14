"""
Simple tests to verify the signal processing chain implementation.
"""

import numpy as np
from sigexec import GraphData, Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress


def test_gdata():
    """Test GraphData class."""
    print("Testing GraphData...")
    
    # Create signal data
    data = np.random.randn(10, 20) + 1j * np.random.randn(10, 20)
    sig = GraphData(data=data, metadata={'sample_rate': 1e6, 'test': True})
    
    assert sig.shape == (10, 20), f"Expected shape (10, 20), got {sig.shape}"
    assert sig.dtype == complex, f"Expected complex dtype, got {sig.dtype}"
    assert sig.test == True
    
    # Test copy
    sig_copy = sig.copy()
    sig_copy.data[0, 0] = 999
    assert sig.data[0, 0] != 999, "Copy should be independent"
    
    print("  ✓ GraphData tests passed")


def test_lfm_generator():
    """Test LFMGenerator block."""
    print("Testing LFMGenerator...")
    
    gen = LFMGenerator(
        num_pulses=64,
        pulse_duration=10e-6,
        sample_rate=10e6,
        target_delay=20e-6,
        target_doppler=1000.0
    )
    
    signal_out = gen()
    
    assert signal_out.shape[0] == 64, f"Expected 64 pulses, got {signal_out.shape[0]}"
    assert signal_out.dtype == np.complex128
    assert signal_out.has_port('reference_pulse')
    assert signal_out.has_port('target_delay')
    
    print("  ✓ LFMGenerator tests passed")


def test_stack_pulses():
    """Test StackPulses block."""
    print("Testing StackPulses...")
    
    # Create test data
    data = np.random.randn(32, 50) + 1j * np.random.randn(32, 50)
    sig = GraphData(data=data, metadata={'sample_rate': 1e6})
    
    stacker = StackPulses()
    sig_out = stacker(sig)
    
    assert sig_out.shape == data.shape
    assert sig_out.pulse_stacked == True
    
    print("  ✓ StackPulses tests passed")


def test_range_compress():
    """Test RangeCompress block."""
    print("Testing RangeCompress...")
    
    # Create test data with reference pulse
    num_pulses = 32
    num_samples = 100
    pulse_length = 20
    data = np.random.randn(num_pulses, num_samples) + 1j * np.random.randn(num_pulses, num_samples)
    reference_pulse = np.random.randn(pulse_length) + 1j * np.random.randn(pulse_length)
    
    sig = GraphData(
        data=data,
        metadata={
            'sample_rate': 1e6,
            'reference_pulse': reference_pulse
        }
    )
    
    rc = RangeCompress()
    sig_out = rc(sig)
    
    # Output should be same or larger due to oversample_factor
    assert sig_out.shape[0] == num_pulses
    assert sig_out.range_compressed == True
    
    print("  ✓ RangeCompress tests passed")


def test_doppler_compress():
    """Test DopplerCompress block."""
    print("Testing DopplerCompress...")
    
    # Create test data
    num_pulses = 64
    num_samples = 50
    data = np.random.randn(num_pulses, num_samples) + 1j * np.random.randn(num_pulses, num_samples)
    
    sig = GraphData(
        data=data,
        metadata={
            'sample_rate': 1e6,
            'pulse_repetition_interval': 1e-3
        }
    )
    
    dc = DopplerCompress(window='hann')
    sig_out = dc(sig)
    
    assert sig_out.shape[0] == num_pulses  # May be larger with oversample_factor
    assert sig_out.doppler_compressed == True
    assert sig_out.has_port('doppler_frequencies')
    
    print("  ✓ DopplerCompress tests passed")


def test_pipeline():
    """Test Graph execution."""
    print("Testing Graph...")
    
    result = (Graph("Test")
        .add(LFMGenerator(num_pulses=32, target_delay=2e-6, target_doppler=300.0))
        .add(StackPulses())
        .add(RangeCompress())
        .add(DopplerCompress())
        .run()
    )
    
    assert result.shape[0] == 32  # num_pulses (may be larger with oversample_factor)
    assert result.range_doppler_map == True
    
    # Verify there's a peak (target detection)
    magnitude = np.abs(result.data)
    max_val = np.max(magnitude)
    mean_val = np.mean(magnitude)
    assert max_val > 2 * mean_val, "Expected clear target peak"
    
    print("  ✓ Graph tests passed")


def test_complete_pipeline():
    """Test the complete radar processing graph with direct chaining."""
    print("Testing complete graph...")
    
    # Create blocks
    gen = LFMGenerator(
        num_pulses=32,
        pulse_duration=5e-6,
        sample_rate=5e6,
        target_delay=2e-6,
        target_doppler=300.0
    )
    stack = StackPulses()
    range_comp = RangeCompress()
    doppler_comp = DopplerCompress()
    
    # Execute graph via direct chaining
    sig = gen()
    sig = stack(sig)
    sig = range_comp(sig)
    sig = doppler_comp(sig)
    
    # Verify output
    assert sig.shape[0] == 32  # num_pulses
    assert sig.range_doppler_map == True
    
    # Verify there's a peak (target detection)
    magnitude = np.abs(sig.data)
    max_val = np.max(magnitude)
    mean_val = np.mean(magnitude)
    assert max_val > 2 * mean_val, "Expected clear target peak"
    
    print("  ✓ Complete graph tests passed")

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Signal Processing Chain Tests")
    print("=" * 60)
    print()
    
    try:
        test_gdata()
        test_lfm_generator()
        test_stack_pulses()
        test_range_compress()
        test_doppler_compress()
        test_pipeline()
        test_complete_pipeline()
        
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Error during testing: {e}")
        print("=" * 60)
        return False


import sys


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
