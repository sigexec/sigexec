"""
Tests for the functional/data class blocks.
"""

import numpy as np
from sigexec import GraphData, Graph
from sigexec.blocks import (
    LFMGenerator,
    StackPulses,
    RangeCompress,
    DopplerCompress,
    ToMagnitudeDB,
    Normalize
)


def test_lfm_generator():
    """Test LFMGenerator data class block."""
    print("Testing LFMGenerator...")
    
    gen = LFMGenerator(
        num_pulses=32,
        pulse_duration=5e-6,
        sample_rate=5e6,
        target_delay=10e-6
    )
    
    # Call as function
    signal = gen()
    
    assert isinstance(signal, GraphData)
    assert signal.shape[0] == 32
    assert signal.has_port('reference_pulse')
    
    print("  ✓ LFMGenerator tests passed")


def test_stack_pulses():
    """Test StackPulses data class block."""
    print("Testing StackPulses...")
    
    data = np.random.randn(16, 32) + 1j * np.random.randn(16, 32)
    sig = GraphData(data=data, metadata={'sample_rate': 1e6})
    
    stack = StackPulses()
    result = stack(sig)
    
    assert isinstance(result, GraphData)
    assert result.shape == data.shape
    assert result.pulse_stacked == True
    
    print("  ✓ StackPulses tests passed")


def test_range_compress():
    """Test RangeCompress data class block."""
    print("Testing RangeCompress...")
    
    num_pulses = 16
    num_samples = 32
    pulse_length = 16  # Reference pulse shorter than observation
    data = np.random.randn(num_pulses, num_samples) + 1j * np.random.randn(num_pulses, num_samples)
    reference_pulse = np.random.randn(pulse_length) + 1j * np.random.randn(pulse_length)
    
    sig = GraphData(
        data=data,
        metadata={
            'sample_rate': 1e6,
            'reference_pulse': reference_pulse
        }
    )
    
    compress = RangeCompress()
    result = compress(sig)
    
    assert isinstance(result, GraphData)
    # With 'same' mode: output length = num_samples (maintains input length)
    expected_output_length = num_samples
    assert result.shape == (num_pulses, expected_output_length)
    assert result.range_compressed == True
    
    print("  ✓ RangeCompress tests passed")


def test_doppler_compress():
    """Test DopplerCompress data class block."""
    print("Testing DopplerCompress...")
    
    data = np.random.randn(32, 64) + 1j * np.random.randn(32, 64)
    sig = GraphData(
        data=data,
        metadata={
            'sample_rate': 1e6,
            'pulse_repetition_interval': 1e-3
        }
    )
    
    compress = DopplerCompress(window='hann')
    result = compress(sig)
    
    assert isinstance(result, GraphData)
    assert result.shape == data.shape
    assert result.doppler_compressed == True
    assert result.has_port('doppler_frequencies')
    
    print("  ✓ DopplerCompress tests passed")


def test_to_magnitude_db():
    """Test ToMagnitudeDB data class block."""
    print("Testing ToMagnitudeDB...")
    
    data = np.random.randn(16, 32) + 1j * np.random.randn(16, 32)
    sig = GraphData(data=data, metadata={'sample_rate': 1e6})
    
    to_db = ToMagnitudeDB()
    result = to_db(sig)
    
    assert isinstance(result, GraphData)
    assert result.shape == data.shape
    assert result.dtype == np.float64
    assert result.magnitude_db == True
    
    print("  ✓ ToMagnitudeDB tests passed")


def test_normalize():
    """Test Normalize data class block."""
    print("Testing Normalize...")
    
    data = np.random.randn(16, 32) * 10 + 5
    sig = GraphData(data=data, metadata={'sample_rate': 1e6})
    
    normalize = Normalize(method='max')
    result = normalize(sig)
    
    assert isinstance(result, GraphData)
    assert result.shape == data.shape
    assert np.max(np.abs(result.data)) <= 1.0
    assert result.normalized == 'max'
    
    print("  ✓ Normalize tests passed")


def test_direct_chaining():
    """Test direct chaining of data class blocks."""
    print("Testing direct chaining...")
    
    gen = LFMGenerator(num_pulses=16, target_delay=10e-6, target_doppler=500.0)
    stack = StackPulses()
    compress_range = RangeCompress()
    compress_doppler = DopplerCompress()
    
    # Direct chaining
    signal = gen()
    signal = stack(signal)
    signal = compress_range(signal)
    signal = compress_doppler(signal)
    
    assert isinstance(signal, GraphData)
    assert signal.range_doppler_map == True
    
    print("  ✓ Direct chaining tests passed")


def test_pipeline_with_functional_blocks():
    """Test Graph with functional data class blocks."""
    print("Testing Graph with functional blocks...")
    
    graph = (Graph("TestGraph")
        .add(LFMGenerator(num_pulses=16, target_delay=5e-6))
        .add(StackPulses())
        .add(RangeCompress())
        .add(DopplerCompress())
    )
    
    assert len(graph) == 4
    
    result = graph.run()
    
    assert isinstance(result, GraphData)
    assert result.range_doppler_map == True
    
    print("  ✓ Graph with functional blocks tests passed")


def test_inline_composition():
    """Test inline functional composition."""
    print("Testing inline composition...")
    
    # One-liner composition
    result = DopplerCompress()(
        RangeCompress()(
            StackPulses()(
                LFMGenerator(num_pulses=16)()
            )
        )
    )
    
    assert isinstance(result, GraphData)
    assert result.doppler_compressed == True
    
    print("  ✓ Inline composition tests passed")


def run_all_tests():
    """Run all functional block tests."""
    print("=" * 60)
    print("Running Functional Block Tests")
    print("=" * 60)
    print()
    
    try:
        test_lfm_generator()
        test_stack_pulses()
        test_range_compress()
        test_doppler_compress()
        test_to_magnitude_db()
        test_normalize()
        test_direct_chaining()
        test_pipeline_with_functional_blocks()
        test_inline_composition()
        
        print()
        print("=" * 60)
        print("All functional block tests passed! ✓")
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
        import traceback
        traceback.print_exc()
        return False


import sys


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
