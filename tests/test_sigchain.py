"""
Simple tests to verify the signal processing chain implementation.
"""

import numpy as np
from sigchain import SignalData, DAG, ProcessingBlock
from sigchain.blocks import RadarGenerator, PulseStacker, MatchedFilter, DopplerProcessor


def test_signal_data():
    """Test SignalData class."""
    print("Testing SignalData...")
    
    # Create signal data
    data = np.random.randn(10, 20) + 1j * np.random.randn(10, 20)
    sig = SignalData(data=data, sample_rate=1e6, metadata={'test': True})
    
    assert sig.shape == (10, 20), f"Expected shape (10, 20), got {sig.shape}"
    assert sig.dtype == complex, f"Expected complex dtype, got {sig.dtype}"
    assert sig.metadata['test'] == True
    
    # Test copy
    sig_copy = sig.copy()
    sig_copy.data[0, 0] = 999
    assert sig.data[0, 0] != 999, "Copy should be independent"
    
    print("  ✓ SignalData tests passed")


def test_processing_block():
    """Test ProcessingBlock base class."""
    print("Testing ProcessingBlock...")
    
    class TestBlock(ProcessingBlock):
        def process(self, signal_data):
            # Simple pass-through
            return signal_data
    
    block1 = TestBlock(name="Block1")
    block2 = TestBlock(name="Block2")
    
    # Test connection
    block1.connect(block2)
    assert block2 in block1.outputs
    assert block1 in block2.inputs
    
    print("  ✓ ProcessingBlock tests passed")


def test_dag():
    """Test DAG class."""
    print("Testing DAG...")
    
    class TestBlock(ProcessingBlock):
        def process(self, signal_data):
            return signal_data
    
    dag = DAG()
    block1 = TestBlock(name="Block1")
    block2 = TestBlock(name="Block2")
    block3 = TestBlock(name="Block3")
    
    # Add chain
    dag.add_chain(block1, block2, block3)
    
    assert len(dag.blocks) == 3
    assert block2 in block1.outputs
    assert block3 in block2.outputs
    
    print("  ✓ DAG tests passed")


def test_radar_generator():
    """Test RadarGenerator block."""
    print("Testing RadarGenerator...")
    
    radar_gen = RadarGenerator(
        num_pulses=64,
        pulse_duration=10e-6,
        sample_rate=10e6,
        target_delay=20e-6,
        target_doppler=1000.0
    )
    
    signal_out = radar_gen.process()
    
    assert signal_out.shape[0] == 64, f"Expected 64 pulses, got {signal_out.shape[0]}"
    assert signal_out.dtype == np.complex128
    assert 'reference_pulse' in signal_out.metadata
    assert 'target_delay' in signal_out.metadata
    
    print("  ✓ RadarGenerator tests passed")


def test_pulse_stacker():
    """Test PulseStacker block."""
    print("Testing PulseStacker...")
    
    # Create test data
    data = np.random.randn(32, 50) + 1j * np.random.randn(32, 50)
    sig = SignalData(data=data, sample_rate=1e6, metadata={})
    
    stacker = PulseStacker()
    sig_out = stacker.process(sig)
    
    assert sig_out.shape == data.shape
    assert sig_out.metadata['pulse_stacked'] == True
    
    print("  ✓ PulseStacker tests passed")


def test_matched_filter():
    """Test MatchedFilter block."""
    print("Testing MatchedFilter...")
    
    # Create test data with reference pulse
    num_pulses = 32
    num_samples = 50
    pulse_length = 20  # Reference pulse shorter than observation
    data = np.random.randn(num_pulses, num_samples) + 1j * np.random.randn(num_pulses, num_samples)
    reference_pulse = np.random.randn(pulse_length) + 1j * np.random.randn(pulse_length)
    
    sig = SignalData(
        data=data,
        sample_rate=1e6,
        metadata={'reference_pulse': reference_pulse}
    )
    
    mf = MatchedFilter()
    sig_out = mf.process(sig)
    
    # With 'valid' mode: output length = num_samples - pulse_length + 1
    expected_output_length = num_samples - pulse_length + 1
    assert sig_out.shape == (num_pulses, expected_output_length)
    assert sig_out.metadata['range_compressed'] == True
    
    print("  ✓ MatchedFilter tests passed")


def test_doppler_processor():
    """Test DopplerProcessor block."""
    print("Testing DopplerProcessor...")
    
    # Create test data
    num_pulses = 64
    num_samples = 50
    data = np.random.randn(num_pulses, num_samples) + 1j * np.random.randn(num_pulses, num_samples)
    
    sig = SignalData(
        data=data,
        sample_rate=1e6,
        metadata={'pulse_repetition_interval': 1e-3}
    )
    
    dp = DopplerProcessor(window='hann')
    sig_out = dp.process(sig)
    
    assert sig_out.shape == data.shape
    assert sig_out.metadata['doppler_compressed'] == True
    assert 'doppler_frequencies' in sig_out.metadata
    
    print("  ✓ DopplerProcessor tests passed")


def test_complete_pipeline():
    """Test the complete radar processing pipeline."""
    print("Testing complete pipeline...")
    
    # Create minimal pipeline
    radar_gen = RadarGenerator(
        num_pulses=32,
        pulse_duration=5e-6,
        sample_rate=5e6,
        target_delay=10e-6,
        target_doppler=500.0
    )
    
    pulse_stacker = PulseStacker()
    matched_filter = MatchedFilter()
    doppler_processor = DopplerProcessor()
    
    # Execute pipeline
    sig = radar_gen.process()
    sig = pulse_stacker.process(sig)
    sig = matched_filter.process(sig)
    sig = doppler_processor.process(sig)
    
    # Verify output
    assert sig.shape[0] == 32  # num_pulses
    assert sig.metadata['range_doppler_map'] == True
    
    # Verify there's a peak (target detection)
    magnitude = np.abs(sig.data)
    max_val = np.max(magnitude)
    mean_val = np.mean(magnitude)
    assert max_val > 2 * mean_val, "Expected clear target peak"
    
    print("  ✓ Complete pipeline tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Signal Processing Chain Tests")
    print("=" * 60)
    print()
    
    try:
        test_signal_data()
        test_processing_block()
        test_dag()
        test_radar_generator()
        test_pulse_stacker()
        test_matched_filter()
        test_doppler_processor()
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
