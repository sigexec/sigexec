"""
Tests for custom block extensibility.

This test file demonstrates and validates that users can create their own
custom blocks that work seamlessly with the sigchain framework.
"""

import numpy as np
from dataclasses import dataclass
from sigchain import SignalData, Pipeline, ProcessingBlock


# Test custom block using dataclass pattern
@dataclass
class TestAmplifier:
    """Test block that amplifies signal by a gain factor."""
    gain: float = 2.0
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Amplify the signal."""
        amplified = signal_data.data * self.gain
        
        metadata = signal_data.metadata.copy()
        metadata['amplified'] = True
        metadata['gain'] = self.gain
        
        return SignalData(
            data=amplified,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


# Test custom block using ProcessingBlock inheritance
class TestAttenuator(ProcessingBlock):
    """Test block that attenuates signal."""
    
    def __init__(self, attenuation=0.5, name=None):
        super().__init__(name)
        self.attenuation = attenuation
    
    def process(self, signal_data: SignalData) -> SignalData:
        """Attenuate the signal."""
        attenuated = signal_data.data * self.attenuation
        
        metadata = signal_data.metadata.copy()
        metadata['attenuated'] = True
        metadata['attenuation'] = self.attenuation
        
        return SignalData(
            data=attenuated,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


# Test custom generator block
@dataclass
class TestSignalGenerator:
    """Test block that generates a simple signal."""
    frequency: float = 1000.0
    duration: float = 0.001
    sample_rate: float = 10000.0
    
    def __call__(self, signal_data: SignalData = None) -> SignalData:
        """Generate a simple sinusoidal signal."""
        t = np.arange(0, self.duration, 1/self.sample_rate)
        signal = np.sin(2 * np.pi * self.frequency * t)
        
        return SignalData(
            data=signal,
            sample_rate=self.sample_rate,
            metadata={
                'frequency': self.frequency,
                'duration': self.duration,
                'generated': True,
            }
        )


# Test custom analysis block (doesn't modify signal)
@dataclass
class TestStatistics:
    """Test block that adds statistics to metadata."""
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Compute and add statistics."""
        data = signal_data.data
        
        stats = {
            'mean': float(np.mean(np.abs(data))),
            'std': float(np.std(np.abs(data))),
            'max': float(np.max(np.abs(data))),
            'min': float(np.min(np.abs(data))),
        }
        
        metadata = signal_data.metadata.copy()
        metadata['statistics'] = stats
        
        return SignalData(
            data=data,  # Unchanged
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


def test_custom_dataclass_block():
    """Test custom block using dataclass pattern."""
    print("Testing custom dataclass block...")
    
    # Create test signal
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    signal = SignalData(data=data, sample_rate=1000.0, metadata={})
    
    # Apply custom amplifier
    amplifier = TestAmplifier(gain=3.0)
    result = amplifier(signal)
    
    # Verify
    expected = data * 3.0
    np.testing.assert_array_almost_equal(result.data, expected)
    assert result.sample_rate == signal.sample_rate
    assert result.metadata['amplified'] == True
    assert result.metadata['gain'] == 3.0
    
    print("  ✓ Custom dataclass block works correctly")


def test_custom_processing_block():
    """Test custom block using ProcessingBlock inheritance."""
    print("Testing custom ProcessingBlock...")
    
    # Create test signal
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    signal = SignalData(data=data, sample_rate=2000.0, metadata={})
    
    # Apply custom attenuator
    attenuator = TestAttenuator(attenuation=0.5)
    result = attenuator(signal)
    
    # Verify
    expected = data * 0.5
    np.testing.assert_array_almost_equal(result.data, expected)
    assert result.sample_rate == signal.sample_rate
    assert result.metadata['attenuated'] == True
    assert result.metadata['attenuation'] == 0.5
    
    print("  ✓ Custom ProcessingBlock works correctly")


def test_custom_generator():
    """Test custom generator block."""
    print("Testing custom generator...")
    
    # Generate signal
    gen = TestSignalGenerator(frequency=500.0, duration=0.01, sample_rate=10000.0)
    signal = gen()
    
    # Verify
    assert signal.data.shape[0] == 100  # 0.01s * 10000 Hz
    assert signal.sample_rate == 10000.0
    assert signal.metadata['generated'] == True
    assert signal.metadata['frequency'] == 500.0
    
    print("  ✓ Custom generator block works correctly")


def test_custom_blocks_in_pipeline():
    """Test using custom blocks in Pipeline."""
    print("Testing custom blocks in Pipeline...")
    
    # Create pipeline with mix of custom blocks
    pipeline = (Pipeline("CustomTest")
        .add(TestSignalGenerator(frequency=1000.0), name="Generate")
        .add(TestAmplifier(gain=2.0), name="Amplify")
        .add(TestStatistics(), name="Analyze")
        .add(TestAttenuator(attenuation=0.5), name="Attenuate")
    )
    
    result = pipeline.run()
    
    # Verify pipeline executed correctly
    assert result.metadata['generated'] == True
    assert result.metadata['amplified'] == True
    assert result.metadata['attenuated'] == True
    assert 'statistics' in result.metadata
    
    # Verify final result (2.0 gain * 0.5 attenuation = 1.0x original)
    # Generate reference
    ref_gen = TestSignalGenerator(frequency=1000.0)
    ref_signal = ref_gen()
    
    # Should be close to original after amplify then attenuate
    np.testing.assert_array_almost_equal(result.data, ref_signal.data, decimal=5)
    
    print("  ✓ Custom blocks work correctly in Pipeline")


def test_custom_blocks_composition():
    """Test composing custom blocks."""
    print("Testing custom block composition...")
    
    # Create blocks
    gen = TestSignalGenerator(frequency=2000.0)
    amp = TestAmplifier(gain=5.0)
    att = TestAttenuator(attenuation=0.2)
    stats = TestStatistics()
    
    # Compose directly
    result = stats(att(amp(gen())))
    
    # Verify
    assert result.metadata['generated'] == True
    assert result.metadata['amplified'] == True
    assert result.metadata['attenuated'] == True
    assert 'statistics' in result.metadata
    
    # Net gain should be 5.0 * 0.2 = 1.0
    expected_max = result.metadata['statistics']['max']
    assert 0.8 < expected_max < 1.2  # Should be close to 1.0
    
    print("  ✓ Custom block composition works correctly")


def test_custom_blocks_branching():
    """Test custom blocks with pipeline branching."""
    print("Testing custom blocks with branching...")
    
    # Create base pipeline
    base = (Pipeline("Base")
        .add(TestSignalGenerator(frequency=1000.0))
        .add(TestAmplifier(gain=2.0))
    )
    
    # Create branches with different processing
    # Note: Due to current cache implementation, we need different operation names
    branch1 = base.branch().add(TestAttenuator(attenuation=0.1), name="Att_0.1")
    branch2 = base.branch().add(TestAttenuator(attenuation=0.5), name="Att_0.5")
    branch3 = base.branch().add(TestStatistics(), name="Stats")
    
    # Run branches (should reuse cached base results)
    result1 = branch1.run()
    result2 = branch2.run()
    result3 = branch3.run()
    
    # Verify branches produced different results due to different attenuation
    max1 = np.max(np.abs(result1.data))
    max2 = np.max(np.abs(result2.data))
    # Branch 1 has 0.1 attenuation, branch 2 has 0.5, so max1 should be smaller
    assert max1 < max2, f"Expected branch1 ({max1}) < branch2 ({max2})"
    
    # Verify branch 3 has statistics
    assert 'statistics' in result3.metadata
    
    # Verify all have amplified metadata (from shared base)
    assert result1.metadata['amplified'] == True
    assert result2.metadata['amplified'] == True
    assert result3.metadata['amplified'] == True
    
    # Verify correct attenuation values
    assert result1.metadata['attenuation'] == 0.1
    assert result2.metadata['attenuation'] == 0.5
    
    print("  ✓ Custom blocks work correctly with branching")


def test_custom_block_metadata_preservation():
    """Test that custom blocks preserve metadata correctly."""
    print("Testing metadata preservation...")
    
    # Create signal with initial metadata
    data = np.array([1.0, 2.0, 3.0])
    signal = SignalData(
        data=data,
        sample_rate=1000.0,
        metadata={'initial_key': 'initial_value', 'count': 0}
    )
    
    # Apply multiple custom blocks
    amp = TestAmplifier(gain=2.0)
    att = TestAttenuator(attenuation=0.5)
    
    result = att(amp(signal))
    
    # Verify original metadata preserved
    assert result.metadata['initial_key'] == 'initial_value'
    
    # Verify new metadata added
    assert result.metadata['amplified'] == True
    assert result.metadata['attenuated'] == True
    
    print("  ✓ Metadata preserved correctly through custom blocks")


def run_all_tests():
    """Run all custom block tests."""
    print("=" * 70)
    print("Testing Custom Block Extensibility")
    print("=" * 70)
    print()
    
    try:
        test_custom_dataclass_block()
        test_custom_processing_block()
        test_custom_generator()
        test_custom_blocks_in_pipeline()
        test_custom_blocks_composition()
        test_custom_blocks_branching()
        test_custom_block_metadata_preservation()
        
        print()
        print("=" * 70)
        print("All custom block tests passed! ✓")
        print("=" * 70)
        print()
        print("Summary:")
        print("- Custom dataclass blocks work correctly")
        print("- Custom ProcessingBlock subclasses work correctly")
        print("- Custom generators work correctly")
        print("- Custom blocks integrate with Pipeline")
        print("- Custom blocks support composition")
        print("- Custom blocks work with branching/memoization")
        print("- Metadata is preserved correctly")
        print()
        print("Conclusion: SigChain framework is fully extensible!")
        return True
        
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"Test failed: {e}")
        print("=" * 70)
        return False
    except Exception as e:
        print()
        print("=" * 70)
        print(f"Error during testing: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
