"""
Tests for custom block extensibility.

This test file demonstrates and validates that users can create their own
custom blocks that work seamlessly with the sigchain framework.
"""

import numpy as np
from dataclasses import dataclass
from sigexec import GraphData, Graph


# Custom block using dataclass pattern
@dataclass
class CustomAmplifier:
    """Custom block that amplifies signal by a gain factor."""
    gain: float = 2.0
    
    def __call__(self, gdata: GraphData) -> GraphData:
        """Amplify the signal."""
        amplified = gdata.data * self.gain
        gdata.data = amplified
        gdata.amplified = True
        gdata.gain = self.gain
        
        return gdata


# Custom block using dataclass pattern with attenuation
@dataclass
class CustomAttenuator:
    """Custom block that attenuates signal."""
    attenuation: float = 0.5
    
    def __call__(self, gdata: GraphData) -> GraphData:
        """Attenuate the signal."""
        attenuated = gdata.data * self.attenuation
        gdata.data = attenuated
        gdata.attenuated = True
        gdata.attenuation = self.attenuation
        
        return gdata


# Custom generator block
@dataclass
class CustomSignalGenerator:
    """Custom block that generates a simple signal."""
    frequency: float = 1000.0
    duration: float = 0.001
    sample_rate: float = 10000.0
    
    def __call__(self, gdata: GraphData = None) -> GraphData:
        """Generate a simple sinusoidal signal."""
        t = np.arange(0, self.duration, 1/self.sample_rate)
        signal = np.sin(2 * np.pi * self.frequency * t)
        
        return GraphData(
            data=signal,
            metadata={
                'sample_rate': self.sample_rate,
                'frequency': self.frequency,
                'duration': self.duration,
                'generated': True,
            }
        )


# Custom analysis block (doesn't modify signal)
@dataclass
class CustomStatistics:
    """Custom block that adds statistics to ports."""
    
    def __call__(self, gdata: GraphData) -> GraphData:
        """Compute and add statistics."""
        data = gdata.data
        
        stats = {
            'mean': float(np.mean(np.abs(data))),
            'std': float(np.std(np.abs(data))),
            'max': float(np.max(np.abs(data))),
            'min': float(np.min(np.abs(data))),
        }
        gdata.statistics = stats
        
        return gdata


def test_custom_dataclass_block():
    """Test custom block using dataclass pattern."""
    print("Testing custom dataclass block...")
    
    # Create test signal
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    signal = GraphData(data=data, metadata={'sample_rate': 1000.0})
    
    # Apply custom amplifier
    amplifier = CustomAmplifier(gain=3.0)
    result = amplifier(signal)
    
    # Verify
    expected = data * 3.0
    np.testing.assert_array_almost_equal(result.data, expected)
    assert result.sample_rate == signal.sample_rate
    assert result.amplified == True
    assert result.gain == 3.0
    
    print("  ✓ Custom dataclass block works correctly")


def test_custom_generator():
    """Test custom generator block."""
    print("Testing custom generator...")
    
    # Generate signal
    gen = CustomSignalGenerator(frequency=500.0, duration=0.01, sample_rate=10000.0)
    signal = gen()
    
    # Verify
    assert signal.data.shape[0] == 100  # 0.01s * 10000 Hz
    assert signal.sample_rate == 10000.0
    assert signal.generated == True
    assert signal.frequency == 500.0
    
    print("  ✓ Custom generator block works correctly")


def test_custom_blocks_in_pipeline():
    """Test using custom blocks in Graph."""
    print("Testing custom blocks in Graph...")
    
    # Create graph with mix of custom blocks
    graph = (Graph("CustomTest")
        .add(CustomSignalGenerator(frequency=1000.0), name="Generate")
        .add(CustomAmplifier(gain=2.0), name="Amplify")
        .add(CustomStatistics(), name="Analyze")
        .add(CustomAttenuator(attenuation=0.5), name="Attenuate")
    )
    
    result = graph.run()
    
    # Verify graph executed correctly
    assert result.generated == True
    assert result.amplified == True
    assert result.attenuated == True
    assert result.has_port('statistics')
    
    # Verify final result (2.0 gain * 0.5 attenuation = 1.0x original)
    # Generate reference
    ref_gen = CustomSignalGenerator(frequency=1000.0)
    ref_signal = ref_gen()
    
    # Should be close to original after amplify then attenuate
    np.testing.assert_array_almost_equal(result.data, ref_signal.data, decimal=5)
    
    print("  ✓ Custom blocks work correctly in Graph")


def test_custom_blocks_composition():
    """Test composing custom blocks."""
    print("Testing custom block composition...")
    
    # Create blocks
    gen = CustomSignalGenerator(frequency=2000.0)
    amp = CustomAmplifier(gain=5.0)
    att = CustomAttenuator(attenuation=0.2)
    stats = CustomStatistics()
    
    # Compose directly
    result = stats(att(amp(gen())))
    
    # Verify
    assert result.generated == True
    assert result.amplified == True
    assert result.attenuated == True
    assert result.has_port('statistics')
    
    # Net gain should be 5.0 * 0.2 = 1.0
    expected_max = result.statistics['max']
    assert 0.8 < expected_max < 1.2  # Should be close to 1.0
    
    print("  ✓ Custom block composition works correctly")


def test_custom_blocks_multiple_graphs():
    """Test custom blocks work in multiple independent graphs."""
    print("Testing custom blocks in multiple graphs...")
    
    # Create three independent graphs with shared initial stages
    gen = CustomSignalGenerator(frequency=1000.0)
    amp = CustomAmplifier(gain=2.0)
    
    # Disable cache to ensure independent execution
    graph1 = Graph("Graph1", enable_cache=False).add(gen).add(amp).add(CustomAttenuator(attenuation=0.1))
    graph2 = Graph("Graph2", enable_cache=False).add(gen).add(amp).add(CustomAttenuator(attenuation=0.5))
    graph3 = Graph("Graph3", enable_cache=False).add(gen).add(amp).add(CustomStatistics())
    
    # Run graphs independently
    result1 = graph1.run()
    result2 = graph2.run()
    result3 = graph3.run()
    
    # Verify graphs produced different results due to different attenuation
    max1 = np.max(np.abs(result1.data))
    max2 = np.max(np.abs(result2.data))
    # Graph 1 has 0.1 attenuation, graph 2 has 0.5, so max1 should be smaller
    assert max1 < max2, f"Expected graph1 ({max1}) < graph2 ({max2})"
    
    # Verify graph 3 has statistics
    assert result3.has_port('statistics')
    
    # Verify all have amplified port (from shared base)
    assert result1.amplified == True
    assert result2.amplified == True
    assert result3.amplified == True
    
    # Verify correct attenuation values
    assert result1.attenuation == 0.1
    assert result2.attenuation == 0.5
    
    print("  ✓ Custom blocks work correctly in multiple graphs")


def test_custom_block_metadata_preservation():
    """Test that custom blocks preserve metadata correctly."""
    print("Testing metadata preservation...")
    
    # Create signal with initial metadata
    data = np.array([1.0, 2.0, 3.0])
    signal = GraphData(
        data=data,
        metadata={'sample_rate': 1000.0, 'initial_key': 'initial_value', 'count': 0}
    )
    
    # Apply multiple custom blocks
    amp = CustomAmplifier(gain=2.0)
    att = CustomAttenuator(attenuation=0.5)
    
    result = att(amp(signal))
    
    # Verify original metadata preserved
    assert result.initial_key == 'initial_value'
    
    # Verify new metadata added
    assert result.amplified == True
    assert result.attenuated == True
    
    print("  ✓ Metadata preserved correctly through custom blocks")


def run_all_tests():
    """Run all custom block tests."""
    print("=" * 70)
    print("Testing Custom Block Extensibility")
    print("=" * 70)
    print()
    
    try:
        test_custom_dataclass_block()
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
        print("- Custom generators work correctly")
        print("- Custom blocks integrate with Graph")
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
