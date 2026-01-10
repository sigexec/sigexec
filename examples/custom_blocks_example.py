"""
Example: Creating and Using Custom Blocks

This example demonstrates how users can create their own custom processing
blocks and use them with the sigchain framework.
"""

from dataclasses import dataclass
import numpy as np
from sigchain import Pipeline, SignalData
from sigchain.blocks import LFMGenerator, StackPulses


# Example 1: Simple custom block using dataclass pattern
@dataclass
class AmplitudeThreshold:
    """
    Custom block that applies amplitude thresholding.
    
    Zeros out any values below the threshold.
    """
    threshold: float = 0.5
    mode: str = 'absolute'  # 'absolute' or 'relative'
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Apply threshold to signal."""
        data = signal_data.data.copy()
        
        if self.mode == 'relative':
            # Threshold relative to max value
            threshold_val = self.threshold * np.max(np.abs(data))
        else:
            # Absolute threshold
            threshold_val = self.threshold
        
        # Apply threshold
        data[np.abs(data) < threshold_val] = 0
        
        # Update metadata
        metadata = signal_data.metadata.copy()
        metadata['threshold_applied'] = True
        metadata['threshold_value'] = float(threshold_val)
        metadata['threshold_mode'] = self.mode
        
        return SignalData(
            data=data,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


# Example 2: Custom filter block
@dataclass
class ExponentialSmoothing:
    """
    Custom exponential smoothing filter.
    
    Applies exponential weighted moving average.
    """
    alpha: float = 0.3  # Smoothing factor (0-1)
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Apply exponential smoothing."""
        data = signal_data.data
        
        # Handle complex data
        if np.iscomplexobj(data):
            real_smooth = self._smooth(data.real)
            imag_smooth = self._smooth(data.imag)
            smoothed = real_smooth + 1j * imag_smooth
        else:
            smoothed = self._smooth(data)
        
        metadata = signal_data.metadata.copy()
        metadata['exponential_smoothing_applied'] = True
        metadata['smoothing_alpha'] = self.alpha
        
        return SignalData(
            data=smoothed,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )
    
    def _smooth(self, data: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to real-valued array."""
        smoothed = np.zeros_like(data)
        
        if data.ndim == 1:
            smoothed[0] = data[0]
            for i in range(1, len(data)):
                smoothed[i] = self.alpha * data[i] + (1 - self.alpha) * smoothed[i-1]
        else:
            # Apply along last axis
            for i in range(data.shape[0]):
                smoothed[i, 0] = data[i, 0]
                for j in range(1, data.shape[1]):
                    smoothed[i, j] = self.alpha * data[i, j] + (1 - self.alpha) * smoothed[i, j-1]
        
        return smoothed


# Example 3: Custom statistical analysis block
@dataclass
class StatisticalAnalyzer:
    """
    Custom block that computes statistics without modifying the signal.
    
    This demonstrates a block that adds information to metadata
    without changing the signal data.
    """
    compute_percentiles: bool = True
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Analyze signal statistics."""
        data = signal_data.data
        magnitude = np.abs(data)
        
        # Compute statistics
        stats = {
            'mean': float(np.mean(magnitude)),
            'std': float(np.std(magnitude)),
            'max': float(np.max(magnitude)),
            'min': float(np.min(magnitude)),
        }
        
        if self.compute_percentiles:
            stats['percentiles'] = {
                'p25': float(np.percentile(magnitude, 25)),
                'p50': float(np.percentile(magnitude, 50)),
                'p75': float(np.percentile(magnitude, 75)),
                'p95': float(np.percentile(magnitude, 95)),
                'p99': float(np.percentile(magnitude, 99)),
            }
        
        # Add to metadata without changing data
        metadata = signal_data.metadata.copy()
        metadata['statistics'] = stats
        
        return SignalData(
            data=data,  # Unchanged
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


# Example 4: Custom normalization with multiple modes
@dataclass
class CustomNormalizer:
    """
    Advanced normalization block with multiple modes.
    """
    mode: str = 'max'  # 'max', 'rms', 'peak', 'zscore'
    target_level: float = 1.0
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Normalize signal according to specified mode."""
        data = signal_data.data
        
        if self.mode == 'max':
            # Normalize to maximum absolute value
            normalized = data / (np.max(np.abs(data)) + 1e-10) * self.target_level
            
        elif self.mode == 'rms':
            # Normalize to RMS level
            rms = np.sqrt(np.mean(np.abs(data) ** 2))
            normalized = data / (rms + 1e-10) * self.target_level
            
        elif self.mode == 'peak':
            # Peak normalization (max of real and imaginary)
            peak = max(np.max(np.abs(data.real)), np.max(np.abs(data.imag)))
            normalized = data / (peak + 1e-10) * self.target_level
            
        elif self.mode == 'zscore':
            # Z-score normalization
            mean = np.mean(data)
            std = np.std(data)
            normalized = (data - mean) / (std + 1e-10)
            
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")
        
        metadata = signal_data.metadata.copy()
        metadata['normalized'] = self.mode
        metadata['target_level'] = self.target_level
        
        return SignalData(
            data=normalized,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


def demonstrate_custom_blocks():
    """Demonstrate using custom blocks in a pipeline."""
    
    print("=" * 70)
    print("Custom Blocks Example")
    print("=" * 70)
    
    # Generate test signal
    print("\n1. Generating test signal...")
    gen = LFMGenerator(
        num_pulses=64,
        pulse_duration=10e-6,
        sample_rate=10e6,
        target_delay=15e-6,
        target_doppler=500.0,
        noise_power=0.05
    )
    
    signal = gen()
    print(f"   Generated signal shape: {signal.shape}")
    print(f"   Signal dtype: {signal.dtype}")
    
    # Example 1: Direct chaining with custom blocks
    print("\n2. Using custom blocks directly...")
    
    threshold = AmplitudeThreshold(threshold=0.3, mode='relative')
    smoothing = ExponentialSmoothing(alpha=0.2)
    analyzer = StatisticalAnalyzer()
    
    result = analyzer(smoothing(threshold(signal)))
    
    print(f"   After processing shape: {result.shape}")
    print(f"   Statistics computed: {list(result.metadata['statistics'].keys())}")
    print(f"   Mean magnitude: {result.metadata['statistics']['mean']:.4f}")
    print(f"   Max magnitude: {result.metadata['statistics']['max']:.4f}")
    
    # Example 2: Using Pipeline with mix of built-in and custom blocks
    print("\n3. Using Pipeline with custom blocks...")
    
    pipeline = (Pipeline("CustomExample")
        .add(LFMGenerator(num_pulses=32, target_delay=10e-6), name="Generate")
        .add(StackPulses(), name="Stack")
        .add(CustomNormalizer(mode='rms', target_level=1.0), name="Normalize")
        .add(AmplitudeThreshold(threshold=0.1, mode='relative'), name="Threshold")
        .add(StatisticalAnalyzer(), name="Analyze")
    )
    
    result = pipeline.run(verbose=True)
    
    print(f"\n   Final result shape: {result.shape}")
    print(f"   Threshold applied: {result.metadata.get('threshold_applied', False)}")
    print(f"   Normalized mode: {result.metadata.get('normalized', 'none')}")
    
    # Example 3: Testing different normalization modes
    print("\n4. Comparing normalization modes...")
    
    test_signal = gen()
    modes = ['max', 'rms', 'peak', 'zscore']
    
    for mode in modes:
        normalizer = CustomNormalizer(mode=mode, target_level=1.0)
        normalized = normalizer(test_signal)
        
        max_val = np.max(np.abs(normalized.data))
        mean_val = np.mean(np.abs(normalized.data))
        
        print(f"   Mode '{mode}': max={max_val:.4f}, mean={mean_val:.4f}")
    
    # Example 4: Branching with custom blocks
    print("\n5. Using branching with custom blocks...")
    
    base = (Pipeline("Base")
        .add(gen)
        .add(StackPulses())
    )
    
    # Create branches with different processing
    branch_threshold_low = base.branch().add(AmplitudeThreshold(threshold=0.2, mode='relative'))
    branch_threshold_high = base.branch().add(AmplitudeThreshold(threshold=0.5, mode='relative'))
    branch_smooth = base.branch().add(ExponentialSmoothing(alpha=0.3))
    
    result_low = branch_threshold_low.run()
    result_high = branch_threshold_high.run()  # Reuses cached base results!
    result_smooth = branch_smooth.run()  # Reuses cached base results!
    
    print(f"   Branch 1 (low threshold): {np.count_nonzero(result_low.data)} non-zero elements")
    print(f"   Branch 2 (high threshold): {np.count_nonzero(result_high.data)} non-zero elements")
    print(f"   Branch 3 (smoothed): mean = {np.mean(np.abs(result_smooth.data)):.4f}")
    
    print("\n" + "=" * 70)
    print("Custom blocks demonstration complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("- Custom blocks are easy to create using @dataclass")
    print("- They work seamlessly with built-in blocks")
    print("- Pipeline features (branching, memoization) work with custom blocks")
    print("- Blocks can add information to metadata without changing signal")
    print("- You can distribute custom blocks as separate packages")


if __name__ == "__main__":
    demonstrate_custom_blocks()
