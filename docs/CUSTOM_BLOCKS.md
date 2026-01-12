# Creating Custom Blocks

SigChain is designed to be extensible. You can create your own custom processing blocks and use them seamlessly with the framework.

## Overview

SigChain provides the **framework** for building signal processing graphs. The radar processing blocks included in `sigexec.blocks` are **examples** of how to use the framework. You are encouraged to create your own blocks for your specific use cases.

## Two Approaches for Custom Blocks

### 1. Data Class Blocks (Recommended)

The modern, recommended approach uses Python data classes:

```python
from dataclasses import dataclass
from sigexec import SignalData
import numpy as np

@dataclass
class MyCustomBlock:
    """My custom processing block."""
    
    # Configuration parameters
    gain: float = 1.0
    threshold: float = 0.5
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Process the signal."""
        # Apply your custom processing
        processed = signal_data.data * self.gain
        processed[np.abs(processed) < self.threshold] = 0
        
        # Preserve and update metadata
        metadata = signal_data.metadata.copy()
        metadata['custom_processed'] = True
        metadata['gain_applied'] = self.gain
        
        # Return new SignalData
        return SignalData(
            data=processed,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )

# Usage
custom = MyCustomBlock(gain=2.0, threshold=0.1)
result = custom(input_signal)
```

## Creating a Custom Block Package

You can distribute your custom blocks as a separate Python package that depends on sigexec.

### Step 1: Create Package Structure

```
my_blocks/
├── pyproject.toml
├── README.md
└── my_blocks/
    ├── __init__.py
    ├── filters.py
    └── transforms.py
```

### Step 2: Define Your Blocks

**my_blocks/filters.py:**
```python
from dataclasses import dataclass
from sigexec import SignalData
from scipy import signal
import numpy as np

@dataclass
class BandpassFilter:
    """Bandpass filter block."""
    
    low_freq: float
    high_freq: float
    order: int = 4
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Apply bandpass filter."""
        nyquist = signal_data.sample_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        b, a = signal.butter(self.order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data.data, axis=-1)
        
        metadata = signal_data.metadata.copy()
        metadata['filtered'] = True
        metadata['filter_type'] = 'bandpass'
        metadata['low_freq'] = self.low_freq
        metadata['high_freq'] = self.high_freq
        
        return SignalData(
            data=filtered,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )

@dataclass
class MovingAverage:
    """Moving average filter."""
    
    window_size: int = 5
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Apply moving average."""
        kernel = np.ones(self.window_size) / self.window_size
        smoothed = np.convolve(signal_data.data.flatten(), kernel, mode='same')
        smoothed = smoothed.reshape(signal_data.data.shape)
        
        metadata = signal_data.metadata.copy()
        metadata['smoothed'] = True
        metadata['window_size'] = self.window_size
        
        return SignalData(
            data=smoothed,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )
```

**my_blocks/__init__.py:**
```python
"""Custom signal processing blocks."""

from .filters import BandpassFilter, MovingAverage

__all__ = ['BandpassFilter', 'MovingAverage']
```

### Step 3: Define Package Dependencies

**pyproject.toml:**
```toml
[build-system]
requires = ["setuptools>=45"]
build-backend = "setuptools.build_meta"

[project]
name = "my_blocks"
version = "0.1.0"
description = "Custom signal processing blocks for sigexec"
requires-python = ">=3.7"
dependencies = [
    "sigexec>=0.1.0",
    "numpy>=1.20.0",
    "scipy>=1.7.0",
]
```

### Step 4: Install and Use

```bash
# Install your custom blocks package
pip install -e .

# Use in your code
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses
from my_blocks import BandpassFilter, MovingAverage

result = (Graph("CustomPipeline")
    .add(LFMGenerator(num_pulses=128))
    .add(BandpassFilter(low_freq=1e6, high_freq=5e6))
    .add(MovingAverage(window_size=3))
    .add(StackPulses())
    .run()
)
```

## Block Design Guidelines

### 1. Follow the SignalData Contract

All blocks must:
- Accept `SignalData` as input (or `None` for generators)
- Return `SignalData` as output
- Preserve `sample_rate` unless explicitly transforming time/frequency domain
- Copy and update `metadata` dictionary

### 2. Keep Blocks Focused

Each block should perform **one specific operation**. For complex processing:
```python
# Good: Separate blocks for separate operations
result = normalize(filter(denoise(signal)))

# Avoid: One block doing everything
result = mega_processor(signal)  # Does too much
```

### 3. Make Blocks Configurable

Use dataclass fields or `__init__` parameters:
```python
@dataclass
class ConfigurableBlock:
    param1: float = 1.0
    param2: str = 'default'
    param3: bool = True
```

### 4. Document Your Blocks

```python
@dataclass
class WellDocumentedBlock:
    """
    A well-documented processing block.
    
    This block performs X operation on the input signal.
    It is useful for Y scenarios and assumes Z about the input data.
    
    Attributes:
        param1: Controls the strength of the effect (range: 0-1)
        param2: Type of processing to apply ('fast' or 'accurate')
    
    Example:
        >>> block = WellDocumentedBlock(param1=0.5)
        >>> result = block(input_signal)
    """
    param1: float = 0.5
    param2: str = 'fast'
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """
        Process the signal data.
        
        Args:
            signal_data: Input signal to process
            
        Returns:
            Processed signal data
            
        Raises:
            ValueError: If param1 is outside valid range
        """
        if not 0 <= self.param1 <= 1:
            raise ValueError("param1 must be between 0 and 1")
        
        # ... processing logic ...
```

### 5. Update Metadata Appropriately

```python
def __call__(self, signal_data: SignalData) -> SignalData:
    # Always copy metadata first
    metadata = signal_data.metadata.copy()
    
    # Add information about what this block did
    metadata['my_processing_applied'] = True
    metadata['parameters_used'] = {
        'param1': self.param1,
        'param2': self.param2,
    }
    
    # Add any results or derived information
    metadata['peak_value'] = float(np.max(np.abs(processed)))
    
    return SignalData(data=processed, sample_rate=sr, metadata=metadata)
```

### 6. Handle Edge Cases

```python
def __call__(self, signal_data: SignalData) -> SignalData:
    # Check input validity
    if signal_data.data.size == 0:
        raise ValueError("Cannot process empty signal")
    
    # Handle different input shapes
    original_shape = signal_data.data.shape
    if signal_data.data.ndim == 1:
        data = signal_data.data.reshape(1, -1)
    else:
        data = signal_data.data
    
    # ... process ...
    
    # Restore original shape if needed
    result = processed.reshape(original_shape)
```

## Testing Custom Blocks

Create tests for your custom blocks:

```python
import numpy as np
from sigexec import SignalData
from my_blocks import BandpassFilter

def test_bandpass_filter():
    """Test bandpass filter."""
    # Create test signal
    data = np.random.randn(100) + 1j * np.random.randn(100)
    signal = SignalData(data=data, sample_rate=10e6, metadata={})
    
    # Apply filter
    filt = BandpassFilter(low_freq=1e6, high_freq=5e6)
    result = filt(signal)
    
    # Verify output
    assert result.shape == signal.shape
    assert result.sample_rate == signal.sample_rate
    assert result.metadata['filtered'] == True
    assert 'filter_type' in result.metadata
    
    print("✓ Bandpass filter test passed")

if __name__ == "__main__":
    test_bandpass_filter()
```

## Example: Complete Custom Block Package

Here's a complete example of a custom block package for audio processing:

**audio_blocks/__init__.py:**
```python
from dataclasses import dataclass
import numpy as np
from sigexec import SignalData

@dataclass
class Reverb:
    """Add reverb effect to signal."""
    
    decay: float = 0.5
    delay_samples: int = 1000
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Apply reverb effect."""
        signal = signal_data.data
        reverb = np.zeros_like(signal)
        
        for i in range(len(signal)):
            reverb[i] = signal[i]
            if i >= self.delay_samples:
                reverb[i] += self.decay * reverb[i - self.delay_samples]
        
        metadata = signal_data.metadata.copy()
        metadata['reverb_applied'] = True
        
        return SignalData(
            data=reverb,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )

@dataclass
class Compressor:
    """Dynamic range compressor."""
    
    threshold: float = -20.0  # dB
    ratio: float = 4.0
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Apply compression."""
        signal_db = 20 * np.log10(np.abs(signal_data.data) + 1e-10)
        
        # Apply compression above threshold
        mask = signal_db > self.threshold
        excess_db = signal_db[mask] - self.threshold
        signal_db[mask] = self.threshold + excess_db / self.ratio
        
        # Convert back to linear
        compressed = 10 ** (signal_db / 20) * np.exp(1j * np.angle(signal_data.data))
        
        metadata = signal_data.metadata.copy()
        metadata['compressed'] = True
        
        return SignalData(
            data=compressed,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )

__all__ = ['Reverb', 'Compressor']
```

## Integration with Graph

Your custom blocks work seamlessly with all sigexec features:

```python
from sigexec import Graph
from my_blocks import BandpassFilter, MovingAverage

# Direct chaining
filtered = BandpassFilter(1e6, 5e6)(input_signal)
smoothed = MovingAverage(5)(filtered)

# Graph
result = (Graph("CustomPipeline")
    .add(BandpassFilter(1e6, 5e6))
    .add(MovingAverage(5))
    .tap(lambda s: print(f"Intermediate shape: {s.shape}"))
    .run()
)

# Branching
base = Graph().add(BandpassFilter(1e6, 5e6))
branch1 = base.branch().add(MovingAverage(5))
branch2 = base.branch().add(MovingAverage(10))

result1 = branch1.run()  # Uses cached filter result
result2 = branch2.run()  # Reuses cached filter result
```

## Publishing Your Blocks

To share your custom blocks with others:

1. **Choose a descriptive name**: `sigexec-audio-blocks`, `sigexec-radar-extras`, etc.
2. **Add sigexec as a dependency** in your `pyproject.toml`
3. **Write good documentation** in your README
4. **Include examples** showing how to use your blocks
5. **Publish to PyPI**: `python -m build && twine upload dist/*`

## Summary

- **Framework vs. Blocks**: SigChain provides the framework. The included radar blocks are examples.
- **Easy Extension**: Create custom blocks following the `SignalData` contract
- **Separate Packages**: Distribute custom blocks as independent packages
- **Full Integration**: Custom blocks work with all Graph features (branching, memoization, etc.)
- **Community**: Share your blocks with others via PyPI

The framework is designed to be minimal and focused, while allowing unlimited extensibility through custom blocks.
