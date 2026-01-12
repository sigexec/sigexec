# Clean DAG-like Signal Processing

This document explains the clean, data-class-based approach to building signal processing graphs in sigexec.

## Core Concept

Each processing block is a **data class** (using `@dataclass`) that:
1. Stores configuration parameters
2. Is callable via `__call__()` method
3. Takes `SignalData` as input (or None for generators)
4. Returns `SignalData` as output
5. Can be chained naturally

## Benefits

- **Type Safety**: Same type (`SignalData`) flows through entire graph
- **Composability**: Blocks can be combined in any order
- **Clarity**: Configuration separate from execution
- **Immutability**: Each block returns new data, doesn't modify input
- **Testability**: Each block is independently testable

## Available Data Class Blocks

### Signal Generation

```python
from sigexec.blocks import LFMGenerator

gen = LFMGenerator(
    num_pulses=128,
    pulse_duration=10e-6,
    sample_rate=10e6,
    bandwidth=5e6,
    target_delay=20e-6,
    target_doppler=1000.0
)

signal = gen()  # Returns SignalData
```

### Processing Blocks

```python
from sigexec.blocks import (
    StackPulses,      # Organize pulses in 2D matrix
    RangeCompress,    # Matched filtering
    DopplerCompress,  # FFT-based Doppler processing
    ToMagnitudeDB,    # Convert to dB scale
    Normalize         # Normalize data
)

stack = StackPulses()
compress_range = RangeCompress()
compress_doppler = DopplerCompress(window='hann')
to_db = ToMagnitudeDB()
normalize = Normalize(method='max')
```

## Usage Patterns

### Pattern 1: Direct Chaining (Most Explicit)

The simplest approach - directly call blocks in sequence:

```python
# Configure blocks
gen = LFMGenerator(num_pulses=128, target_delay=20e-6)
stack = StackPulses()
range_comp = RangeCompress()
doppler_comp = DopplerCompress()

# Single object flows through
signal = gen()
signal = stack(signal)
signal = range_comp(signal)
signal = doppler_comp(signal)

# signal is now the range-doppler map!
```

**Key Point**: The same `SignalData` object type flows through every stage.

### Pattern 2: Using Graph (Better Organization)

Use the `Graph` class for better structure and debugging:

```python
from sigexec import Graph

# Build graph
graph = (Graph("MyRadar")
    .add(gen)
    .add(stack)
    .add(range_comp)
    .add(doppler_comp)
)

# Execute
result = graph.run(verbose=True)
```

### Pattern 3: Inline Configuration (Most Compact)

Configure blocks inline for quick prototyping:

```python
result = (Graph("QuickRadar")
    .add(LFMGenerator(num_pulses=64, target_delay=15e-6))
    .add(StackPulses())
    .add(RangeCompress())
    .add(DopplerCompress(window='hann'))
    .run()
)
```

## Complete Example

```python
from sigexec import Graph
from sigexec.blocks import (
    LFMGenerator,
    StackPulses,
    RangeCompress,
    DopplerCompress
)

# Configure processing blocks
gen = LFMGenerator(
    num_pulses=128,
    pulse_duration=10e-6,
    pulse_repetition_interval=1e-3,
    sample_rate=10e6,
    bandwidth=5e6,
    target_delay=20e-6,  # ~3 km range
    target_doppler=1000.0,  # ~150 m/s velocity at 10 GHz
    noise_power=0.1
)

stack = StackPulses()
range_comp = RangeCompress()
doppler_comp = DopplerCompress(window='hann')

# Method 1: Direct chaining
signal = doppler_comp(range_comp(stack(gen())))

# Method 2: Step by step (easier to debug)
signal = gen()
signal = stack(signal)
signal = range_comp(signal)
signal = doppler_comp(signal)

# Method 3: Using Graph
result = (Graph("Radar")
    .add(gen)
    .add(stack)
    .add(range_comp)
    .add(doppler_comp)
    .run()
)

# Access result
range_doppler_map = result.data
metadata = result.metadata
```

## Creating Custom Blocks

To create your own block, use the `@dataclass` decorator:

```python
from dataclasses import dataclass
import numpy as np
from sigexec import SignalData

@dataclass
class MyCustomBlock:
    """My custom processing block."""
    
    # Configuration parameters
    param1: float = 1.0
    param2: str = 'default'
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Process the signal."""
        
        # Your processing logic
        processed_data = your_algorithm(signal_data.data, self.param1)
        
        # Update metadata
        metadata = signal_data.metadata.copy()
        metadata['my_custom_processing'] = True
        metadata['param1_used'] = self.param1
        
        # Return new SignalData
        return SignalData(
            data=processed_data,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )

# Use it
my_block = MyCustomBlock(param1=2.5, param2='custom')
result = my_block(input_signal)
```

## Key Design Principles

1. **Immutability**: Blocks don't modify input, they create new output
2. **Type Consistency**: Every block takes and returns `SignalData`
3. **Configuration vs Execution**: Separate initialization (config) from calling (execution)
4. **Metadata Propagation**: Each block preserves and enhances metadata
5. **Independence**: Each block can be tested and used independently

## Best Practices

1. **Name your blocks**: Use descriptive names for debugging
2. **Check metadata**: Use metadata to verify processing stages
3. **Test incrementally**: Run partial graphs during development
4. **Use tap()**: Add inspection points without modifying data
5. **Keep blocks focused**: Each block should do one thing well
6. **Document parameters**: Use docstrings in custom blocks
