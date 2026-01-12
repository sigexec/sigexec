# SigChain - Quick Reference

## Three Ways to Build Signal Processing Pipelines

### 1. Clean DAG with Data Classes (RECOMMENDED)

**Simplest and most readable** - Direct chaining with configured blocks:

```python
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress

# Configure blocks (data classes)
gen = LFMGenerator(num_pulses=128, target_delay=20e-6, target_doppler=1000.0)
stack = StackPulses()
compress_range = RangeCompress()
compress_doppler = DopplerCompress(window='hann')

# Single SignalData object flows through
signal = gen()
signal = stack(signal)
signal = compress_range(signal)
signal = compress_doppler(signal)

# Result is ready to use
range_doppler_map = signal.data
```

**Key Features:**
- ✅ Cleanest syntax
- ✅ Type-safe (SignalData in/out)
- ✅ Easy to debug (step through each stage)
- ✅ No boilerplate
- ✅ Configuration separate from execution

### 2. Graph with Fluent Interface

**Best for complex chains** - Better organization and debugging:

```python
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress

# Build graph with fluent chaining
result = (Graph("RadarProcessing")
    .add(LFMGenerator(num_pulses=128, target_delay=20e-6, target_doppler=1000.0))
    .add(StackPulses())
    .add(RangeCompress())
    .add(DopplerCompress(window='hann'))
    .tap(lambda sig: print(f"Peak: {np.max(np.abs(sig.data))}"))  # Inspect
    .run(verbose=True)  # Execute with logging
)

range_doppler_map = result.data
```

**Key Features:**
- ✅ Named graph for clarity
- ✅ Verbose mode for debugging
- ✅ Tap points for inspection
- ✅ Clean execution logging
- ✅ Same SignalData consistency

## Comparison

| Feature | Direct Chaining | Graph |
|---------|-----------------|----------|
| Lines of code | Fewest | Medium |
| Readability | Excellent | Good |
| Debugging | Easy | Excellent |
| Type safety | ✅ | ✅ |
| Configuration | Dataclass | Dataclass |
| Execution | Direct call | `.run()` |
| Best for | Simple chains | Complex graphs |

## Which Should I Use?

- **Starting fresh?** → Use **Direct Chaining** (Approach 1)
- **Complex graph with branching?** → Use **Graph** (Approach 2)

## Quick Examples

### Generate and Process in One Line
```python
result = DopplerCompress()(RangeCompress()(StackPulses()(LFMGenerator()())))
```

### Custom Processing Block
```python
from dataclasses import dataclass
from sigexec import SignalData

@dataclass
class MyBlock:
    param: float = 1.0
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        processed = signal_data.data * self.param
        return SignalData(processed, signal_data.sample_rate, signal_data.metadata)

# Use it
my_block = MyBlock(param=2.5)
result = my_block(input_signal)
```

### Add Custom Operation to Graph
```python
graph = (Graph()
    .add(LFMGenerator())
    .transform(lambda data: data * 2)  # Custom transform on data array
    .add(StackPulses())
    .tap(lambda sig: print(f"Shape: {sig.shape}"))  # Inspect
    .run()
)
```

## Complete Radar Example

```python
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress

# Configure radar
gen = LFMGenerator(
    num_pulses=128,
    pulse_duration=10e-6,
    pulse_repetition_interval=1e-3,
    sample_rate=10e6,
    bandwidth=5e6,
    target_delay=20e-6,      # ~3 km range
    target_doppler=1000.0,   # ~150 m/s velocity
    noise_power=0.1
)

# Process (choose your style)

# Style 1: Direct
signal = DopplerCompress()(RangeCompress()(StackPulses()(gen())))

# Style 2: Step by step
signal = gen()
signal = StackPulses()(signal)
signal = RangeCompress()(signal)
signal = DopplerCompress()(signal)

# Style 3: Graph
signal = (Graph()
    .add(gen)
    .add(StackPulses())
    .add(RangeCompress())
    .add(DopplerCompress())
    .run()
)

# All produce the same result!
import matplotlib.pyplot as plt
import numpy as np

rdm_db = 20 * np.log10(np.abs(signal.data) + 1e-10)
plt.imshow(rdm_db, aspect='auto', cmap='jet')
plt.xlabel('Range')
plt.ylabel('Doppler')
plt.title('Range-Doppler Map')
plt.colorbar(label='Magnitude (dB)')
plt.savefig('range_doppler_map.png')
```

## See Also

- **[README.md](../README.md)** - Full documentation
- **[CLEAN_DAG.md](CLEAN_DAG.md)** - Detailed guide to data class approach
- **[examples/](../examples/)** - Working code examples
- **[tests/](../tests/)** - Unit tests showing usage patterns
