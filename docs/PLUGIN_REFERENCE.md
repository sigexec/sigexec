# Plugin System Quick Reference

## TL;DR

**SigChain = Framework + Your Blocks**

- **Framework**: `SignalData`, `Pipeline`, `ProcessingBlock` (in `sigchain.core`)
- **Example Blocks**: Radar blocks in `sigchain.blocks` (just examples!)
- **Your Blocks**: Easy to create, easy to distribute

## Creating a Custom Block (30 seconds)

```python
from dataclasses import dataclass
from sigchain import SignalData

@dataclass
class MyBlock:
    param: float = 1.0
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        # Your processing here
        result = signal_data.data * self.param
        
        metadata = signal_data.metadata.copy()
        metadata['my_block_applied'] = True
        
        return SignalData(result, signal_data.sample_rate, metadata)
```

That's it! No registration, no complex interfaces.

## Using Custom Blocks

```python
from sigchain import Pipeline
from sigchain.blocks import LFMGenerator  # Example block
from my_package import MyBlock            # Your block

result = (Pipeline()
    .add(LFMGenerator())
    .add(MyBlock(param=2.0))
    .run()
)
```

Works seamlessly with all Pipeline features (branching, memoization, etc.).

## Distributing Custom Blocks

### 1. Create Package Structure

```
my_blocks/
├── pyproject.toml
├── README.md
└── my_blocks/
    ├── __init__.py
    └── blocks.py
```

### 2. Set Dependencies

**pyproject.toml:**
```toml
[project]
name = "my_blocks"
version = "0.1.0"
dependencies = ["sigchain>=0.1.0"]
```

### 3. Publish

```bash
python -m build
twine upload dist/*
```

### 4. Users Install

```bash
pip install my_blocks
```

## Block Contract

✅ **Must**:
- Accept `SignalData` input (or `None` for generators)
- Return `SignalData` output
- Copy and update metadata (don't mutate input)

❌ **Don't**:
- Modify input signal data
- Require framework internals
- Use complex inheritance

## Real-World Example

**Package: sigchain-audio**

```python
# sigchain_audio/effects.py
from dataclasses import dataclass
from sigchain import SignalData
import numpy as np

@dataclass
class Reverb:
    decay: float = 0.5
    delay_ms: int = 50
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        delay_samples = int(self.delay_ms * signal_data.sample_rate / 1000)
        reverb = np.zeros_like(signal_data.data)
        
        for i in range(len(signal_data.data)):
            reverb[i] = signal_data.data[i]
            if i >= delay_samples:
                reverb[i] += self.decay * reverb[i - delay_samples]
        
        metadata = signal_data.metadata.copy()
        metadata['reverb_applied'] = True
        
        return SignalData(reverb, signal_data.sample_rate, metadata)

@dataclass
class EQ:
    low_gain: float = 1.0
    high_gain: float = 1.0
    cutoff: float = 1000.0
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        # Your EQ implementation
        ...
```

**Usage:**
```python
from sigchain import Pipeline
from sigchain_audio import Reverb, EQ

result = (Pipeline("Audio")
    .add(WavReader("input.wav"))
    .add(Reverb(decay=0.6, delay_ms=100))
    .add(EQ(low_gain=1.2, high_gain=0.8))
    .run()
)
```

## Common Patterns

### Generator Block (No Input)

```python
@dataclass
class Generator:
    def __call__(self, signal_data: SignalData = None) -> SignalData:
        # Generate signal from scratch
        data = np.random.randn(1000)
        return SignalData(data, sample_rate=44100, metadata={'generated': True})
```

### Analysis Block (Unchanged Output)

```python
@dataclass
class Analyzer:
    def __call__(self, signal_data: SignalData) -> SignalData:
        # Analyze but don't modify signal
        stats = {'mean': np.mean(signal_data.data)}
        
        metadata = signal_data.metadata.copy()
        metadata['stats'] = stats
        
        return SignalData(
            signal_data.data,  # Unchanged
            signal_data.sample_rate,
            metadata
        )
```

### Filter Block

```python
@dataclass
class Filter:
    cutoff: float
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        filtered = apply_filter(signal_data.data, self.cutoff)
        
        metadata = signal_data.metadata.copy()
        metadata['filtered'] = True
        
        return SignalData(filtered, signal_data.sample_rate, metadata)
```

## Testing Custom Blocks

```python
import numpy as np
from my_blocks import MyBlock
from sigchain import SignalData

def test_my_block():
    # Create test input
    data = np.array([1.0, 2.0, 3.0])
    signal = SignalData(data, sample_rate=1000.0, metadata={})
    
    # Apply block
    block = MyBlock(param=2.0)
    result = block(signal)
    
    # Verify
    assert np.array_equal(result.data, data * 2.0)
    assert result.sample_rate == 1000.0
    assert result.metadata['my_block_applied'] == True
```

## Key Design Decisions

### Why No Plugin Registration?

**Other frameworks:**
```python
# Complex plugin registration
@register_plugin('my_block')
class MyBlock:
    ...

framework.discover_plugins()
```

**SigChain:**
```python
# Just import and use
from my_blocks import MyBlock
```

**Reason**: Python's import system IS the plugin system. Keep it simple.

### Why Dataclasses?

**Benefits:**
- Clean syntax
- Built-in `__init__` and `__repr__`
- Type hints for IDE support
- Easy testing
- No complex inheritance

### Why SignalData?

**Consistent type** throughout pipeline:
- Type safety
- Easy composition
- Clear contracts
- Extensible via metadata

## Documentation

- **Full Guide**: [docs/CUSTOM_BLOCKS.md](../CUSTOM_BLOCKS.md)
- **Architecture**: [docs/ARCHITECTURE.md](../ARCHITECTURE.md)
- **Examples**: [examples/custom_blocks_example.py](../examples/custom_blocks_example.py)
- **Tests**: [tests/test_custom_blocks.py](../tests/test_custom_blocks.py)

## FAQ

**Q: Do I need to inherit from ProcessingBlock?**
A: No! Just implement `__call__(self, SignalData) -> SignalData`

**Q: Can I use blocks from multiple packages?**
A: Yes! Mix and match freely.

**Q: How do I handle errors?**
A: Raise exceptions normally. Pipeline will propagate them.

**Q: Can I modify the framework?**
A: Fork it! But the core is intentionally minimal.

**Q: Where should I put the radar blocks?**
A: The radar blocks in `sigchain.blocks` are examples. They can stay (for demonstration) or be moved to a separate package like `sigchain-radar`.

**Q: Will custom blocks work with future versions?**
A: Yes! The `SignalData` contract is stable. As long as you follow it, blocks will work.

## Example Packages You Could Create

- `sigchain-audio`: Audio effects and processing
- `sigchain-communications`: Modulation, demodulation, channel coding
- `sigchain-medical`: Medical imaging processing blocks
- `sigchain-video`: Video processing blocks
- `sigchain-ml`: Machine learning integration blocks
- `sigchain-io`: File readers/writers for various formats
- `sigchain-viz`: Visualization blocks

## Next Steps

1. Read [docs/CUSTOM_BLOCKS.md](../CUSTOM_BLOCKS.md) for detailed guide
2. Try [examples/custom_blocks_example.py](../examples/custom_blocks_example.py)
3. Create your own blocks!
4. Share with the community

**Remember**: SigChain provides the framework. You provide the blocks. The included radar blocks are just examples showing the pattern.
