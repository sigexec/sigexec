# Answering the Plugin Architecture Question

## Original Question

> This is a bit of a question more than an ask for implementation. 
> 
> What if we have someone who wants to create their own blocks, but use them with sigchain? 
> 
> Another way to say that is, do we want to keep the actual blocks in here, or just the framework for making/running them?
> 
> How would be allow for new blocks with other installed modules?

## Answer

### 1. Creating Custom Blocks - Very Easy!

**Short Answer:** Anyone can create custom blocks by simply following the dataclass pattern:

```python
from dataclasses import dataclass
from sigchain import SignalData

@dataclass
class MyCustomBlock:
    param: float = 1.0
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        # Your processing here
        result = signal_data.data * self.param
        metadata = signal_data.metadata.copy()
        metadata['my_block'] = True
        return SignalData(result, signal_data.sample_rate, metadata)
```

That's it! No inheritance required, no registration, no complex interfaces.

**See:** [docs/CUSTOM_BLOCKS.md](docs/CUSTOM_BLOCKS.md) for complete guide

### 2. Should We Keep the Blocks or Just the Framework?

**Answer:** Keep both, but with clear separation.

**Framework (Core)**:
- `sigchain.core.data` - SignalData class
- `sigchain.core.pipeline` - Pipeline execution engine
- `sigchain.core.block` - ProcessingBlock base class (optional)

**Blocks (Examples)**:
- `sigchain.blocks.*` - Radar processing blocks

**Recommendation:**
1. **Keep the radar blocks** - They serve as excellent examples and demonstrations
2. **Clearly document** they are examples (✅ Done - see updated README and blocks/__init__.py)
3. **Show users how to create their own** (✅ Done - comprehensive documentation)

The framework is already properly separated and minimal (~500 LOC core). The radar blocks don't bloat it and are valuable as reference implementations.

### 3. How to Allow Blocks from Other Installed Modules?

**Answer:** Python's import system IS the plugin system.

**Pattern:**
```python
from sigchain import Pipeline

# Built-in example blocks
from sigchain.blocks import LFMGenerator, StackPulses

# Third-party blocks from other packages
from sigchain_audio import Reverb, Compressor
from sigchain_medical import DICOMReader
from my_company.blocks import ProprietaryBlock

# Mix and match freely!
result = (Pipeline()
    .add(LFMGenerator())
    .add(ProprietaryBlock())
    .add(Reverb())
    .run()
)
```

**No registration needed** - just install the package and import the blocks.

**Creating a Plugin Package:**

```bash
# Package structure
my_blocks/
├── pyproject.toml          # depends on sigchain>=0.1.0
└── my_blocks/
    ├── __init__.py
    └── filters.py

# Install
pip install sigchain  # Framework
pip install my_blocks # Your blocks

# Use
from my_blocks import CustomFilter
```

**See:** [docs/PLUGIN_REFERENCE.md](docs/PLUGIN_REFERENCE.md) for quick reference

## Implementation Summary

This PR provides:

### Documentation (3 comprehensive guides)
1. **CUSTOM_BLOCKS.md** - Complete guide to creating and distributing custom blocks
2. **ARCHITECTURE.md** - Framework design and separation of concerns
3. **PLUGIN_REFERENCE.md** - Quick reference with examples

### Working Example
- **custom_blocks_example.py** - Demonstrates 4 different custom blocks working with framework

### Tests
- **test_custom_blocks.py** - 7 tests validating extensibility

### Updates
- **README.md** - Clarified framework vs blocks, added quick links
- **blocks/__init__.py** - Added documentation explaining blocks are examples

## Design Decisions

### 1. No Plugin Registration
**Why:** Python's import system is sufficient. Adding registration would complicate without benefit.

**Comparison:**
```python
# Other frameworks (complex)
@register_plugin('my_block')
class MyBlock:
    ...
framework.discover_plugins()

# SigChain (simple)
from my_blocks import MyBlock
```

### 2. Dataclass Pattern
**Why:** 
- Clean, modern Python
- No inheritance required
- Easy to understand and test
- IDE support via type hints

### 3. Single Type (SignalData)
**Why:**
- Type safety
- Clear contract
- Easy composition
- Extensible via metadata

### 4. Keep Example Blocks
**Why:**
- Demonstrate best practices
- Useful for radar domain users
- Reference implementation
- Already well-implemented

## Benefits

### For Users
✅ Easy to create custom blocks  
✅ No framework expertise needed  
✅ Mix blocks from any source  
✅ Standard Python packaging  

### For Framework
✅ Stays minimal and focused  
✅ No plugin management complexity  
✅ Stable API (just SignalData)  
✅ Easy to maintain  

### For Ecosystem
✅ Anyone can create block packages  
✅ Domain-specific packages possible  
✅ No central registry needed  
✅ Natural Python workflow  

## Example Use Cases

### Audio Processing Package
```python
# Package: sigchain-audio
from sigchain_audio import Reverb, EQ, Compressor

pipeline = (Pipeline()
    .add(WavReader('input.wav'))
    .add(Reverb(decay=0.5))
    .add(EQ(bass=1.2, treble=0.8))
    .add(Compressor(ratio=4.0))
    .run()
)
```

### Medical Imaging Package
```python
# Package: sigchain-medical
from sigchain_medical import DICOMReader, Denoise, Segment

pipeline = (Pipeline()
    .add(DICOMReader('scan.dcm'))
    .add(Denoise(method='bilateral'))
    .add(Segment(threshold=0.5))
    .run()
)
```

### Communications Package
```python
# Package: sigchain-comms
from sigchain_comms import Modulate, ChannelCode, Demodulate

pipeline = (Pipeline()
    .add(Modulate(type='QPSK'))
    .add(ChannelCode(type='LDPC'))
    .add(AddNoise(snr_db=10))
    .add(Demodulate(type='QPSK'))
    .run()
)
```

## Conclusion

**The framework is already extensible!** 

What was needed (and now provided):
1. ✅ Clear documentation showing how to create blocks
2. ✅ Examples of custom blocks
3. ✅ Tests validating extensibility
4. ✅ Guide to distributing block packages
5. ✅ Clarification of framework vs blocks separation

**Answer to "should we keep the blocks?"**: Yes, as examples. The framework is properly separated in `sigchain.core`, and the radar blocks in `sigchain.blocks` serve as valuable reference implementations.

**Answer to "how to allow blocks from other modules?"**: Just import them. Python's import system handles everything. No registration or discovery mechanism needed.

## Next Steps for Users

1. Read [docs/CUSTOM_BLOCKS.md](docs/CUSTOM_BLOCKS.md)
2. Try [examples/custom_blocks_example.py](examples/custom_blocks_example.py)
3. Create your own blocks!
4. (Optional) Publish as a package for others to use

The framework makes it easy. The documentation makes it clear. The examples show the way.
