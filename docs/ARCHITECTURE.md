# SigChain Architecture: Framework vs. Blocks

## Overview

SigChain follows a clean separation between **framework** and **blocks**:

- **Framework** (Core): The infrastructure for building pipelines (`SignalData`, `Pipeline`, `ProcessingBlock`)
- **Blocks** (Extensions): Processing operations that transform signals

This design allows the framework to remain minimal and focused while supporting unlimited extensibility.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Applications                     │
│         (Radar, Audio, Medical, Communications)         │
└─────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Custom Block Packages                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│   │ audio-blocks │  │ radar-blocks │  │ medical-     │ │
│   │              │  │   (example)  │  │  blocks      │ │
│   └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   SigChain Framework                     │
│   ┌──────────────────────────────────────────────────┐  │
│   │  Core Components                                 │  │
│   │  • SignalData: Type-safe data container         │  │
│   │  • Pipeline: Execution and composition engine   │  │
│   │  • ProcessingBlock: Base abstraction (optional) │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Core Framework Components

### 1. SignalData
**Purpose**: Type-safe container for signal data flowing through pipelines

```python
@dataclass
class SignalData:
    data: np.ndarray          # Signal samples
    sample_rate: float        # Sampling rate
    metadata: Dict[str, Any]  # Extensible metadata
```

**Design Decision**: Single, consistent type throughout all processing stages enables:
- Type safety
- Easy composition
- Clear contracts between blocks

### 2. Pipeline
**Purpose**: Execution engine with composition and optimization features

**Features**:
- Sequential execution
- Branching and variants
- Automatic memoization
- Inspection hooks (`.tap()`)
- Verbose debugging

**Example**:
```python
result = (Pipeline("Example")
    .add(Block1())
    .add(Block2())
    .add(Block3())
    .run()
)
```

### 3. ProcessingBlock (Optional)
**Purpose**: Abstract base for traditional OOP blocks

**Note**: This is **optional**. The recommended approach is dataclass blocks that implement `__call__`.

## Block Extension Model

### Recommended Pattern: Dataclass Blocks

Blocks are simple dataclasses implementing `__call__`:

```python
from dataclasses import dataclass
from sigchain import SignalData

@dataclass
class MyBlock:
    """A custom processing block."""
    param: float = 1.0
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        # Process and return new SignalData
        return SignalData(...)
```

**Benefits**:
- Simple and pythonic
- No inheritance required
- Easy to test
- Configuration via dataclass fields
- Type hints for IDE support

### Block Contract

All blocks must follow this simple contract:

1. **Input**: Accept `SignalData` (or `None` for generators)
2. **Output**: Return `SignalData`
3. **Immutability**: Don't modify input, create new output
4. **Metadata**: Copy and update metadata dictionary

That's it! No complex interfaces or inheritance hierarchies.

## Package Structure

### Framework Package (sigchain)

```
sigchain/
├── __init__.py           # Exports: SignalData, Pipeline, ProcessingBlock
├── core/
│   ├── data.py          # SignalData class
│   ├── pipeline.py      # Pipeline execution engine
│   ├── block.py         # ProcessingBlock base (optional)
│   └── dag.py           # DAG implementation (legacy)
└── blocks/              # Example blocks (can be in separate package)
    ├── functional.py    # Example radar blocks
    ├── radar_generator.py
    └── ...
```

### Custom Block Package (e.g., sigchain-audio)

```
sigchain_audio/
├── pyproject.toml       # Depends on sigchain>=0.1.0
├── README.md
└── sigchain_audio/
    ├── __init__.py      # Export your blocks
    ├── filters.py       # Your custom blocks
    └── effects.py
```

## Extension Mechanisms

### 1. Direct Import Pattern

Users import and use blocks directly:

```python
from sigchain import Pipeline
from sigchain.blocks import LFMGenerator  # Example block
from my_blocks import CustomFilter        # Your block

result = (Pipeline()
    .add(LFMGenerator())
    .add(CustomFilter())
    .run()
)
```

### 2. Namespace Packages (Advanced)

For larger ecosystems, use namespace packages:

```python
# sigchain/plugins/radar.py
# sigchain/plugins/audio.py

from sigchain.plugins import radar, audio
```

### 3. Configuration-based Discovery (Future)

Potential future enhancement:

```yaml
# pipeline.yaml
pipeline:
  - block: sigchain.blocks.LFMGenerator
    params:
      num_pulses: 128
  - block: my_blocks.CustomFilter
    params:
      cutoff: 1000
```

## Design Principles

### 1. Minimal Framework
The framework provides only essential infrastructure:
- Data container (`SignalData`)
- Execution engine (`Pipeline`)
- Optional base class (`ProcessingBlock`)

Everything else is an extension.

### 2. No Plugin Registration
Unlike some frameworks, SigChain doesn't require registering plugins or entry points. Just import and use.

**Why?** Simplicity. Python's import system is the plugin system.

### 3. Explicit Dependencies
Each block package declares its dependencies:

```toml
dependencies = [
    "sigchain>=0.1.0",
    "numpy>=1.20.0",
    "scipy>=1.7.0",  # If needed by your blocks
]
```

### 4. Framework Stability
The core framework (`SignalData`, `Pipeline`) has a stable API. Blocks can evolve independently.

### 5. No Framework Lock-in
Blocks are just functions that transform `SignalData`. They can be:
- Used standalone
- Wrapped in other frameworks
- Combined with non-sigchain code

## Example: Multiple Block Packages

```python
from sigchain import Pipeline

# Framework + Example blocks
from sigchain.blocks import LFMGenerator, StackPulses

# Third-party audio blocks
from sigchain_audio import Reverb, Compressor

# Third-party medical imaging blocks
from sigchain_medical import DICOMReader, SegmentationBlock

# Your custom blocks
from my_project.blocks import CustomAnalyzer

# All work together seamlessly
result = (Pipeline("Multi-domain")
    .add(LFMGenerator())       # Example radar block
    .add(StackPulses())        # Example radar block
    .add(CustomAnalyzer())     # Your custom block
    .run()
)
```

## Benefits of This Architecture

### For Framework Maintainers
- **Small codebase**: Core framework is ~500 lines
- **Stable API**: Minimal surface area reduces breaking changes
- **Easy testing**: Core components are isolated and testable
- **Clear boundaries**: Framework vs. blocks separation

### For Block Developers
- **No framework expertise needed**: Just implement `__call__`
- **Independent versioning**: Blocks evolve separately from framework
- **Easy distribution**: Standard Python packaging
- **Full Python ecosystem**: Use any libraries you need

### For Users
- **Choose what you need**: Install only blocks you use
- **Mix and match**: Combine blocks from different sources
- **No lock-in**: Easy to migrate or extend
- **IDE support**: Type hints throughout

## Comparison to Other Frameworks

### SigChain vs. Monolithic Frameworks

**Monolithic** (e.g., TensorFlow, PyTorch):
- Framework includes many operations
- Large installation size
- Harder to extend
- Framework updates affect all blocks

**SigChain** (Modular):
- Framework is minimal (~500 LOC)
- Blocks are separate packages
- Easy to extend
- Framework and blocks evolve independently

### SigChain vs. Plugin Systems

**Plugin Systems** (e.g., pytest, Flask):
- Explicit plugin registration
- Entry points or decorators
- Framework discovers plugins

**SigChain** (Import-based):
- No registration required
- Just import and use
- Python import system is the "plugin system"

## Future Enhancements

Possible additions that maintain the architecture:

1. **Block Registry** (Optional): For discoverability
2. **Type System**: Enhanced type checking for block compatibility
3. **Serialization**: Save/load pipeline configurations
4. **Visualization**: Pipeline DAG visualization
5. **Parallel Execution**: Multi-threaded/multi-process execution

All these would be **optional** features that don't break the core simplicity.

## Conclusion

SigChain's architecture separates framework from blocks:

- **Framework**: Minimal, stable infrastructure for building pipelines
- **Blocks**: Extensible processing operations distributed as separate packages
- **Integration**: Simple import-based composition, no registration required

This design enables:
- Easy extensibility
- Independent evolution
- Simple mental model
- No framework lock-in

The included radar blocks are **examples** demonstrating the pattern. Users should create their own blocks for their specific domains.
