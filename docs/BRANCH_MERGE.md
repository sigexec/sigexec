# Branch/Merge DAG Support

## Overview

SigChain now supports DAG (Directed Acyclic Graph) pipelines with explicit branch and merge operations. This allows you to:

1. **Split** a pipeline into multiple parallel branches
2. **Process** each branch independently
3. **Merge** branches back together with a custom combiner function

This is different from `.variants()` which explores all parameter combinations. Branch/merge creates actual parallel processing paths in the DAG.

## API

### `.branch(labels, functions=None)`

Create named branches for parallel execution.

**Parameters:**
- `labels`: List of branch names
- `functions`: Optional list of functions to apply. If None, signal is duplicated to all branches.

**Returns:** `Pipeline` (for chaining)

### `.add(operation, branch=None)`

Add operation to specific branch or all active branches.

**Parameters:**
- `operation`: Function `SignalData -> SignalData`
- `branch`: Optional branch name. If None, applies to all active branches.

**Returns:** `Pipeline` (for chaining)

### `.merge(branch_names, combiner, output_name='merged')`

Merge multiple branches into one.

**Parameters:**
- `branch_names`: List of branch names to merge (order matters!)
- `combiner`: Function `List[SignalData] -> SignalData`
- `output_name`: Name for merged branch (default: 'merged')

**Returns:** `Pipeline` (for chaining)

## Examples

### Example 1: Simple Duplicate and Merge

```python
from sigchain import Pipeline, SignalData
import numpy as np

def multiply_by_2(sig):
    return SignalData(sig.data * 2, metadata={'sample_rate': sig.sample_rate})

def multiply_by_3(sig):
    return SignalData(sig.data * 3, metadata={'sample_rate': sig.sample_rate})

def add_branches(signals):
    return SignalData(signals[0].data + signals[1].data, 
                     metadata={'sample_rate': signals[0].sample_rate})

data = np.array([1.0, 2.0, 3.0, 4.0])

result = (Pipeline()
    .input_data(SignalData(data, metadata={'sample_rate': 1000}))
    .branch(["b1", "b2"])  # Duplicate signal to two branches
    .add(multiply_by_2, branch="b1")
    .add(multiply_by_3, branch="b2")
    .merge(["b1", "b2"], combiner=add_branches)
    .run()
)

# Result: [5, 10, 15, 20] = [1,2,3,4]*2 + [1,2,3,4]*3
```

### Example 2: Branch with Functions (No Duplication)

```python
def extract_amplitude(sig):
    return SignalData(np.abs(sig.data), 
                     metadata={'sample_rate': sig.sample_rate})

def extract_phase(sig):
    return SignalData(np.angle(sig.data), 
                     metadata={'sample_rate': sig.sample_rate})

def reconstruct(signals):
    amp, phase = signals[0], signals[1]
    return SignalData(amp.data * np.exp(1j * phase.data),
                     metadata={'sample_rate': amp.sample_rate})

data = np.array([1.0 + 2.0j, 3.0 + 4.0j])

result = (Pipeline()
    .input_data(SignalData(data, metadata={'sample_rate': 1000}))
    .branch(labels=["amp", "phase"], 
            functions=[extract_amplitude, extract_phase])
    .merge(["amp", "phase"], combiner=reconstruct)
    .run()
)

# Result reconstructs the original complex signal
```

### Example 3: Using SignalData for Scalar Parameters

```python
def identity(sig):
    return sig

def extract_parameter(sig):
    # Create a "parameter" as SignalData containing a scalar
    scale_factor = 5.0
    return SignalData(np.array([scale_factor]), 
                     metadata={'sample_rate': 1.0, 'type': 'parameter'})

def scale_by_param(signals):
    data_sig, param_sig = signals[0], signals[1]
    factor = param_sig.data[0]  # Extract scalar
    return SignalData(data_sig.data * factor,
                     metadata={'sample_rate': data_sig.sample_rate})

data = np.array([1.0, 2.0, 3.0])

result = (Pipeline()
    .input_data(SignalData(data, metadata={'sample_rate': 1000}))
    .branch(labels=["data", "param"], 
            functions=[identity, extract_parameter])
    .add(lambda sig: SignalData(sig.data + 1, 
                                metadata={'sample_rate': sig.sample_rate}), 
         branch="data")
    .merge(["data", "param"], combiner=scale_by_param)
    .run()
)

# Result: [(1+1)*5, (2+1)*5, (3+1)*5] = [10, 15, 20]
```

## Design Notes

### SignalData as Universal Container

`SignalData` can hold any numpy array - not just sampled signals:
- Signal data: `SignalData(time_series_data, ...)`
- Scalars: `SignalData(np.array([value]), ...)`
- Parameters: `SignalData(np.array([freq, bandwidth]), ...)`
- Matrices: `SignalData(2d_array, ...)`

This keeps the type system simple while allowing flexible data flow.

### Combiner Function Signature

The combiner always receives `List[SignalData]` in the order specified:

```python
pipeline.merge(["branch_a", "branch_b", "branch_c"], combiner)

def combiner(signals):
    # signals[0] is from branch_a
    # signals[1] is from branch_b  
    # signals[2] is from branch_c
    a, b, c = signals
    return SignalData(...)
```

Order is explicit and documented in the code.

### Relationship to `.variants()`

- **`.variants()`**: Explores parameter combinations, duplicates entire downstream graph
- **`.branch()/.merge()`**: Creates actual DAG structure, branches process independently

They're complementary - you can use both in the same pipeline!

## Implementation Details

- DAG execution uses topological ordering based on dependencies
- Each branch maintains separate cached results
- Branches can be merged multiple times
- Supports arbitrary DAG topologies (not just split-merge pairs)
