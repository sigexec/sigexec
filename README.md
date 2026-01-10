# SigChain - Signal Processing Chain Framework

A Python framework for building signal processing pipelines using Directed Acyclic Graphs (DAGs) and data classes. This project demonstrates radar signal processing with Linear Frequency Modulated (LFM) signals, including range and Doppler compression.

## Features

- **DAG-based Architecture**: Build complex signal processing chains with clear dependencies
- **Data Class Design**: Type-safe signal data handling with metadata
- **Modular Processing Blocks**: Reusable components for different processing stages
- **Radar Processing Pipeline**: Complete example with:
  - LFM signal generation with delay and Doppler shift
  - Pulse stacking
  - Matched filtering (range compression)
  - FFT processing (Doppler compression)
  - Range-Doppler map visualization

## Installation

### From Source

```bash
git clone https://github.com/briday1/sigchain.git
cd sigchain
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.7
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0

## Quick Start

### Running the Radar Example

```bash
cd examples
python radar_range_doppler.py
```

This will:
1. Generate a synthetic LFM radar signal with a simulated target
2. Process the signal through the complete pipeline
3. Display and save a Range-Doppler map showing the target location

### Basic Usage

```python
from sigchain import SignalData, DAG
from sigchain.blocks import RadarGenerator, PulseStacker, MatchedFilter, DopplerProcessor

# Create processing blocks
radar_gen = RadarGenerator(
    num_pulses=128,
    target_delay=20e-6,  # 3 km range
    target_doppler=1000.0  # 1 kHz Doppler
)
pulse_stacker = PulseStacker()
matched_filter = MatchedFilter()
doppler_processor = DopplerProcessor()

# Build the processing chain
dag = DAG()
dag.add_chain(radar_gen, pulse_stacker, matched_filter, doppler_processor)

# Execute the chain
signal_out = radar_gen.process()
signal_out = pulse_stacker.process(signal_out)
signal_out = matched_filter.process(signal_out)
signal_out = doppler_processor.process(signal_out)

# Access the Range-Doppler map
rdm = signal_out.data
```

## Architecture

### Core Components

#### SignalData
A data class that wraps signal arrays with metadata:
```python
@dataclass
class SignalData:
    data: np.ndarray          # Signal data
    sample_rate: float        # Sampling rate
    metadata: Dict[str, Any]  # Additional information
```

#### ProcessingBlock
Abstract base class for all processing blocks:
```python
class ProcessingBlock(ABC):
    @abstractmethod
    def process(self, signal_data: SignalData) -> SignalData:
        pass
```

#### DAG
Manages execution order and dependencies:
```python
dag = DAG()
dag.add_chain(block1, block2, block3)  # Sequential chain
dag.execute(initial_data)
```

### Processing Blocks

#### RadarGenerator
Generates LFM radar signals with configurable parameters:
- Pulse duration and bandwidth
- Target delay and Doppler shift
- Noise characteristics

#### PulseStacker
Organizes pulses into a 2D matrix for coherent processing.

#### MatchedFilter
Performs range compression using matched filtering:
- Correlates received signal with transmitted waveform
- Improves SNR and range resolution

#### DopplerProcessor
Performs Doppler compression using FFT:
- FFT along pulse dimension
- Windowing for sidelobe reduction
- Generates Range-Doppler map

## Example Output

The radar example produces a Range-Doppler map showing:
- **Left plot**: 2D Range-Doppler map with target peak
- **Right plot**: Range profile at peak Doppler frequency

Target characteristics:
- Range: ~3 km (20 μs delay)
- Velocity: ~150 m/s (1 kHz Doppler at 10 GHz carrier)

## Project Structure

```
sigchain/
├── sigchain/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data.py          # SignalData class
│   │   ├── block.py         # ProcessingBlock base class
│   │   └── dag.py           # DAG implementation
│   └── blocks/
│       ├── __init__.py
│       ├── radar_generator.py
│       ├── pulse_stacker.py
│       ├── matched_filter.py
│       └── doppler_processor.py
├── examples/
│   └── radar_range_doppler.py
├── setup.py
├── requirements.txt
└── README.md
```

## Creating Custom Blocks

To create your own processing block:

```python
from sigchain.core import ProcessingBlock, SignalData

class MyCustomBlock(ProcessingBlock):
    def __init__(self, param1, param2, name=None):
        super().__init__(name)
        self.param1 = param1
        self.param2 = param2
    
    def process(self, signal_data: SignalData) -> SignalData:
        # Your processing logic here
        processed_data = your_algorithm(signal_data.data)
        
        # Update metadata
        metadata = signal_data.metadata.copy()
        metadata['my_processing'] = True
        
        return SignalData(
            data=processed_data,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

This framework demonstrates fundamental radar signal processing concepts and serves as a foundation for building more complex signal processing pipelines.