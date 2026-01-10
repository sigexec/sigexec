"""
Example: Clean DAG-like Pipeline with Data Class Blocks

This example shows the cleanest approach where:
1. Each block is a data class with configuration
2. Blocks are called directly and return SignalData
3. A single object (SignalData) flows through the pipeline
4. Dependencies are implicit through method chaining
"""

import numpy as np
import matplotlib.pyplot as plt
from sigchain import Pipeline
from sigchain.blocks import (
    LFMGenerator,
    StackPulses,
    RangeCompress,
    DopplerCompress,
    ToMagnitudeDB
)


def main():
    """Demonstrate the clean DAG-like pipeline."""
    
    print("=" * 70)
    print("Clean DAG-like Pipeline with Data Class Blocks")
    print("=" * 70)
    
    # Configure blocks as data classes
    generate = LFMGenerator(
        num_pulses=128,
        pulse_duration=10e-6,
        pulse_repetition_interval=1e-3,
        sample_rate=10e6,
        bandwidth=5e6,
        target_delay=20e-6,
        target_doppler=1000.0,
        noise_power=0.1
    )
    
    stack = StackPulses()
    compress_range = RangeCompress()
    compress_doppler = DopplerCompress(window='hann')
    
    print("\nApproach 1: Manual chaining (most explicit)")
    print("-" * 70)
    
    # Call blocks in sequence - single object flows through
    signal = generate()
    signal = stack(signal)
    signal = compress_range(signal)
    signal = compress_doppler(signal)
    
    print(f"  Final shape: {signal.shape}")
    print(f"  Data type: {signal.dtype}")
    print(f"  Has range-doppler map: {signal.metadata.get('range_doppler_map', False)}")
    
    display_result(signal, "clean_dag_manual.png")
    
    print("\nApproach 2: Using Pipeline for better organization")
    print("-" * 70)
    
    # Build pipeline with data class blocks
    pipeline = (Pipeline("RadarDAG")
        .add(generate, name="Generate")
        .add(stack, name="Stack")
        .add(compress_range, name="RangeCompress")
        .add(compress_doppler, name="DopplerCompress")
    )
    
    print(f"  {pipeline}")
    
    # Execute - returns single SignalData object
    result = pipeline.run(verbose=True)
    
    display_result(result, "clean_dag_pipeline.png")
    
    print("\nApproach 3: Inline configuration (most compact)")
    print("-" * 70)
    
    # Create and execute pipeline in one go
    result = (Pipeline("CompactRadar")
        .add(LFMGenerator(
            num_pulses=64,
            target_delay=15e-6,
            target_doppler=500.0
        ))
        .add(StackPulses())
        .add(RangeCompress())
        .add(DopplerCompress(window='hann'))
        .tap(lambda sig: print(f"  Peak magnitude: {np.max(np.abs(sig.data)):.2f}"))
        .run()
    )
    
    print(f"  Result shape: {result.shape}")
    
    print("\n" + "=" * 70)
    print("Complete! All approaches work with the same SignalData object.")
    print("=" * 70)


def simple_example():
    """
    Simplest possible example showing the data class approach.
    """
    
    print("\n" + "=" * 70)
    print("Simplest Example: Direct Block Composition")
    print("=" * 70)
    
    # Configure processing blocks
    gen = LFMGenerator(num_pulses=32, target_delay=10e-6, target_doppler=500.0)
    stack = StackPulses()
    range_comp = RangeCompress()
    doppler_comp = DopplerCompress()
    
    # Process: single object flows through
    signal = gen()
    signal = stack(signal)
    signal = range_comp(signal)
    signal = doppler_comp(signal)
    
    print(f"\nInput: None (generator)")
    print(f"Output: SignalData with shape {signal.shape}")
    print(f"Metadata keys: {list(signal.metadata.keys())[:5]}...")
    
    # Each stage returns SignalData - same type throughout!
    print(f"\nType consistency:")
    print(f"  After generate: {type(gen()).__name__}")
    print(f"  After stack: {type(stack(gen())).__name__}")
    print(f"  After range compress: {type(range_comp(stack(gen()))).__name__}")
    print(f"  After doppler compress: {type(doppler_comp(range_comp(stack(gen())))).__name__}")


def display_result(signal_data, filename):
    """Display the range-doppler map."""
    rdm = signal_data.data
    metadata = signal_data.metadata
    
    # Convert to dB if not already
    if not metadata.get('magnitude_db', False):
        rdm_db = 20 * np.log10(np.abs(rdm) + 1e-10)
    else:
        rdm_db = rdm
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_doppler, num_range = rdm_db.shape
    range_axis = np.arange(num_range) / signal_data.sample_rate * 3e8 / 2 / 1000
    doppler_axis = metadata.get('doppler_frequencies', np.arange(num_doppler))
    
    im = ax.imshow(
        rdm_db,
        aspect='auto',
        extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        origin='lower',
        cmap='jet',
        vmin=np.max(rdm_db) - 50,
        vmax=np.max(rdm_db)
    )
    
    ax.set_xlabel('Range (km)', fontsize=12)
    ax.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
    ax.set_title('Range-Doppler Map (Clean DAG)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    # Mark target if known
    if 'target_delay' in metadata and 'target_doppler' in metadata:
        target_range = metadata['target_delay'] * 3e8 / 2 / 1000
        target_doppler = metadata['target_doppler']
        ax.plot(target_range, target_doppler, 'rx', markersize=12, 
                markeredgewidth=2, label='True Target')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    # Run the main examples
    main()
    
    # Show simplest example
    simple_example()
