"""
Example: Radar Range-Doppler Map using Fluent Pipeline API

This example demonstrates the cleaner, more functional pipeline interface
using method chaining and lambda functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sigchain import SignalData, Pipeline
from sigchain.blocks import RadarGenerator, PulseStacker, MatchedFilter, DopplerProcessor


def main():
    """Run the radar signal processing using the fluent pipeline API."""
    
    print("=" * 70)
    print("Radar Range-Doppler Map Example - Fluent Pipeline API")
    print("=" * 70)
    
    # Configure radar parameters
    radar_config = {
        'num_pulses': 128,
        'pulse_duration': 10e-6,
        'pulse_repetition_interval': 1e-3,
        'sample_rate': 10e6,
        'bandwidth': 5e6,
        'carrier_freq': 10e9,
        'target_delay': 20e-6,
        'target_doppler': 1000.0,
        'noise_power': 0.1
    }
    
    print("\nBuilding pipeline with fluent API...")
    
    # Create pipeline using fluent interface
    pipeline = (Pipeline(name="RadarProcessing")
        # Generate radar signal
        .add_block(RadarGenerator(**radar_config))
        
        # Pulse stacking
        .add_block(PulseStacker())
        
        # Matched filtering (range compression)
        .add_block(MatchedFilter())
        
        # Doppler processing (velocity compression)
        .add_block(DopplerProcessor(window='hann'))
        
        # Add a tap for intermediate inspection
        .tap(lambda sig: print(f"  Final output shape: {sig.shape}"))
    )
    
    print(f"  {pipeline}")
    print(f"  Total operations: {len(pipeline)}")
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result = pipeline.run(verbose=True)
    
    # Display results
    print("\nDisplaying Range-Doppler Map...")
    display_range_doppler_map(result)
    
    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


def main_with_lambdas():
    """
    Alternative approach using lambda functions for custom operations.
    
    This demonstrates maximum flexibility with inline lambda functions.
    """
    
    print("=" * 70)
    print("Radar Processing with Lambda Functions")
    print("=" * 70)
    
    # Radar parameters
    num_pulses = 128
    pulse_duration = 10e-6
    sample_rate = 10e6
    bandwidth = 5e6
    target_delay = 20e-6
    target_doppler = 1000.0
    
    print("\nBuilding pipeline with lambda functions...")
    
    # Create pipeline with lambda operations
    pipeline = (Pipeline(name="LambdaRadar")
        # Generate LFM signal
        .add(lambda _: generate_lfm_signal(
            num_pulses, pulse_duration, sample_rate,
            bandwidth, target_delay, target_doppler
        ), name="GenerateSignal")
        
        # Apply matched filter using lambda
        .add(lambda sig: apply_matched_filtering(sig), name="MatchedFilter")
        
        # Apply Doppler FFT using lambda
        .add(lambda sig: apply_doppler_fft(sig), name="DopplerFFT")
        
        # Normalize to dB scale
        .transform(
            lambda data: 20 * np.log10(np.abs(data) + 1e-10),
            name="ConvertToDB"
        )
        
        # Print statistics
        .tap(lambda sig: print(f"  Max: {np.max(sig.data):.2f} dB, "
                               f"Min: {np.min(sig.data):.2f} dB"))
    )
    
    print(f"  {pipeline}")
    
    # Execute
    print("\nExecuting pipeline...")
    result = pipeline.run(verbose=True)
    
    print(f"\nFinal result shape: {result.shape}")


def generate_lfm_signal(num_pulses, pulse_duration, sample_rate,
                        bandwidth, target_delay, target_doppler):
    """Generate LFM signal using lambda-friendly function."""
    gen = RadarGenerator(
        num_pulses=num_pulses,
        pulse_duration=pulse_duration,
        sample_rate=sample_rate,
        bandwidth=bandwidth,
        target_delay=target_delay,
        target_doppler=target_doppler
    )
    return gen.process()


def apply_matched_filtering(signal_data):
    """Apply matched filtering."""
    mf = MatchedFilter()
    return mf.process(signal_data)


def apply_doppler_fft(signal_data):
    """Apply Doppler FFT."""
    dp = DopplerProcessor(window='hann')
    return dp.process(signal_data)


def display_range_doppler_map(signal_data: SignalData):
    """Display the Range-Doppler Map."""
    rdm = signal_data.data
    metadata = signal_data.metadata
    
    # Convert to magnitude (dB scale)
    rdm_magnitude = 20 * np.log10(np.abs(rdm) + 1e-10)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Range-Doppler Map
    ax1 = axes[0]
    
    num_doppler_bins, num_range_bins = rdm.shape
    range_axis = np.arange(num_range_bins) / signal_data.sample_rate * 3e8 / 2 / 1000  # km
    doppler_axis = metadata.get('doppler_frequencies', np.arange(num_doppler_bins))
    
    im1 = ax1.imshow(
        rdm_magnitude,
        aspect='auto',
        extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        origin='lower',
        cmap='jet',
        vmin=np.max(rdm_magnitude) - 60,
        vmax=np.max(rdm_magnitude)
    )
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Doppler Frequency (Hz)')
    ax1.set_title('Range-Doppler Map (Fluent Pipeline)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    if 'target_delay' in metadata and 'target_doppler' in metadata:
        target_range = metadata['target_delay'] * 3e8 / 2 / 1000
        target_doppler = metadata['target_doppler']
        ax1.plot(target_range, target_doppler, 'rx', markersize=15, 
                markeredgewidth=2, label='True Target')
        ax1.legend()
    
    # Plot 2: Range cut
    ax2 = axes[1]
    peak_idx = np.unravel_index(np.argmax(np.abs(rdm)), rdm.shape)
    peak_doppler_idx = peak_idx[0]
    
    range_cut = rdm_magnitude[peak_doppler_idx, :]
    ax2.plot(range_axis, range_cut, 'b-', linewidth=2)
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title(f'Range Profile at Peak Doppler ({doppler_axis[peak_doppler_idx]:.1f} Hz)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.max(range_cut) - 3, color='r', linestyle='--', 
                alpha=0.5, label='-3 dB')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('range_doppler_map_fluent.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: range_doppler_map_fluent.png")


if __name__ == "__main__":
    # Run the main fluent API example
    main()
    
    print("\n" + "=" * 70)
    print("\n")
    
    # Uncomment to see lambda-based example
    # main_with_lambdas()
