"""
Example: Radar Range-Doppler Map Generation

This example demonstrates the complete signal processing chain:
1. Generate LFM radar signal with target (delayed and Doppler-shifted)
2. Pulse stacking
3. Matched filtering (range compression)
4. FFT processing (Doppler compression)
5. Display Range-Doppler Map
"""

import numpy as np
import matplotlib.pyplot as plt
from sigchain import SignalData, DAG
from sigchain.blocks import RadarGenerator, PulseStacker, MatchedFilter, DopplerProcessor


def main():
    """Run the radar signal processing chain example."""
    
    print("=" * 60)
    print("Radar Range-Doppler Map Example")
    print("=" * 60)
    
    # Create processing blocks
    print("\n1. Creating processing blocks...")
    
    # Generate radar signal with a target at 3km range and 150 m/s velocity
    # Target delay: 20 microseconds (approximately 3 km range)
    # Target Doppler: 1 kHz (approximately 150 m/s at 10 GHz carrier)
    radar_gen = RadarGenerator(
        num_pulses=128,
        pulse_duration=10e-6,
        pulse_repetition_interval=1e-3,
        sample_rate=10e6,
        bandwidth=5e6,
        carrier_freq=10e9,
        target_delay=20e-6,
        target_doppler=1000.0,
        noise_power=0.1,
        name="RadarGenerator"
    )
    
    pulse_stacker = PulseStacker(name="PulseStacker")
    matched_filter = MatchedFilter(name="MatchedFilter")
    doppler_processor = DopplerProcessor(name="DopplerProcessor", window='hann')
    
    print(f"   - {radar_gen}")
    print(f"   - {pulse_stacker}")
    print(f"   - {matched_filter}")
    print(f"   - {doppler_processor}")
    
    # Create DAG and add blocks in sequence
    print("\n2. Building processing chain (DAG)...")
    dag = DAG()
    dag.add_chain(radar_gen, pulse_stacker, matched_filter, doppler_processor)
    print(f"   - {dag}")
    print(f"   - Chain: {' -> '.join([b.name for b in dag.blocks])}")
    
    # Execute the processing chain
    print("\n3. Generating radar signal...")
    signal_out = radar_gen.process()
    print(f"   - Generated signal shape: {signal_out.shape}")
    print(f"   - Data type: {signal_out.dtype}")
    
    print("\n4. Pulse stacking...")
    signal_out = pulse_stacker.process(signal_out)
    print(f"   - Stacked signal shape: {signal_out.shape}")
    
    print("\n5. Matched filtering (range compression)...")
    signal_out = matched_filter.process(signal_out)
    print(f"   - Range compressed signal shape: {signal_out.shape}")
    
    print("\n6. Doppler processing (FFT)...")
    signal_out = doppler_processor.process(signal_out)
    print(f"   - Range-Doppler map shape: {signal_out.shape}")
    
    # Display results
    print("\n7. Displaying Range-Doppler Map...")
    display_range_doppler_map(signal_out)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


def display_range_doppler_map(signal_data: SignalData):
    """
    Display the Range-Doppler Map.
    
    Args:
        signal_data: SignalData containing the range-doppler map
    """
    # Extract data
    rdm = signal_data.data
    metadata = signal_data.metadata
    
    # Convert to magnitude (dB scale)
    rdm_magnitude = 20 * np.log10(np.abs(rdm) + 1e-10)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Range-Doppler Map
    ax1 = axes[0]
    
    # Calculate axes
    num_doppler_bins, num_range_bins = rdm.shape
    range_axis = np.arange(num_range_bins) / signal_data.sample_rate * 3e8 / 2 / 1000  # km
    doppler_axis = metadata.get('doppler_frequencies', np.arange(num_doppler_bins))
    
    # Display as image
    im1 = ax1.imshow(
        rdm_magnitude,
        aspect='auto',
        extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        origin='lower',
        cmap='jet',
        vmin=np.max(rdm_magnitude) - 60,  # Dynamic range of 60 dB
        vmax=np.max(rdm_magnitude)
    )
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Doppler Frequency (Hz)')
    ax1.set_title('Range-Doppler Map')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    # Add target marker (if known)
    if 'target_delay' in metadata and 'target_doppler' in metadata:
        target_range = metadata['target_delay'] * 3e8 / 2 / 1000  # km
        target_doppler = metadata['target_doppler']
        ax1.plot(target_range, target_doppler, 'rx', markersize=15, 
                markeredgewidth=2, label='True Target')
        ax1.legend()
    
    # Plot 2: Range cut at peak Doppler
    ax2 = axes[1]
    
    # Find peak
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
    plt.savefig('range_doppler_map.png', dpi=150, bbox_inches='tight')
    print(f"   - Range-Doppler map saved to: range_doppler_map.png")
    print(f"   - Peak detected at: Range = {range_axis[peak_idx[1]]:.2f} km, "
          f"Doppler = {doppler_axis[peak_doppler_idx]:.1f} Hz")
    
    # Print statistics
    print(f"\n   Statistics:")
    print(f"   - Max magnitude: {np.max(rdm_magnitude):.2f} dB")
    print(f"   - Min magnitude: {np.min(rdm_magnitude):.2f} dB")
    print(f"   - Dynamic range: {np.max(rdm_magnitude) - np.min(rdm_magnitude):.2f} dB")


if __name__ == "__main__":
    main()
