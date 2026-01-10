"""
Comprehensive Radar Processing Demo with Plotly and Staticdash

This example demonstrates the complete radar signal processing pipeline
with interactive visualizations at each stage. Uses staticdash Directory
to create a multi-dashboard site.
"""

import numpy as np
import pandas as pd
from sigchain import Pipeline, SignalData
from sigchain.blocks import (
    LFMGenerator,
    StackPulses,
    RangeCompress,
    DopplerCompress,
)
from sigchain.visualization import (
    plot_timeseries,
    plot_pulse_matrix,
    plot_range_profile,
    plot_range_doppler_map,
    plot_spectrum,
)

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False
    print("Warning: staticdash not installed. Dashboard generation will be skipped.")


def create_radar_demo_dashboard(
    num_pulses: int = 128,
    target_delay: float = 2e-6,
    target_doppler: float = 200.0,
    noise_power: float = 0.01,
) -> sd.Dashboard:
    """
    Create a comprehensive radar processing demo dashboard.
    
    Args:
        num_pulses: Number of radar pulses
        target_delay: Target delay (seconds)
        target_doppler: Target Doppler frequency (Hz)
        noise_power: Noise power level
        
    Returns:
        Dashboard object ready to be added to a Directory
    """
    
    # Create dashboard
    dashboard = sd.Dashboard('Radar Signal Processing Pipeline')
    
    # Create page
    page = sd.Page('radar-demo', 'Radar Signal Processing Pipeline Demo')
    
    # Add title and introduction
    page.add_header("Radar Signal Processing Pipeline", level=1)
    page.add_text("""
    This demonstration shows the complete radar signal processing pipeline,
    from LFM waveform generation through to range-Doppler map creation.
    Each stage is visualized with interactive Plotly plots.
    """)
    
    # Configuration info - Create pandas DataFrame table
    page.add_header("Configuration", level=2)
    page.add_text("""
    Radar system parameters and target characteristics for this simulation:
    """)
    
    # Radar system parameters (should match LFMGenerator parameters)
    sample_rate_mhz = 10
    pulse_duration_us = 10
    bandwidth_mhz = 5
    pri_ms = 1
    
    # Calculate derived values
    c = 3e8  # Speed of light
    target_range_m = target_delay * c / 2
    pri = pri_ms * 1e-3  # Convert to seconds
    prf = 1 / pri
    max_doppler = prf / 2
    
    config_data = {
        'Parameter': [
            'Number of Pulses',
            'Sample Rate',
            'Pulse Duration',
            'Bandwidth',
            'Pulse Repetition Interval (PRI)',
            'Pulse Repetition Frequency (PRF)',
            'Max Unambiguous Doppler',
            'Target Delay',
            'Target Range',
            'Target Doppler',
            'Noise Power',
        ],
        'Value': [
            num_pulses,
            sample_rate_mhz,
            pulse_duration_us,
            bandwidth_mhz,
            pri_ms,
            f'{prf:.0f}',
            f'±{max_doppler:.0f}',
            f'{target_delay * 1e6:.2f}',
            f'{target_range_m:.0f}',
            f'{target_doppler:.1f}',
            f'{noise_power:.3f}',
        ],
        'Unit': [
            'pulses',
            'MHz',
            'μs',
            'MHz',
            'ms',
            'Hz',
            'Hz',
            'μs',
            'm',
            'Hz',
            'relative',
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    page.add_table(config_df)
    
    # Stage 1: Generate LFM Signal
    page.add_header("Stage 1: LFM Signal Generation", level=2)
    page.add_text("""
    Generate Linear Frequency Modulated (LFM) chirp pulses with simulated
    target return (delayed and Doppler shifted) plus noise.
    """)
    
    gen = LFMGenerator(
        num_pulses=num_pulses,
        pulse_duration=pulse_duration_us * 1e-6,
        pulse_repetition_interval=pri_ms * 1e-3,
        sample_rate=sample_rate_mhz * 1e6,
        bandwidth=bandwidth_mhz * 1e6,
        target_delay=target_delay,
        target_doppler=target_doppler,
        noise_power=noise_power,
    )
    
    signal_generated = gen()
    
    # Plot generated waveform (single pulse)
    fig1 = plot_timeseries(
        signal_generated,
        title="Generated LFM Pulse (First Pulse)",
        show_real=True,
        show_imag=True,
        show_magnitude=True,
        height=400,
    )
    page.add_plot(fig1, height=400)
    
    page.add_text("""
    The plot above shows one complete pulse. The blue/red lines show the
    real and imaginary components of the complex signal, while the green
    line shows the magnitude envelope.
    """)
    
    # Plot spectrum of reference pulse
    reference_pulse = signal_generated.metadata['reference_pulse']
    ref_signal = SignalData(
        data=reference_pulse,
        sample_rate=signal_generated.sample_rate,
        metadata={}
    )
    fig2 = plot_spectrum(ref_signal, title="Reference Pulse Spectrum", height=350)
    page.add_plot(fig2, height=350)
    
    page.add_text("""
    The frequency spectrum shows the 5 MHz bandwidth of the LFM chirp.
    This wideband signal provides good range resolution after matched filtering.
    """)
    
    # Stage 2: Stack Pulses
    page.add_header("Stage 2: Pulse Stacking", level=2)
    page.add_text("""
    Organize the pulses into a 2D matrix (pulses × samples) for coherent processing.
    This allows us to process range and Doppler dimensions separately.
    """)
    
    stack = StackPulses()
    signal_stacked = stack(signal_generated)
    
    # Visualize stacked pulses
    fig3 = plot_pulse_matrix(
        signal_stacked,
        title=f"Stacked Pulses ({num_pulses} pulses × {signal_stacked.shape[1]} samples)",
        colorscale="Viridis",
        height=500,
        use_db=True,
    )
    page.add_plot(fig3, height=500)
    
    page.add_text(f"""
    The heatmap shows all {num_pulses} pulses stacked vertically. You can see:
    - The delayed target return (vertical bright region around sample {int(target_delay * signal_generated.sample_rate)})
    - Background noise across the entire matrix
    - Each horizontal line represents one pulse
    """)
    
    # Stage 3: Range Compression (Matched Filtering)
    page.add_header("Stage 3: Range Compression", level=2)
    page.add_text("""
    Apply matched filtering using the transmitted waveform as the filter.
    This correlates the received signal with the known transmitted pulse,
    compressing it in time and improving SNR.
    """)
    
    range_comp = RangeCompress()
    signal_range_compressed = range_comp(signal_stacked)
    
    # Plot range-compressed pulse matrix
    fig4 = plot_pulse_matrix(
        signal_range_compressed,
        title="Range-Compressed Pulses",
        colorscale="Hot",
        height=500,
        use_db=True,
    )
    page.add_plot(fig4, height=500)
    
    page.add_text("""
    After matched filtering, the target appears as a sharp peak in the range
    dimension. The SNR is significantly improved compared to the raw data.
    """)
    
    # Plot single range profile
    fig5 = plot_range_profile(
        signal_range_compressed,
        title="Range Profile (Averaged over all pulses)",
        pulse_index=None,  # Average
        use_db=True,
        height=400,
    )
    page.add_plot(fig5, height=400)
    
    page.add_text("""
    The range profile shows the target as a clear peak at the expected range.
    The red dashed line marks the true target location.
    """)
    
    # Stage 4: Doppler Compression (FFT)
    page.add_header("Stage 4: Doppler Compression", level=2)
    page.add_text("""
    Apply FFT along the pulse dimension to resolve Doppler frequency.
    This creates the final Range-Doppler Map (RDM) that shows both
    target range and velocity.
    """)
    
    doppler_comp = DopplerCompress(window='hann')
    signal_rdm = doppler_comp(signal_range_compressed)
    
    # Plot final Range-Doppler Map
    fig6 = plot_range_doppler_map(
        signal_rdm,
        title="Range-Doppler Map",
        colorscale="Jet",
        height=600,
        use_db=True,
        db_range=50,
        mark_target=True,
    )
    page.add_plot(fig6, height=600)
    
    page.add_text("""
    **This is the final result!** The Range-Doppler Map shows:
    - **X-axis**: Target range (distance from radar)
    - **Y-axis**: Doppler frequency (related to target velocity)
    - **Color**: Signal strength in dB
    - **Red X**: True target location
    
    The bright spot near the red X mark is our detected target.
    The background shows the noise floor and any sidelobes from the processing.
    """)
    
    # Summary statistics
    page.add_header("Processing Summary", level=2)
    
    # Calculate some statistics
    rdm_data = np.abs(signal_rdm.data)
    peak_idx = np.unravel_index(np.argmax(rdm_data), rdm_data.shape)
    peak_doppler_idx, peak_range_idx = peak_idx
    
    if 'doppler_frequencies' in signal_rdm.metadata:
        detected_doppler = signal_rdm.metadata['doppler_frequencies'][peak_doppler_idx]
    else:
        detected_doppler = peak_doppler_idx - num_pulses // 2
    
    c = 3e8
    detected_range = peak_range_idx / signal_generated.sample_rate * c / 2 / 1000
    true_range = target_delay * c / 2 / 1000
    
    snr_db = 20 * np.log10(rdm_data[peak_idx] / np.median(rdm_data))
    
    page.add_text(f"""
    **Detection Results:**
    - **True Target**: Range = {true_range:.3f} km, Doppler = {target_doppler:.1f} Hz
    - **Detected Peak**: Range = {detected_range:.3f} km, Doppler = {detected_doppler:.1f} Hz
    - **Range Error**: {abs(detected_range - true_range)*1000:.1f} m
    - **Doppler Error**: {abs(detected_doppler - target_doppler):.1f} Hz
    - **Estimated SNR**: {snr_db:.1f} dB
    
    **Processing Stages Completed:**
    1. ✓ LFM Signal Generation ({num_pulses} pulses)
    2. ✓ Pulse Stacking (2D matrix organization)
    3. ✓ Range Compression (Matched filtering)
    4. ✓ Doppler Compression (FFT processing)
    """)
    
    # Add code example
    page.add_header("Code Example", level=2)
    page.add_text("""
    Here's how to reproduce this processing pipeline using sigchain:
    """)
    
    code_example = f"""
from sigchain import Pipeline
from sigchain.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress

# Create processing pipeline
result = (Pipeline("Radar")
    .add(LFMGenerator(
        num_pulses={num_pulses},
        target_delay={target_delay},
        target_doppler={target_doppler},
    ))
    .add(StackPulses())
    .add(RangeCompress())
    .add(DopplerCompress(window='hann'))
    .run()
)

# Result contains the Range-Doppler Map
rdm = result.data
"""
    page.add_syntax(code_example, language='python')
    
    # Add page to dashboard and publish
    dashboard.add_page(page)
    
    return dashboard


def run_basic_demo():
    """Run a basic demo without staticdash (just show processing works)."""
    
    print("\n" + "="*70)
    print("Radar Processing Demo (Basic Mode)")
    print("="*70 + "\n")
    
    # Create pipeline
    pipeline = Pipeline("RadarDemo")
    pipeline.add(LFMGenerator(num_pulses=64, target_delay=15e-6, target_doppler=500.0), name="Generate")
    pipeline.add(StackPulses(), name="Stack")
    pipeline.add(RangeCompress(), name="RangeCompress")
    pipeline.add(DopplerCompress(window='hann'), name="DopplerCompress")
    
    # Run with verbose output
    result = pipeline.run(verbose=True)
    
    print(f"\nFinal Result:")
    print(f"  Shape: {result.shape}")
    print(f"  Type: {result.dtype}")
    print(f"  Max value: {np.max(np.abs(result.data)):.4f}")
    print(f"  Metadata keys: {list(result.metadata.keys())[:5]}...")
    
    # Try to create plots anyway (will display in notebook or save to files)
    try:
        fig1 = plot_range_doppler_map(result, title="Range-Doppler Map")
        print(f"\nRange-Doppler map created. To display:")
        print(f"  fig1.show()  # In notebook")
        print(f"  fig1.write_html('rdm.html')  # Save to file")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    print("\n" + "="*70)


def create_dashboard_with_custom_blocks() -> sd.Dashboard:
    """
    Example showing how to add custom processing stages to the dashboard.
    
    This demonstrates the pattern of passing the page object and adding
    plots and text at each stage.
    
    Returns:
        Dashboard object ready to be added to a Directory
    """
    
    dashboard = sd.Dashboard('Custom Processing Tutorial')
    page = sd.Page('custom-demo', 'Custom Processing Demo')
    
    page.add_header("Custom Processing Pipeline", level=1)
    page.add_text("This demonstrates adding custom stages to the processing pipeline.")
    
    # Stage 1
    page.add_header("Stage 1: Generate Signal", level=2)
    gen = LFMGenerator(num_pulses=32)
    signal = gen()
    
    fig = plot_timeseries(signal, title="Generated Signal")
    page.add_plot(fig)
    page.add_text("Signal generated successfully.")
    
    # Stage 2 - Custom processing
    page.add_header("Stage 2: Custom Processing", level=2)
    page.add_text("Apply custom threshold...")
    
    # Custom processing
    data = signal.data.copy()
    threshold = 0.1 * np.max(np.abs(data))
    data[np.abs(data) < threshold] = 0
    
    signal_processed = SignalData(data, signal.sample_rate, signal.metadata.copy())
    
    fig2 = plot_timeseries(signal_processed, title="After Thresholding")
    page.add_plot(fig2)
    page.add_text(f"Applied threshold: {threshold:.4f}")
    
    dashboard.add_page(page)
    return dashboard


if __name__ == "__main__":
    if not STATICDASH_AVAILABLE:
        print("staticdash not available. Running basic demo instead.")
        run_basic_demo()
    else:
        # Create directory to hold multiple dashboards
        directory = sd.Directory(
            title='SigChain Interactive Demos',
            page_width=1000
        )
        
        # Create and add radar processing dashboard
        print("Creating radar processing dashboard...")
        radar_dashboard = create_radar_demo_dashboard(
            num_pulses=128,
            target_delay=3e-6,
            target_doppler=200.0,
            noise_power=0.01,
        )
        directory.add_dashboard(radar_dashboard, slug='radar-processing')
        
        # Create and add custom blocks tutorial dashboard
        print("Creating custom blocks tutorial...")
        custom_dashboard = create_dashboard_with_custom_blocks()
        directory.add_dashboard(custom_dashboard, slug='custom-blocks-tutorial')
        
        # Publish everything to docs/ directory
        print("Publishing dashboards...")
        directory.publish('docs')
        
        print(f"\n{'='*70}")
        print(f"✓ Dashboards created successfully!")
        print(f"{'='*70}")
        print(f"\nGenerated files:")
        print(f"  docs/index.html           - Landing page with all dashboards")
        print(f"  docs/radar-processing/    - Complete radar processing demo")
        print(f"  docs/custom-blocks-tutorial/ - Custom blocks tutorial")
        print(f"\nTo view: Open docs/index.html in a web browser")
        print(f"{'='*70}\n")
