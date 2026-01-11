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
from sigchain.diagnostics import (
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
    target_delay: float = 1e-6,
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
    
    # Add code example
    page.add_header("Code Example", level=2)
    page.add_text("""
    Here's how to reproduce this processing pipeline using sigchain:
    """)
    
    code_example = f"""
import staticdash as sd
from sigchain import Pipeline
from sigchain.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigchain.diagnostics import plot_timeseries, plot_pulse_matrix, plot_range_profile, plot_range_doppler_map

page = sd.Page('radar', 'Radar Processing')
page.add_header("Radar Signal Processing Pipeline", level=1)

result = (Pipeline("Radar")
    .add(LFMGenerator(num_pulses={num_pulses}, target_delay={target_delay}, 
                       target_doppler={target_doppler}))
    .tap(lambda s: page.add_header("Stage 1: Signal Generation", level=2))
    .tap(lambda s: page.add_plot(plot_timeseries(s, title="Generated LFM Pulse")))
    
    .add(StackPulses())
    .tap(lambda s: page.add_header("Stage 2: Pulse Stacking", level=2))
    .tap(lambda s: page.add_plot(plot_pulse_matrix(s, title="Stacked Pulses")))
    
    .add(RangeCompress(window='hamming', oversample_factor=2))
    .tap(lambda s: page.add_header("Stage 3: Range Compression", level=2))
    .tap(lambda s: page.add_plot(plot_range_profile(s, title="Range Profile")))
    
    .add(DopplerCompress(window='hann', oversample_factor=2))
    .tap(lambda s: page.add_header("Stage 4: Doppler Compression", level=2))
    .tap(lambda s: page.add_plot(plot_range_doppler_map(s, title="Range-Doppler Map")))
    .run()
)
"""
    page.add_syntax(code_example, language='python')
    
    # Build the pipeline with inline plotting
    page.add_header("Processing Pipeline", level=2)
    
    signal_rdm = (Pipeline("Radar")
        .add(LFMGenerator(
            num_pulses=num_pulses,
            pulse_duration=pulse_duration_us * 1e-6,
            pulse_repetition_interval=pri_ms * 1e-3,
            sample_rate=sample_rate_mhz * 1e6,
            bandwidth=bandwidth_mhz * 1e6,
            target_delay=target_delay,
            target_doppler=target_doppler,
            noise_power=noise_power,
        ), name="Generate")
        .tap(lambda s: page.add_header("Stage 1: LFM Signal Generation", level=2))
        .tap(lambda s: page.add_text("""
            Generate Linear Frequency Modulated (LFM) chirp pulses with simulated
            target return (delayed and Doppler shifted) plus noise.
        """))
        .tap(lambda s: page.add_plot(plot_timeseries(s, title="Generated LFM Pulse", 
              show_real=True, show_imag=True, show_magnitude=True, height=400), height=400))
        
        .add(StackPulses(), name="Stack")
        .tap(lambda s: page.add_header("Stage 2: Pulse Stacking", level=2))
        .tap(lambda s: page.add_text("""
            Organize the pulses into a 2D matrix (pulses × samples) for coherent processing.
        """))
        .tap(lambda s: page.add_plot(plot_pulse_matrix(s, title=f"Stacked Pulses ({num_pulses} pulses)", 
              colorscale="Greys", height=500, use_db=True), height=500))
        
        .add(RangeCompress(window='hamming', oversample_factor=2), name="RangeCompress")
        .tap(lambda s: page.add_header("Stage 3: Range Compression", level=2))
        .tap(lambda s: page.add_text("""
            Apply matched filtering using the transmitted waveform. A Hamming window 
            is applied to reduce sidelobe levels.
        """))
        .tap(lambda s: page.add_plot(plot_pulse_matrix(s, title="Range-Compressed Pulses", 
              colorscale="Greys", height=500, use_db=True), height=500))
        .tap(lambda s: page.add_plot(plot_range_profile(s, title="Range Profile (Averaged)", 
              pulse_index=None, use_db=True, height=400), height=400))
        
        .add(DopplerCompress(window='hann', oversample_factor=2), name="DopplerCompress")
        .tap(lambda s: page.add_header("Stage 4: Doppler Compression", level=2))
        .tap(lambda s: page.add_text("""
            Apply FFT along the pulse dimension to resolve Doppler frequency.
            This creates the final Range-Doppler Map (RDM).
        """))
        .tap(lambda s: page.add_plot(plot_range_doppler_map(s, title="Range-Doppler Map", 
              colorscale="Greys", height=600, use_db=True, db_range=50, mark_target=True), height=600))
        .run()
    )
    
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
    detected_range = peak_range_idx / signal_rdm.sample_rate * c / 2 / 1000
    true_range = target_delay * c / 2 / 1000
    
    snr_db = 20 * np.log10(rdm_data[peak_idx] / np.median(rdm_data))
    
    # Create results table
    results_data = {
        'Metric': [
            'True Target Range',
            'Detected Peak Range',
            'Range Error',
            'True Target Doppler',
            'Detected Peak Doppler',
            'Doppler Error',
            'Estimated SNR',
        ],
        'Value': [
            f'{true_range:.3f}',
            f'{detected_range:.3f}',
            f'{abs(detected_range - true_range)*1000:.1f}',
            f'{target_doppler:.1f}',
            f'{detected_doppler:.1f}',
            f'{abs(detected_doppler - target_doppler):.1f}',
            f'{snr_db:.1f}',
        ],
        'Unit': [
            'km',
            'km',
            'm',
            'Hz',
            'Hz',
            'Hz',
            'dB',
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    page.add_table(results_df)
    
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
    Example showing how to create custom processing blocks inline and compose them.
    
    This demonstrates defining custom processing logic as simple functions
    and integrating them into the pipeline with visualization.
    
    Returns:
        Dashboard object ready to be added to a Directory
    """
    
    dashboard = sd.Dashboard('Custom Processing Tutorial')
    page = sd.Page('custom-demo', 'Custom Processing Demo')
    
    page.add_header("Custom Processing Pipeline", level=1)
    page.add_text("""
    This example demonstrates how to create custom processing blocks inline
    and compose them with built-in blocks in a processing pipeline.
    """)
    
    # Add code example
    page.add_header("Code Example", level=2)
    code_example = """
from sigchain import Pipeline, SignalData
from sigchain.blocks import LFMGenerator
from sigchain.diagnostics import plot_timeseries
import numpy as np

# Define custom processing function inline
def apply_threshold(signal_data: SignalData, threshold_factor=0.1) -> SignalData:
    data = signal_data.data.copy()
    threshold = threshold_factor * np.max(np.abs(data))
    data[np.abs(data) < threshold] = 0
    return SignalData(data, signal_data.sample_rate, signal_data.metadata)

# Define another custom block - normalize signal
def normalize_signal(signal_data: SignalData) -> SignalData:
    data = signal_data.data.copy()
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return SignalData(data, signal_data.sample_rate, signal_data.metadata)

# Compose pipeline with custom blocks
page = sd.Page('demo', 'Demo')
result = (Pipeline("CustomDemo")
    .add(LFMGenerator(num_pulses=32))
    .tap(lambda s: page.add_plot(plot_timeseries(s, title="Original Signal")))
    
    .add(lambda s: apply_threshold(s, threshold_factor=0.1))
    .tap(lambda s: page.add_plot(plot_timeseries(s, title="After Thresholding")))
    
    .add(normalize_signal)
    .tap(lambda s: page.add_plot(plot_timeseries(s, title="After Normalization")))
    .run()
)
"""
    page.add_syntax(code_example, language='python')
    
    # Define custom processing functions inline
    def apply_threshold(signal_data: SignalData, threshold_factor=0.1) -> SignalData:
        """Remove values below a threshold."""
        data = signal_data.data.copy()
        threshold = threshold_factor * np.max(np.abs(data))
        data[np.abs(data) < threshold] = 0
        metadata = signal_data.metadata.copy()
        metadata['threshold_applied'] = threshold
        return SignalData(data, signal_data.sample_rate, metadata)
    
    def normalize_signal(signal_data: SignalData) -> SignalData:
        """Normalize signal to max amplitude of 1."""
        data = signal_data.data.copy()
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        metadata = signal_data.metadata.copy()
        metadata['normalized'] = True
        metadata['original_max'] = max_val
        return SignalData(data, signal_data.sample_rate, metadata)
    
    # Build pipeline with custom blocks
    page.add_header("Pipeline Execution", level=2)
    
    result = (Pipeline("CustomDemo")
        .add(LFMGenerator(num_pulses=32, target_delay=5e-6, noise_power=0.1), name="Generate")
        .tap(lambda s: page.add_header("Stage 1: Generated Signal", level=2))
        .tap(lambda s: page.add_plot(plot_timeseries(s, title="Original LFM Signal", 
              show_magnitude=True, height=400), height=400))
        
        .add(lambda s: apply_threshold(s, threshold_factor=0.15), name="Threshold")
        .tap(lambda s: page.add_header("Stage 2: After Thresholding", level=2))
        .tap(lambda s: page.add_text(f"Applied threshold: {s.metadata.get('threshold_applied', 0):.4f}"))
        .tap(lambda s: page.add_plot(plot_timeseries(s, title="Signal After Thresholding", 
              show_magnitude=True, height=400), height=400))
        
        .add(normalize_signal, name="Normalize")
        .tap(lambda s: page.add_header("Stage 3: After Normalization", level=2))
        .tap(lambda s: page.add_text(f"Normalized to max amplitude 1.0 (original max: {s.metadata.get('original_max', 0):.4f})"))
        .tap(lambda s: page.add_plot(plot_timeseries(s, title="Normalized Signal", 
              show_magnitude=True, height=400), height=400))
        .run()
    )
    
    page.add_header("Summary", level=2)
    page.add_text("""
    This example showed how to:
    1. Define custom processing functions inline (not in the blocks directory)
    2. Use lambda functions to pass parameters to custom blocks
    3. Compose custom blocks with built-in blocks in a pipeline
    4. Add visualizations at each stage with `.tap()`
    """)
    
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
            target_delay=2e-6,
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
