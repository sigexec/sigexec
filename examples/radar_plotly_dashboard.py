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


def create_parameter_exploration_dashboard() -> sd.Dashboard:
    """
    Example showing chained variants to explore cartesian product.
    
    Chaining .variants() automatically explores all combinations, so this example
    tries 3 range windows × 2 Doppler windows = 6 total combinations.
    
    Returns:
        Dashboard object ready to be added to a Directory
    """
    
    dashboard = sd.Dashboard('Parameter Exploration')
    page = sd.Page('param-exploration', 'Parameter Exploration Demo')
    
    page.add_header("Chained Variants Parameter Exploration", level=1)
    page.add_text("""
    This example demonstrates exploring the parameter space by chaining .variants() calls.
    Each .variants() adds a dimension, and .run() automatically explores all combinations.
    """)
    
    # Add code example
    page.add_header("Code Example", level=2)
    code_example = """
from sigchain import Pipeline
from sigchain.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress

# Build pipeline and chain variants
results = (Pipeline("Radar")
    .add(LFMGenerator(num_pulses=64, target_delay=2e-6, target_doppler=200.0))
    .add(StackPulses())
    .variants(lambda w: RangeCompress(window=w, oversample_factor=2), 
              ['hamming', 'hann', 'blackman'],
              names=['Hamming', 'Hann', 'Blackman'])
    .variants(lambda w: DopplerCompress(window=w, oversample_factor=2), 
              ['hamming', 'hann'],
              names=['Hamming', 'Hann'])
    .run()
)

# Results is a list of (params_dict, result_data) tuples
# 3 range windows × 2 doppler windows = 6 total combinations
for params, result in results:
    # Access variants as a list: params['variant'][0], params['variant'][1], etc.
    print(f"Range: {params['variant'][0]}, Doppler: {params['variant'][1]}")
    print(f"  Peak SNR: {calculate_snr(result):.1f} dB")
"""
    page.add_syntax(code_example, language='python')
    
    # Build pipeline with chained variants
    page.add_header("Parameter Sweep Results", level=2)
    page.add_text("""
    Testing all combinations of window functions:
    - Range Compression: hamming, hann, blackman
    - Doppler Compression: hamming, hann
    
    Total combinations: 3 × 2 = 6
    """)
    
    results = (Pipeline("RadarBase")
        .add(LFMGenerator(
            num_pulses=64,
            target_delay=2e-6,
            target_doppler=200.0,
            noise_power=0.01,
        ))
        .add(StackPulses())
        .variants(lambda w: RangeCompress(window=w, oversample_factor=2), 
                 ['hamming', 'hann', 'blackman'],
                 names=['Hamming', 'Hann', 'Blackman'])
        .variants(lambda w: DopplerCompress(window=w, oversample_factor=2), 
                 ['hamming', 'hann'],
                 names=['Hamming', 'Hann'])
        .run()
    )
    
    # Create comparison table
    comparison_data = []
    for params, result in results:
        rdm_data = np.abs(result.data)
        peak_idx = np.unravel_index(np.argmax(rdm_data), rdm_data.shape)
        snr_db = 20 * np.log10(rdm_data[peak_idx] / np.median(rdm_data))
        
        comparison_data.append({
            'Range Window': params['variant'][0],
            'Doppler Window': params['variant'][1],
            'Peak SNR (dB)': f'{snr_db:.1f}',
            'Peak Location': f"{peak_idx}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    page.add_table(comparison_df)
    
    # Plot each result
    for params, result in results:
        title = f"Range: {params['variant'][0]}, Doppler: {params['variant'][1]}"
        page.add_header(title, level=2)
        fig = plot_range_doppler_map(result, title=title, colorscale="Greys", 
                                     height=500, use_db=True, db_range=50, mark_target=True)
        page.add_plot(fig, height=500)
    
    page.add_header("Summary", level=2)
    page.add_text("""
    Chaining .variants() allows you to:
    1. Automatically try all combinations of parameters
    2. Compare results systematically  
    3. Find optimal parameter settings
    4. Understand parameter interactions
    
    The pipeline uses memoization, so common stages (like signal generation and stacking)
    only execute once and are reused across all variants.
    """)
    
    dashboard.add_page(page)
    return dashboard


def create_post_processing_plots_dashboard() -> sd.Dashboard:
    """
    Demo: Plotting after pipeline execution.
    
    Shows how to save intermediate results and plot them after the fact,
    and how to selectively plot variant results.
    """
    dashboard = sd.Dashboard('Post-Processing Plots')
    page = sd.Page('post-plots', 'Plotting After Execution')
    
    page.add_header("Plotting After Pipeline Execution", level=1)
    page.add_text("""
    All plotting functions are pure functions that work on SignalData objects.
    This means you can plot whenever you want - during pipeline execution with .tap(),
    or after the fact with saved results.
    """)
    
    # Demo 1: Intermediate results
    page.add_header("Method 1: Save and Plot Intermediate Results", level=2)
    page.add_text("""
    Use `save_intermediate=True` to capture the output of every pipeline stage.
    Then use `get_intermediate_results()` to retrieve them and plot any stage you want.
    """)
    
    code_example_1 = """
from sigchain import Pipeline
from sigchain.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigchain.diagnostics.visualization import (
    plot_timeseries, plot_pulse_matrix, 
    plot_range_profile, plot_range_doppler_map
)

# Build and run pipeline, saving all intermediate results
pipeline = (Pipeline("Radar")
    .add(LFMGenerator(num_pulses=64, target_delay=2e-6, target_doppler=200.0))
    .add(StackPulses())
    .add(RangeCompress(window='hamming'))
    .add(DopplerCompress(window='hann'))
)

# Run with intermediate results saved
result = pipeline.run(save_intermediate=True)
intermediates = pipeline.get_intermediate_results()

# Now plot any stage you want!
fig1 = plot_timeseries(intermediates[0], title="Stage 1: Generated Signal")
fig2 = plot_pulse_matrix(intermediates[1], title="Stage 2: Stacked Pulses")
fig3 = plot_range_profile(intermediates[2], title="Stage 3: After Range Compression")
fig4 = plot_range_doppler_map(intermediates[3], title="Stage 4: Final Range-Doppler Map")

# Or plot specific stages based on what you're debugging
if need_to_check_range_compression:
    fig = plot_range_profile(intermediates[2], title="Range Compression Check")
    page.add_plot(fig)
"""
    page.add_syntax(code_example_1, language='python')
    
    # Actually run this demo
    page.add_header("Live Example: Intermediate Results", level=3)
    
    pipeline = (Pipeline("IntermediateDemo")
        .add(LFMGenerator(
            num_pulses=32,
            target_delay=2e-6,
            target_doppler=200.0,
            noise_power=0.01,
        ))
        .add(StackPulses())
        .add(RangeCompress(window='hamming', oversample_factor=2))
        .add(DopplerCompress(window='hann', oversample_factor=2))
    )
    
    result = pipeline.run(save_intermediate=True)
    intermediates = pipeline.get_intermediate_results()
    
    # Plot selected stages
    fig1 = plot_pulse_matrix(intermediates[1], title="After Stacking", height=400)
    page.add_plot(fig1, height=400)
    
    fig2 = plot_range_profile(intermediates[2], title="After Range Compression", height=400)
    page.add_plot(fig2, height=400)
    
    fig3 = plot_range_doppler_map(intermediates[3], title="Final Result", 
                                   height=500, use_db=True, mark_target=True)
    page.add_plot(fig3, height=500)
    
    # Demo 2: Filtering variant results
    page.add_header("Method 2: Filter and Plot Variant Results", level=2)
    page.add_text("""
    When exploring parameter spaces with variants, you can selectively plot
    only the combinations you're interested in after running all of them.
    """)
    
    code_example_2 = """
from sigchain import Pipeline
from sigchain.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigchain.diagnostics.visualization import plot_range_doppler_map

# Run all variant combinations
results = (Pipeline("Radar")
    .add(LFMGenerator(num_pulses=64, target_delay=2e-6, target_doppler=200.0))
    .add(StackPulses())
    .variants(lambda w: RangeCompress(window=w), 
              ['hamming', 'hann', 'blackman'],
              names=['Hamming', 'Hann', 'Blackman'])
    .variants(lambda w: DopplerCompress(window=w), 
              ['hamming', 'hann'],
              names=['Hamming', 'Hann'])
    .run()
)

# Now selectively plot what you want:

# Example 1: Plot only Hamming range window results
for params, result in results:
    if params['variant'][0] == 'Hamming':
        title = f"Range: {params['variant'][0]}, Doppler: {params['variant'][1]}"
        fig = plot_range_doppler_map(result, title=title)
        page.add_plot(fig)

# Example 2: Compare two specific combinations
combo1 = next((r for p, r in results if p['variant'] == ['Hamming', 'Hamming']), None)
combo2 = next((r for p, r in results if p['variant'] == ['Blackman', 'Hann']), None)

fig1 = plot_range_doppler_map(combo1, title="Hamming + Hamming")
fig2 = plot_range_doppler_map(combo2, title="Blackman + Hann")

# Example 3: Find and plot best performing combination
best_params, best_result = max(results, 
                                key=lambda x: np.max(np.abs(x[1].data)))
title = f"Best: {' + '.join(best_params['variant'])}"
fig = plot_range_doppler_map(best_result, title=title)
"""
    page.add_syntax(code_example_2, language='python')
    
    # Actually run filtered variant demo
    page.add_header("Live Example: Filtered Variant Results", level=3)
    page.add_text("Running 2×2 variant combinations and plotting only selected results:")
    
    results = (Pipeline("FilteredDemo")
        .add(LFMGenerator(
            num_pulses=32,
            target_delay=2e-6,
            target_doppler=200.0,
            noise_power=0.01,
        ))
        .add(StackPulses())
        .variants(lambda w: RangeCompress(window=w, oversample_factor=2), 
                 ['hamming', 'blackman'],
                 names=['Hamming', 'Blackman'])
        .variants(lambda w: DopplerCompress(window=w, oversample_factor=2), 
                 ['hamming', 'hann'],
                 names=['Hamming', 'Hann'])
        .run()
    )
    
    # Only plot Hamming range window results
    page.add_header("Filtered Results: Only Hamming Range Window", level=4)
    for params, result in results:
        if params['variant'][0] == 'Hamming':
            title = f"Range: {params['variant'][0]}, Doppler: {params['variant'][1]}"
            fig = plot_range_doppler_map(result, title=title, height=450, 
                                         use_db=True, mark_target=True)
            page.add_plot(fig, height=450)
    
    # Find best combination
    page.add_header("Best Performing Combination", level=4)
    best_params, best_result = max(results, 
                                    key=lambda x: np.max(np.abs(x[1].data)))
    title = f"Best SNR: {' + '.join(best_params['variant'])}"
    fig = plot_range_doppler_map(best_result, title=title, height=450,
                                 use_db=True, mark_target=True)
    page.add_plot(fig, height=450)
    
    page.add_header("Summary", level=2)
    page.add_text("""
    Key takeaways:
    
    1. **Intermediate Results**: Use `save_intermediate=True` and `get_intermediate_results()` 
       to capture and plot any stage of your pipeline after execution.
       
    2. **Variant Filtering**: Run all variants, then selectively plot or analyze only the 
       combinations you're interested in.
       
    3. **Pure Functions**: All plot functions work on SignalData objects, so you have complete 
       flexibility in when and how you visualize your results.
       
    4. **Interactive Exploration**: You can load results, explore them, and create new 
       visualizations without re-running expensive processing steps.
    """)
    
    dashboard.add_page(page)
    return dashboard


def create_input_variants_dashboard() -> sd.Dashboard:
    """
    Demo: Processing multiple signals through the same pipeline with input_variants().
    
    Shows how to run the same processing pipeline over different input signals,
    and how to combine input variants with processing variants.
    """
    dashboard = sd.Dashboard('Input Variants')
    page = sd.Page('input-variants', 'Input Variants Demo')
    
    page.add_header("Processing Multiple Signals with input_variants()", level=1)
    page.add_text("""
    When you have multiple signals (from different files, sensors, or test scenarios) that need
    the same processing, use `.input_variants()` to run them all through the same pipeline.
    """)
    
    # Demo 1: Basic input variants
    page.add_header("Method 1: Process Multiple Input Signals", level=2)
    
    code_example_1 = """
from sigchain import Pipeline
from sigchain.blocks import RangeCompress, DopplerCompress
from sigchain.core.data import SignalData

# Load or generate different signals
signal_dataset_a = load_signal("dataset_a.bin")
signal_dataset_b = load_signal("dataset_b.bin")  
signal_dataset_c = load_signal("dataset_c.bin")

# Process all three through the same pipeline
results = (Pipeline()
    .input_variants([signal_dataset_a, signal_dataset_b, signal_dataset_c],
                   names=['Dataset A', 'Dataset B', 'Dataset C'])
    .add(RangeCompress(window='hamming'))
    .add(DopplerCompress(window='hann'))
    .run()
)

# Results contains one (params, result) tuple for each input signal
for params, result in results:
    dataset_name = params['variant'][0]
    print(f"{dataset_name}: peak at {find_peak(result)}")
"""
    page.add_syntax(code_example_1, language='python')
    
    # Live example 1
    page.add_header("Live Example: Three Different Target Scenarios", level=3)
    page.add_text("""
    Processing three radar scenarios with different target parameters through 
    the same range/Doppler compression pipeline.
    """)
    
    # Create three signals with different target parameters
    signal1 = LFMGenerator(
        num_pulses=32, target_delay=2e-6, target_doppler=150.0, noise_power=0.01
    )(None)
    signal2 = LFMGenerator(
        num_pulses=32, target_delay=3e-6, target_doppler=250.0, noise_power=0.01
    )(None)
    signal3 = LFMGenerator(
        num_pulses=32, target_delay=4e-6, target_doppler=-100.0, noise_power=0.01
    )(None)
    
    # Process all three
    results = (Pipeline()
        .input_variants([signal1, signal2, signal3],
                       names=['Near/Slow', 'Mid/Fast', 'Far/Receding'])
        .add(StackPulses())
        .add(RangeCompress(window='hamming', oversample_factor=2))
        .add(DopplerCompress(window='hann', oversample_factor=2))
        .run()
    )
    
    # Plot each result
    for params, result in results:
        scenario_name = params['variant'][0]
        fig = plot_range_doppler_map(result, title=scenario_name, height=450,
                                     use_db=True, mark_target=True)
        page.add_plot(fig, height=450)
    
    # Demo 2: Lazy loading from files with variants
    page.add_header("Method 2: Lazy Loading with Variants", level=2)
    page.add_text("""
    For large datasets or many files, you don't want to load everything into memory at once.
    Use `.variants()` with a loader factory to load data lazily during pipeline execution.
    """)
    
    code_example_2 = """
from sigchain import Pipeline
from sigchain.core.data import SignalData
import numpy as np

# Create a loader factory - data is loaded only when the variant executes
def make_loader(filename):
    def load(_):
        data = np.load(filename)  # Loaded on demand, not upfront
        return SignalData(data, sample_rate=20e6)
    return load

# Process multiple files through the same pipeline
# Each file is loaded only when needed, one at a time
results = (Pipeline()
    .variants(make_loader, 
              ['dataset_a.npy', 'dataset_b.npy', 'dataset_c.npy'],
              names=['Dataset A', 'Dataset B', 'Dataset C'])
    .add(StackPulses())
    .add(RangeCompress(window='hamming'))
    .add(DopplerCompress(window='hann'))
    .run()
)

# Results contains one (params, result) tuple for each file
for params, result in results:
    dataset_name = params['variant'][0]
    print(f"Processed {dataset_name}")
"""
    page.add_syntax(code_example_2, language='python')
    
    # Live example 2: Save signals and load them lazily
    page.add_header("Live Example: Lazy Loading from Saved Files", level=3)
    page.add_text("First, generate and save three different target scenarios to files:")
    
    import tempfile
    import os
    
    # Create temp directory for this demo
    temp_dir = tempfile.mkdtemp()
    
    # Generate and save three signals
    scenarios = [
        ('near_slow', 2e-6, 150.0),
        ('mid_fast', 3e-6, 250.0),
        ('far_receding', 4e-6, -100.0)
    ]
    
    for name, delay, doppler in scenarios:
        signal = LFMGenerator(
            num_pulses=32, target_delay=delay, target_doppler=doppler, noise_power=0.01
        )(None)
        filepath = os.path.join(temp_dir, f'{name}.npz')
        np.savez(filepath, data=signal.data, sample_rate=signal.sample_rate, metadata=signal.metadata)
    
    page.add_text(f"✓ Saved 3 signal files to temporary directory")
    
    # Now demonstrate lazy loading
    page.add_text("\\nNow load and process them lazily - one file at a time:")
    
    def make_loader(filepath, scenario_name):
        def load(_):
            # This is only called when this variant executes
            npz = np.load(filepath)
            return SignalData(npz['data'], sample_rate=float(npz['sample_rate']))
        return load
    
    # Build list of file paths and names
    file_paths = [os.path.join(temp_dir, f'{name}.npz') for name, _, _ in scenarios]
    scenario_names = ['Near/Slow', 'Mid/Fast', 'Far/Receding']
    
    # Process all files through pipeline - loaded one at a time
    results = (Pipeline()
        .variants(lambda fp: make_loader(fp, None), file_paths, names=scenario_names)
        .add(StackPulses())
        .add(RangeCompress(window='hamming', oversample_factor=2))
        .add(DopplerCompress(window='hann', oversample_factor=2))
        .run()
    )
    
    # Plot each result
    for params, result in results:
        scenario_name = params['variant'][0]
        fig = plot_range_doppler_map(result, title=f"{scenario_name} (loaded from file)", 
                                     height=450, use_db=True, mark_target=True)
        page.add_plot(fig, height=450)
    
    # Clean up temp files
    for filepath in file_paths:
        os.remove(filepath)
    os.rmdir(temp_dir)
    
    # Demo 3: Combined lazy loading with processing variants
    page.add_header("Method 3: Combine Lazy Loading with Processing Variants", level=2)
    page.add_text("""
    Combine lazy-loaded data variants with processing parameter variants to explore
    the full cartesian product without loading all data into memory at once.
    """)
    
    code_example_3 = """
from sigchain import Pipeline

# Loader factory
def make_loader(filename):
    def load(_):
        data = np.load(filename)
        return SignalData(data, sample_rate=20e6)
    return load

# 3 files × 2 range windows × 2 Doppler windows = 12 total combinations
# But only one file is in memory at a time!
results = (Pipeline()
    .variants(make_loader, 
              ['sig_a.npy', 'sig_b.npy', 'sig_c.npy'],
              names=['Signal A', 'Signal B', 'Signal C'])
    .add(StackPulses())
    .variants(lambda w: RangeCompress(window=w), 
              ['hamming', 'blackman'],
              names=['Hamming', 'Blackman'])
    .variants(lambda w: DopplerCompress(window=w), 
              ['hann', 'hamming'],
              names=['Hann', 'Hamming'])
    .run()
)

# Access all three levels of variants
for params, result in results:
    signal_name = params['variant'][0]
    range_window = params['variant'][1]
    doppler_window = params['variant'][2]
    print(f"{signal_name} + Range:{range_window} + Doppler:{doppler_window}")
"""
    page.add_syntax(code_example_3, language='python')
    
    # Live example 3
    page.add_header("Live Example: 2 Files × 2 Range × 2 Doppler = 8 Combinations", level=3)
    page.add_text("Generating and saving 2 signals, then loading lazily with processing variants:")
    
    # Create another temp directory
    temp_dir2 = tempfile.mkdtemp()
    
    # Generate and save two signals
    sig_a = LFMGenerator(num_pulses=24, target_delay=2e-6, target_doppler=200.0, noise_power=0.01)(None)
    sig_b = LFMGenerator(num_pulses=24, target_delay=3e-6, target_doppler=-150.0, noise_power=0.01)(None)
    
    file_a = os.path.join(temp_dir2, 'target1.npz')
    file_b = os.path.join(temp_dir2, 'target2.npz')
    
    np.savez(file_a, data=sig_a.data, sample_rate=sig_a.sample_rate)
    np.savez(file_b, data=sig_b.data, sample_rate=sig_b.sample_rate)
    
    # Lazy loader factory
    def make_file_loader(filepath):
        def load(_):
            npz = np.load(filepath)
            return SignalData(npz['data'], sample_rate=float(npz['sample_rate']))
        return load
    
    # Combine lazy loading with processing variants
    combined_results = (Pipeline()
        .variants(make_file_loader, [file_a, file_b], names=['Target 1', 'Target 2'])
        .add(StackPulses())
        .variants(lambda w: RangeCompress(window=w, oversample_factor=2), 
                 ['hamming', 'blackman'],
                 names=['Hamming', 'Blackman'])
        .variants(lambda w: DopplerCompress(window=w, oversample_factor=2), 
                 ['hann', 'hamming'],
                 names=['Hann', 'Hamming'])
        .run()
    )
    
    # Create comparison table
    table_data = []
    for params, result in combined_results:
        rdm_data = np.abs(result.data)
        peak_val = np.max(rdm_data)
        
        table_data.append({
            'Signal': params['variant'][0],
            'Range Window': params['variant'][1],
            'Doppler Window': params['variant'][2],
            'Peak Value': f'{peak_val:.1f}'
        })
    
    page.add_table(pd.DataFrame(table_data))
    
    # Plot all combinations
    for params, result in combined_results:
        title = f"{params['variant'][0]}: Range={params['variant'][1]}, Doppler={params['variant'][2]}"
        fig = plot_range_doppler_map(result, title=title, height=400,
                                     use_db=True, mark_target=True)
        page.add_plot(fig, height=400)
    
    # Clean up temp files
    os.remove(file_a)
    os.remove(file_b)
    os.rmdir(temp_dir2)
    
    page.add_header("Summary", level=2)
    page.add_text("""
    Key benefits of lazy loading with `.variants()`:
    
    1. **Memory Efficient**: Only one signal in memory at a time
    2. **Scalable**: Process hundreds of files without memory issues
    3. **Flexible**: Combine with processing variants for full exploration
    4. **Consistent Processing**: Same pipeline applied to all data
    5. **Easy Pattern**: Just wrap your loader in a factory function
    
    Pattern to remember:
    ```python
    def make_loader(filename):
        def load(_):
            # Load happens here, during execution
            data = load_from_somewhere(filename)
            return SignalData(data, sample_rate=...)
        return load
    
    results = Pipeline().variants(make_loader, file_list, names=...).add(...).run()
    ```
    
    Use cases:
    - Processing large datasets that don't fit in memory
    - Batch processing many files from disk or network
    - Testing algorithms across multiple scenarios
    - Comparing data from different sensors or time periods
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
        
        # Create and add parameter exploration dashboard
        print("Creating parameter exploration demo...")
        param_dashboard = create_parameter_exploration_dashboard()
        directory.add_dashboard(param_dashboard, slug='parameter-exploration')
        
        # Create and add post-processing plots dashboard
        print("Creating post-processing plots demo...")
        post_plots_dashboard = create_post_processing_plots_dashboard()
        directory.add_dashboard(post_plots_dashboard, slug='post-processing-plots')
        
        # Create and add input variants dashboard
        print("Creating input variants demo...")
        input_variants_dashboard = create_input_variants_dashboard()
        directory.add_dashboard(input_variants_dashboard, slug='input-variants')
        
        # Publish everything to docs/ directory
        print("Publishing dashboards...")
        directory.publish('docs')
        
        print(f"\n{'='*70}")
        print(f"✓ Dashboards created successfully!")
        print(f"{'='*70}")
        print(f"\nGenerated files:")
        print(f"  docs/index.html              - Landing page with all dashboards")
        print(f"  docs/radar-processing/       - Complete radar processing demo")
        print(f"  docs/custom-blocks-tutorial/ - Custom blocks tutorial")
        print(f"  docs/parameter-exploration/  - Parameter sweep demo")
        print(f"  docs/post-processing-plots/  - Post-processing plotting demo")
        print(f"  docs/input-variants/         - Input variants demo")
        print(f"\nTo view: Open docs/index.html in a web browser")
        print(f"{'='*70}\n")
