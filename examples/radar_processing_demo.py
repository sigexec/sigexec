"""
Radar Processing Demo

Comprehensive radar signal processing graph from LFM generation through
range-Doppler map creation.
"""

import numpy as np
import pandas as pd
from sigexec import Graph, SignalData
from sigexec.blocks import (
    LFMGenerator,
    StackPulses,
    RangeCompress,
    DopplerCompress,
)
from sigexec.diagnostics import (
    plot_timeseries,
    plot_pulse_matrix,
    plot_range_profile,
    plot_range_doppler_map,
)

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False


def create_dashboard(
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
    dashboard = sd.Dashboard('Radar Signal Processing Graph')
    
    # Create page
    page = sd.Page('radar-demo', 'Radar Signal Processing Graph Demo')
    
    # Add title and introduction
    page.add_header("Radar Signal Processing Graph", level=1)
    page.add_text("""
    This demonstration shows the complete radar signal processing graph,
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
    Here's how to reproduce this processing graph using sigchain:
    """)
    
    code_example = f"""
import staticdash as sd
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigexec.diagnostics import plot_timeseries, plot_pulse_matrix, plot_range_profile, plot_range_doppler_map

page = sd.Page('radar', 'Radar Processing')
page.add_header("Radar Signal Processing Graph", level=1)

result = (Graph("Radar")
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
    
    # Build the graph with inline plotting
    page.add_header("Processing Graph", level=2)
    
    signal_rdm = (Graph("Radar")
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


if __name__ == "__main__":
    if not STATICDASH_AVAILABLE:
        print("staticdash not available. Install with: pip install staticdash")
    else:
        print("Creating radar processing dashboard...")
        dashboard = create_dashboard(
            num_pulses=128,
            target_delay=2e-6,
            target_doppler=200.0,
            noise_power=0.01,
        )
        
        # Publish standalone
        directory = sd.Directory(title='Radar Processing Demo', page_width=1000)
        directory.add_dashboard(dashboard, slug='radar-processing')
        directory.publish('staticdash')
        
        print("✓ Dashboard published to staticdash/")
        print("  Open staticdash/index.html in a web browser")
