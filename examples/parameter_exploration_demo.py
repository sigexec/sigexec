"""
Parameter Exploration Demo

Shows chained variants to explore cartesian product of processing parameters.
"""

import numpy as np
import pandas as pd
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigexec.diagnostics import plot_range_doppler_map

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False


def create_dashboard() -> sd.Dashboard:
    """
    Create parameter exploration dashboard.
    
    Demonstrates exploring the parameter space by chaining .variants() calls.
    
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
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress

# Build graph and chain variants
results = (Graph("Radar")
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
    
    # Build graph with chained variants
    page.add_header("Parameter Sweep Results", level=2)
    page.add_text("""
    Testing all combinations of window functions:
    - Range Compression: hamming, hann, blackman
    - Doppler Compression: hamming, hann
    
    Total combinations: 3 × 2 = 6
    """)
    
    results = (Graph("RadarBase")
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
    
    The graph uses memoization, so common stages (like signal generation and stacking)
    only execute once and are reused across all variants.
    """)
    
    dashboard.add_page(page)
    return dashboard


if __name__ == "__main__":
    if not STATICDASH_AVAILABLE:
        print("staticdash not available. Install with: pip install staticdash")
    else:
        print("Creating parameter exploration dashboard...")
        dashboard = create_dashboard()
        
        # Publish standalone
        directory = sd.Directory(title='Parameter Exploration', page_width=1000)
        directory.add_dashboard(dashboard, slug='parameter-exploration')
        directory.publish('staticdash')
        
        print("✓ Dashboard published to staticdash/")
        print("  Open staticdash/index.html in a web browser")
