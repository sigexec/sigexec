"""
Input Variants Demo

Processing multiple signals through the same graph with input_variants().
Shows how to run the same processing graph over different input signals,
and how to combine input variants with processing variants.
"""

import numpy as np
import pandas as pd
import tempfile
import os
from sigexec import Graph, GraphData
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigexec.diagnostics import plot_range_doppler_map

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False


def create_dashboard() -> sd.Dashboard:
    """
    Create input variants demo dashboard.
    
    Returns:
        Dashboard object ready to be added to a Directory
    """
    dashboard = sd.Dashboard('Input Variants')
    page = sd.Page('input-variants', 'Input Variants Demo')
    
    page.add_header("Processing Multiple Signals with input_variants()", level=1)
    page.add_text("""
    When you have multiple signals (from different files, sensors, or test scenarios) that need
    the same processing, use `.input_variants()` to run them all through the same graph.
    """)
    
    # Demo 1: Basic input variants
    page.add_header("Method 1: Process Multiple Input Signals", level=2)
    
    code_example_1 = """
from sigexec import Graph
from sigexec.blocks import RangeCompress, DopplerCompress
from sigexec import GraphData

# Load or generate different signals
signal_dataset_a = load_signal("dataset_a.bin")
signal_dataset_b = load_signal("dataset_b.bin")  
signal_dataset_c = load_signal("dataset_c.bin")

# Process all three through the same graph
results = (Graph()
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
    the same range/Doppler compression graph.
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
    results = (Graph()
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
    Use `.variants()` with a loader factory to load data lazily during graph execution.
    """)
    
    code_example_2 = """
from sigexec import Graph
from sigexec import GraphData
import numpy as np

# Create a loader factory - data is loaded only when the variant executes
def make_loader(filename):
    def load(_):
        data = np.load(filename)  # Loaded on demand, not upfront
        return GraphData(data, metadata={'sample_rate': 20e6})
    return load

# Process multiple files through the same graph
# Each file is loaded only when needed, one at a time
results = (Graph()
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
        np.savez(filepath, 
                 data=signal.data, 
                 sample_rate=signal.sample_rate,
                 **{f'metadata_{k}': v for k, v in signal.metadata.items() if isinstance(v, (int, float, str, np.ndarray))})
    
    page.add_text("✓ Saved 3 signal files to temporary directory")
    
    # Now demonstrate lazy loading
    page.add_text("\\nNow load and process them lazily - one file at a time:")
    
    def make_loader(filepath, scenario_name):
        def load(_):
            npz = np.load(filepath, allow_pickle=True)
            metadata = {k.replace('metadata_', ''): v for k, v in npz.items() 
                       if k.startswith('metadata_')}
            metadata['sample_rate'] = float(npz['sample_rate'])
            return GraphData(npz['data'], metadata=metadata)
        return load
    
    # Build list of file paths and names
    file_paths = [os.path.join(temp_dir, f'{name}.npz') for name, _, _ in scenarios]
    scenario_names = ['Near/Slow', 'Mid/Fast', 'Far/Receding']
    
    # Process all files through graph - loaded one at a time
    results = (Graph()
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
from sigexec import Graph

# Loader factory
def make_loader(filename):
    def load(_):
        data = np.load(filename)
        return GraphData(data, metadata={'sample_rate': 20e6})
    return load

# 3 files × 2 range windows × 2 Doppler windows = 12 total combinations
# But only one file is in memory at a time!
results = (Graph()
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
    
    # Save with metadata
    np.savez(file_a, 
             data=sig_a.data, 
             sample_rate=sig_a.sample_rate,
             **{f'metadata_{k}': v for k, v in sig_a.metadata.items() if isinstance(v, (int, float, str, np.ndarray))})
    np.savez(file_b, 
             data=sig_b.data, 
             sample_rate=sig_b.sample_rate,
             **{f'metadata_{k}': v for k, v in sig_b.metadata.items() if isinstance(v, (int, float, str, np.ndarray))})
    
    # Lazy loader factory
    def make_file_loader(filepath):
        def load(_):
            npz = np.load(filepath, allow_pickle=True)
            metadata = {k.replace('metadata_', ''): v for k, v in npz.items() 
                       if k.startswith('metadata_')}
            metadata['sample_rate'] = float(npz['sample_rate'])
            return GraphData(npz['data'], metadata=metadata)
        return load
    
    # Combine lazy loading with processing variants
    combined_results = (Graph()
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
    4. **Consistent Processing**: Same graph applied to all data
    5. **Easy Pattern**: Just wrap your loader in a factory function
    
    Pattern to remember:
    ```python
    def make_loader(filename):
        def load(_):
            # Load happens here, during execution
            data = load_from_somewhere(filename)
            return GraphData(data, sample_rate=...)
        return load
    
    results = Graph().variants(make_loader, file_list, names=...).add(...).run()
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
        print("staticdash not available. Install with: pip install staticdash")
    else:
        print("Creating input variants dashboard...")
        dashboard = create_dashboard()
        
        # Publish standalone
        directory = sd.Directory(title='Input Variants', page_width=1000)
        directory.add_dashboard(dashboard, slug='input-variants')
        directory.publish('staticdash')
        
        print("✓ Dashboard published to staticdash/")
        print("  Open staticdash/index.html in a web browser")
