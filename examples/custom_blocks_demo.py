"""
Custom Blocks Tutorial Demo

Shows how to create custom processing blocks inline and compose them
with built-in blocks in a processing pipeline.
"""

import numpy as np
from sigchain import Pipeline, SignalData
from sigchain.blocks import LFMGenerator
from sigchain.diagnostics import plot_timeseries

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False


def create_dashboard() -> sd.Dashboard:
    """
    Create custom processing tutorial dashboard.
    
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
    return SignalData(data, signal_data.metadata)

# Define another custom block - normalize signal
def normalize_signal(signal_data: SignalData) -> SignalData:
    data = signal_data.data.copy()
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return SignalData(data, signal_data.metadata)

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
        return SignalData(data, metadata)
    
    def normalize_signal(signal_data: SignalData) -> SignalData:
        """Normalize signal to max amplitude of 1."""
        data = signal_data.data.copy()
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        metadata = signal_data.metadata.copy()
        metadata['normalized'] = True
        metadata['original_max'] = max_val
        return SignalData(data, metadata)
    
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
        print("staticdash not available. Install with: pip install staticdash")
    else:
        print("Creating custom blocks tutorial dashboard...")
        dashboard = create_dashboard()
        
        # Publish standalone
        directory = sd.Directory(title='Custom Blocks Tutorial', page_width=1000)
        directory.add_dashboard(dashboard, slug='custom-blocks-tutorial')
        directory.publish('staticdash')
        
        print("✓ Dashboard published to staticdash/")
        print("  Open staticdash/index.html in a web browser")
