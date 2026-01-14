"""
Branch and Merge DAG Processing Demo

Demonstrates how to create parallel processing paths (branches) in a graph,
process data differently in each branch, and then merge the results.
"""

import numpy as np
from sigexec import Graph, GraphData
from sigexec.diagnostics import plot_timeseries

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False


def create_dashboard() -> sd.Dashboard:
    """
    Create branch and merge demo dashboard.
    
    Returns:
        Dashboard object ready to be added to a Directory
    """
    
    dashboard = sd.Dashboard('Branch and Merge DAG Processing')
    page = sd.Page('branch-merge', 'Branch and Merge Demo')
    
    page.add_header("DAG-Based Parallel Processing", level=1)
    page.add_text("""
    Branch and merge operations allow you to create Directed Acyclic Graph (DAG) 
    pipelines where data flows through multiple parallel paths before being combined.
    This is different from `.variants()` which explores parameter combinations.
    """)
    
    # Example 1: Duplicate branches
    page.add_header("Example 1: Duplicate Branches", level=2)
    page.add_text("""
    The simplest form creates branches that receive copies of the same data.
    Each branch can then process the data independently.
    """)
    
    code_example_1 = """
from sigexec import Graph, GraphData
import numpy as np

# Generate a simple signal
data = GraphData(data=np.sin(2 * np.pi * np.linspace(0, 1, 100)))

# Create graph with duplicate branches
graph = (
    Graph("duplicate_branches")
    .input_data(data)
    .branch(["high_pass", "low_pass"])  # Duplicate into two branches
    .add(high_pass_filter, branch="high_pass")
    .add(low_pass_filter, branch="low_pass")
    .merge(["high_pass", "low_pass"], 
           combiner=lambda sigs: GraphData(data=sigs[0].data + sigs[1].data))
)

result = graph.run()
"""
    page.add_syntax(code_example_1, language='python')
    
    # Example 2: Function branches
    page.add_header("Example 2: Function Branches", level=2)
    page.add_text("""
    Instead of duplicating data, you can provide functions to extract different
    aspects of the signal into separate branches. This avoids unnecessary copying.
    """)
    
    code_example_2 = """
# Extract different features into separate branches
def extract_magnitude(sig):
    return GraphData(data=np.abs(sig.data))

def extract_phase(sig):
    return GraphData(data=np.angle(sig.data))

# Combiner to reconstruct complex signal from magnitude and phase
def combine_mag_phase(sigs):
    # Combine magnitude and phase back into complex signal
    magnitude = sigs[0].data  # First branch: magnitude
    phase = sigs[1].data      # Second branch: phase
    complex_data = magnitude * np.exp(1j * phase)
    return GraphData(data=complex_data, metadata=sigs[0].metadata)

graph = (
    Graph("feature_extraction")
    .input_data(complex_signal)
    .branch(["magnitude", "phase"], 
            functions=[extract_magnitude, extract_phase])
    .add(process_magnitude, branch="magnitude")
    .add(process_phase, branch="phase")
    .merge(["magnitude", "phase"], 
           combiner=combine_mag_phase)
)
"""
    page.add_syntax(code_example_2, language='python')
    
    # Practical Example
    page.add_header("Practical Example: Parallel Filtering", level=2)
    page.add_text("""
    Let's demonstrate with a practical example: applying different frequency
    filters in parallel and combining the results.
    """)
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    # Mix of low freq (5 Hz) and high freq (50 Hz)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    signal_data = GraphData(data=signal, metadata={'sample_rate': 1000.0})
    
    # Define filters
    def low_pass_filter(sig):
        """Simple moving average low-pass filter."""
        window = 20
        filtered = np.convolve(sig.data, np.ones(window)/window, mode='same')
        return GraphData(data=filtered, metadata=sig.metadata)
    
    def high_pass_filter(sig):
        """High-pass filter (signal minus low-pass)."""
        window = 20
        low_passed = np.convolve(sig.data, np.ones(window)/window, mode='same')
        high_passed = sig.data - low_passed
        return GraphData(data=high_passed, metadata=sig.metadata)
    
    def amplify(factor):
        """Create an amplification function."""
        def _amplify(sig):
            return GraphData(data=sig.data * factor, metadata=sig.metadata)
        return _amplify
    
    # Create graph with branches
    graph = (
        Graph("parallel_filtering")
        .input_data(signal_data)
        .branch(["lowpass", "highpass"])
        .add(low_pass_filter, name="low_pass_filter", branch="lowpass")
        .add(high_pass_filter, name="high_pass_filter", branch="highpass")
        .add(amplify(2.0), name="amplify_low", branch="lowpass")
        .add(amplify(3.0), name="amplify_high", branch="highpass")
        .merge(
            lambda branches: GraphData(
                data=branches['lowpass'].data + branches['highpass'].data,
                metadata=branches['lowpass'].metadata
            ),
            branches=["lowpass", "highpass"],
            name="combined"
        )
    )
    
    result = graph.run()
    
    # Show code
    demo_code = """
# Generate mixed frequency signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)
signal_data = GraphData(data=signal, metadata={'sample_rate': 1000.0})

# Define filters
def low_pass_filter(sig):
    window = 20
    filtered = np.convolve(sig.data, np.ones(window)/window, mode='same')
    return GraphData(data=filtered, metadata=sig.metadata)

def high_pass_filter(sig):
    window = 20
    low_passed = np.convolve(sig.data, np.ones(window)/window, mode='same')
    return GraphData(data=sig.data - low_passed, metadata=sig.metadata)

def amplify(factor):
    def _amplify(sig):
        return GraphData(data=sig.data * factor, metadata=sig.metadata)
    return _amplify

# Create branching graph
graph = (
    Graph("parallel_filtering")
    .input_data(signal_data)
    .branch(["lowpass", "highpass"])                    # Split into two paths
    .add(low_pass_filter, branch="lowpass")             # Filter low frequencies
    .add(high_pass_filter, branch="highpass")           # Filter high frequencies  
    .add(amplify(2.0), branch="lowpass")                # Amplify low 2x
    .add(amplify(3.0), branch="highpass")               # Amplify high 3x
    .merge(["lowpass", "highpass"],                     # Combine results
           combiner=lambda sigs: GraphData(
               data=sigs[0].data + sigs[1].data,
               metadata=sigs[0].metadata
           ))
)

result = graph.run()
"""
    page.add_syntax(demo_code, language='python')
    
    # Visualize results
    page.add_header("Results", level=3)
    page.add_text("Original signal (5 Hz + 50 Hz components):")
    fig1 = plot_timeseries(signal_data, title="Original Signal")
    page.add_plot(fig1)
    
    # Get individual branch results by running separate pipelines
    lowpass_pipeline = (
        Graph("lowpass_only")
        .input_data(signal_data)
        .add(low_pass_filter, name="low_pass_filter")
        .add(amplify(2.0), name="amplify_low")
    )
    lowpass_result = lowpass_pipeline.run()
    
    highpass_pipeline = (
        Graph("highpass_only")
        .input_data(signal_data)
        .add(high_pass_filter, name="high_pass_filter")
        .add(amplify(3.0), name="amplify_high")
    )
    highpass_result = highpass_pipeline.run()
    
    page.add_text("Low-pass branch (amplified 2×):")
    fig_low = plot_timeseries(lowpass_result, title="Low-Pass Branch (5 Hz × 2)")
    page.add_plot(fig_low)
    
    page.add_text("High-pass branch (amplified 3×):")
    fig_high = plot_timeseries(highpass_result, title="High-Pass Branch (50 Hz × 3)")
    page.add_plot(fig_high)
    
    page.add_text("Combined result after parallel filtering and amplification:")
    fig2 = plot_timeseries(result, title="Processed Signal (Low×2 + High×3)")
    page.add_plot(fig2)
    
    # Amplitude/Phase Processing Example
    page.add_header("Real Example: Amplitude/Phase Processing", level=2)
    page.add_text("""
    Here's a practical example using function branches to extract and process
    amplitude and phase components of a complex signal separately.
    """)
    
    # Generate complex signal (e.g., IQ data)
    t = np.linspace(0, 1, 1000)
    i_component = np.cos(2 * np.pi * 10 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
    q_component = np.sin(2 * np.pi * 10 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
    complex_signal = i_component + 1j * q_component
    complex_data = GraphData(data=complex_signal, metadata={'sample_rate': 1000.0, 'type': 'IQ'})
    
    # Define extraction functions
    def extract_amplitude(sig):
        """Extract amplitude (magnitude) from complex signal."""
        return GraphData(data=np.abs(sig.data), metadata=sig.metadata)
    
    def extract_phase(sig):
        """Extract phase from complex signal."""
        return GraphData(data=np.angle(sig.data), metadata=sig.metadata)
    
    # Define processing functions
    def smooth_amplitude(sig):
        """Smooth amplitude with moving average."""
        window = 50
        smoothed = np.convolve(sig.data, np.ones(window)/window, mode='same')
        return GraphData(data=smoothed, metadata=sig.metadata)
    
    def unwrap_phase(sig):
        """Unwrap phase to remove discontinuities."""
        unwrapped = np.unwrap(sig.data)
        return GraphData(data=unwrapped, metadata=sig.metadata)
    
    # Combiner to reconstruct
    def reconstruct_complex(sigs):
        """Combine processed amplitude and phase back into complex signal."""
        amplitude = sigs[0].data
        phase = sigs[1].data
        reconstructed = amplitude * np.exp(1j * phase)
        return GraphData(data=reconstructed, metadata=sigs[0].metadata)
    
    # Create graph with function branches
    amp_phase_pipeline = (
        Graph("amplitude_phase_processing")
        .input_data(complex_data)
        .branch(["amplitude", "phase"], 
                functions=[extract_amplitude, extract_phase])
        .add(smooth_amplitude, name="smooth_amp", branch="amplitude")
        .add(unwrap_phase, name="unwrap_phase", branch="phase")
        .merge(
            reconstruct_complex,
            branches=["amplitude", "phase"],
            name="reconstructed"
        )
    )
    
    amp_phase_result = amp_phase_pipeline.run()
    
    # Show code
    amp_phase_code = """
# Generate complex IQ signal
t = np.linspace(0, 1, 1000)
i_component = np.cos(2*np.pi*10*t) * (1 + 0.3*np.sin(2*np.pi*2*t))
q_component = np.sin(2*np.pi*10*t) * (1 + 0.3*np.sin(2*np.pi*2*t))
complex_signal = i_component + 1j * q_component
complex_data = GraphData(data=complex_signal, metadata={'sample_rate': 1000.0})

# Define extraction and processing
def extract_amplitude(sig):
    return GraphData(data=np.abs(sig.data), metadata=sig.metadata)

def extract_phase(sig):
    return GraphData(data=np.angle(sig.data), metadata=sig.metadata)

def smooth_amplitude(sig):
    window = 50
    smoothed = np.convolve(sig.data, np.ones(window)/window, mode='same')
    return GraphData(data=smoothed, metadata=sig.metadata)

def unwrap_phase(sig):
    return GraphData(data=np.unwrap(sig.data), metadata=sig.metadata)

def reconstruct_complex(sigs):
    amplitude, phase = sigs[0].data, sigs[1].data
    return GraphData(data=amplitude * np.exp(1j * phase), 
                     metadata=sigs[0].metadata)

# Graph with function branches
graph = (
    Graph("amplitude_phase_processing")
    .input_data(complex_data)
    .branch(["amplitude", "phase"],              # Split by function
            functions=[extract_amplitude, extract_phase])
    .add(smooth_amplitude, branch="amplitude")   # Process amplitude
    .add(unwrap_phase, branch="phase")           # Process phase
    .merge(["amplitude", "phase"],               # Reconstruct
           combiner=reconstruct_complex)
)

result = graph.run()
"""
    page.add_syntax(amp_phase_code, language='python')
    
    # Visualize amplitude/phase results
    page.add_header("Results", level=3)
    
    # Get branch results by running separate pipelines for each branch
    amp_raw_pipeline = Graph("amp_raw").input_data(complex_data).add(extract_amplitude)
    raw_amp = amp_raw_pipeline.run()
    
    amp_smooth_pipeline = Graph("amp_smooth").input_data(complex_data).add(extract_amplitude).add(smooth_amplitude)
    smoothed_amp = amp_smooth_pipeline.run()
    
    phase_raw_pipeline = Graph("phase_raw").input_data(complex_data).add(extract_phase)
    raw_phase = phase_raw_pipeline.run()
    
    phase_unwrap_pipeline = Graph("phase_unwrap").input_data(complex_data).add(extract_phase).add(unwrap_phase)
    unwrapped_phase = phase_unwrap_pipeline.run()
    
    page.add_text("Amplitude branch processing:")
    fig_amp = plot_timeseries(raw_amp, title="Extracted Amplitude (Raw)")
    page.add_plot(fig_amp)
    
    fig_amp_smooth = plot_timeseries(smoothed_amp, title="Smoothed Amplitude")
    page.add_plot(fig_amp_smooth)
    
    page.add_text("Phase branch processing:")
    fig_phase = plot_timeseries(raw_phase, title="Extracted Phase (Wrapped)")
    page.add_plot(fig_phase)
    
    fig_phase_unwrap = plot_timeseries(unwrapped_phase, title="Unwrapped Phase")
    page.add_plot(fig_phase_unwrap)
    
    page.add_text("Reconstructed complex signal (real part shown):")
    reconstructed_real = GraphData(data=np.real(amp_phase_result.data), 
                                    metadata=amp_phase_result.metadata)
    fig_recon = plot_timeseries(reconstructed_real, title="Reconstructed Signal (Real Part)")
    page.add_plot(fig_recon)
    
    # Variants vs Branch/Merge
    page.add_header("Variants vs Branch/Merge", level=2)
    page.add_text("""
    It's important to understand when to use each approach:
    
    **`.variants()`** - Parameter Exploration:
    - Explores different configurations of the same graph
    - Duplicates the entire downstream graph for each variant
    - Returns a list of (params, result) tuples
    - Use when you want to compare different parameter settings
    
    **`.branch()` / `.merge()`** - DAG Structure:
    - Creates parallel processing paths within a single execution
    - Different operations on different branches
    - Returns a single result (or list if combined with variants)
    - Use when you want parallel processing and data combination
    
    You can even combine them: use `.variants()` to explore parameters,
    and `.branch()/.merge()` within each variant to create DAG processing!
    """)
    
    # Combined example
    page.add_header("Combining Variants and Branches", level=2)
    page.add_text("""
    Here's how you can combine both approaches for maximum flexibility:
    """)
    
    combined_code = """
# Explore different amplification factors using variants
# While using branches for parallel filtering
graph = (
    Graph("variants_and_branches")
    .input_data(signal_data)
    .variants(amplify, [1.0, 2.0, 5.0], names=["1x", "2x", "5x"])
    .branch(["lowpass", "highpass"])
    .add(low_pass_filter, branch="lowpass")
    .add(high_pass_filter, branch="highpass")
    .merge(["lowpass", "highpass"], 
           combiner=lambda sigs: GraphData(
               data=sigs[0].data + sigs[1].data,
               metadata=sigs[0].metadata
           ))
)

results = graph.run()  # Returns 3 results, one for each amplification factor
for params, result in results:
    print(f"Amplification: {params['variant'][0]}")
    # Each result is the merged output of the branch processing
"""
    page.add_syntax(combined_code, language='python')
    
    # Key points
    page.add_header("Key Points", level=2)
    page.add_text("""
    - **Branches** split data flow into parallel paths
    - **Duplicate mode**: `.branch(["a", "b"])` copies data to both branches
    - **Function mode**: `.branch(["a", "b"], functions=[fa, fb])` applies functions to create branch data
    - **Merge** combines branch results with a custom combiner function
    - **Combiner** receives a list of GraphData in the order of branch names
    - **Targeting**: Use `branch="name"` in `.add()` to target specific branches
    - **Caching**: Each branch operation is cached independently
    - **Variants**: Can be combined with branches for parameter exploration within DAG structure
    """)
    
    dashboard.add_page(page)
    return dashboard


def main():
    """Run demo and optionally save to HTML."""
    if not STATICDASH_AVAILABLE:
        print("staticdash not available. Install with: pip install staticdash")
        return
    
    # Create dashboard
    dashboard = create_dashboard()
    
    # Create directory and publish
    directory = sd.Directory('Branch and Merge Demo')
    directory.add_dashboard(dashboard)
    directory.publish('./docs/branch-merge-demo')
    
    print("✅ Branch and merge demo saved to ./docs/branch-merge-demo/")
    print("   Open ./docs/branch-merge-demo/index.html to view")


if __name__ == '__main__':
    main()
