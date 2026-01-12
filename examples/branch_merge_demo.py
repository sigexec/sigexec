"""
Branch and Merge DAG Processing Demo

Demonstrates how to create parallel processing paths (branches) in a pipeline,
process data differently in each branch, and then merge the results.
"""

import numpy as np
from sigchain import Pipeline, SignalData
from sigchain.diagnostics import plot_timeseries

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
from sigchain import Pipeline, SignalData
import numpy as np

# Generate a simple signal
data = SignalData(data=np.sin(2 * np.pi * np.linspace(0, 1, 100)))

# Create pipeline with duplicate branches
pipeline = (
    Pipeline("duplicate_branches")
    .input_data(data)
    .branch(["high_pass", "low_pass"])  # Duplicate into two branches
    .add(high_pass_filter, branch="high_pass")
    .add(low_pass_filter, branch="low_pass")
    .merge(["high_pass", "low_pass"], 
           combiner=lambda sigs: SignalData(data=sigs[0].data + sigs[1].data))
)

result = pipeline.run()
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
    return SignalData(data=np.abs(sig.data))

def extract_phase(sig):
    return SignalData(data=np.angle(sig.data))

pipeline = (
    Pipeline("feature_extraction")
    .input_data(complex_signal)
    .branch(["magnitude", "phase"], 
            functions=[extract_magnitude, extract_phase])
    .add(process_magnitude, branch="magnitude")
    .add(process_phase, branch="phase")
    .merge(["magnitude", "phase"], 
           combiner=lambda sigs: combine_features(sigs[0], sigs[1]))
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
    signal_data = SignalData(data=signal, metadata={'sample_rate': 1000.0})
    
    # Define filters
    def low_pass_filter(sig):
        """Simple moving average low-pass filter."""
        window = 20
        filtered = np.convolve(sig.data, np.ones(window)/window, mode='same')
        return SignalData(data=filtered, metadata=sig.metadata)
    
    def high_pass_filter(sig):
        """High-pass filter (signal minus low-pass)."""
        window = 20
        low_passed = np.convolve(sig.data, np.ones(window)/window, mode='same')
        high_passed = sig.data - low_passed
        return SignalData(data=high_passed, metadata=sig.metadata)
    
    def amplify(factor):
        """Create an amplification function."""
        def _amplify(sig):
            return SignalData(data=sig.data * factor, metadata=sig.metadata)
        return _amplify
    
    # Create pipeline with branches
    pipeline = (
        Pipeline("parallel_filtering")
        .input_data(signal_data)
        .branch(["lowpass", "highpass"])
        .add(low_pass_filter, name="low_pass_filter", branch="lowpass")
        .add(high_pass_filter, name="high_pass_filter", branch="highpass")
        .add(amplify(2.0), name="amplify_low", branch="lowpass")
        .add(amplify(3.0), name="amplify_high", branch="highpass")
        .merge(
            ["lowpass", "highpass"], 
            combiner=lambda sigs: SignalData(
                data=sigs[0].data + sigs[1].data,
                metadata=sigs[0].metadata
            ),
            output_name="combined"
        )
    )
    
    result = pipeline.run()
    
    # Show code
    demo_code = """
# Generate mixed frequency signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)
signal_data = SignalData(data=signal, metadata={'sample_rate': 1000.0})

# Define filters
def low_pass_filter(sig):
    window = 20
    filtered = np.convolve(sig.data, np.ones(window)/window, mode='same')
    return SignalData(data=filtered, metadata=sig.metadata)

def high_pass_filter(sig):
    window = 20
    low_passed = np.convolve(sig.data, np.ones(window)/window, mode='same')
    return SignalData(data=sig.data - low_passed, metadata=sig.metadata)

def amplify(factor):
    def _amplify(sig):
        return SignalData(data=sig.data * factor, metadata=sig.metadata)
    return _amplify

# Create branching pipeline
pipeline = (
    Pipeline("parallel_filtering")
    .input_data(signal_data)
    .branch(["lowpass", "highpass"])                    # Split into two paths
    .add(low_pass_filter, branch="lowpass")             # Filter low frequencies
    .add(high_pass_filter, branch="highpass")           # Filter high frequencies  
    .add(amplify(2.0), branch="lowpass")                # Amplify low 2x
    .add(amplify(3.0), branch="highpass")               # Amplify high 3x
    .merge(["lowpass", "highpass"],                     # Combine results
           combiner=lambda sigs: SignalData(
               data=sigs[0].data + sigs[1].data,
               metadata=sigs[0].metadata
           ))
)

result = pipeline.run()
"""
    page.add_syntax(demo_code, language='python')
    
    # Visualize results
    page.add_header("Results", level=3)
    page.add_text("Original signal (5 Hz + 50 Hz components):")
    fig1 = plot_timeseries(signal_data, title="Original Signal")
    page.add_plot(fig1)
    
    page.add_text("Combined result after parallel filtering and amplification:")
    fig2 = plot_timeseries(result, title="Processed Signal (Low×2 + High×3)")
    page.add_plot(fig2)
    
    # Variants vs Branch/Merge
    page.add_header("Variants vs Branch/Merge", level=2)
    page.add_text("""
    It's important to understand when to use each approach:
    
    **`.variants()`** - Parameter Exploration:
    - Explores different configurations of the same pipeline
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
pipeline = (
    Pipeline("variants_and_branches")
    .input_data(signal_data)
    .variants(amplify, [1.0, 2.0, 5.0], names=["1x", "2x", "5x"])
    .branch(["lowpass", "highpass"])
    .add(low_pass_filter, branch="lowpass")
    .add(high_pass_filter, branch="highpass")
    .merge(["lowpass", "highpass"], 
           combiner=lambda sigs: SignalData(
               data=sigs[0].data + sigs[1].data,
               metadata=sigs[0].metadata
           ))
)

results = pipeline.run()  # Returns 3 results, one for each amplification factor
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
    - **Combiner** receives a list of SignalData in the order of branch names
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
