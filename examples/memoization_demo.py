"""
Memoization Demo

Demonstrates the performance benefits of graph memoization when exploring
parameter variants. Shows how shared computation stages are cached and reused.

Uses simple operations with artificial delays to clearly show the speedup.
"""

import numpy as np
import pandas as pd
import time
from sigexec import Graph, GraphData

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False


# Create simple processing blocks with artificial delays
def expensive_load_data(delay: float = 0.2):
    """Simulates expensive data loading (e.g., from disk/network)."""
    def load(_):
        time.sleep(delay)
        data = np.random.randn(1000) + 1j * np.random.randn(1000)
        return GraphData(data=data, metadata={'sample_rate': 1e6, 'stage': 'loaded'})
    return load


def expensive_preprocessing(delay: float = 0.15):
    """Simulates expensive preprocessing (e.g., filtering, calibration)."""
    def preprocess(signal_data: GraphData):
        time.sleep(delay)
        data = signal_data.data * np.exp(1j * 0.1)  # Simple phase shift
        metadata = signal_data.metadata.copy()
        metadata['stage'] = 'preprocessed'
        return GraphData(data=data, metadata=metadata)
    return preprocess


def cheap_operation(label: str, factor: float = 1.0):
    """Fast operation that varies by parameter."""
    def process(signal_data: GraphData):
        # No sleep - this is actually fast
        data = signal_data.data * factor
        metadata = signal_data.metadata.copy()
        metadata['stage'] = label
        metadata['factor'] = factor
        return GraphData(data=data, metadata=metadata)
    return process


def create_dashboard() -> sd.Dashboard:
    """
    Create memoization demo dashboard.
    
    Demonstrates the performance benefits of automatic memoization when
    exploring parameter variants.
    
    Returns:
        Dashboard object ready to be added to a Directory
    """
    dashboard = sd.Dashboard('Memoization Performance')
    page = sd.Page('memoization-demo', 'Memoization Demo')
    
    page.add_header("Graph Memoization Performance", level=1)
    page.add_text("""
    When exploring parameter variants, pipelines automatically cache (memoize) intermediate results.
    This means shared computation stages only execute once, dramatically improving performance.
    
    This demo uses simple operations with artificial delays to clearly demonstrate the concept.
    """)
    
    # Show graph visualization
    page.add_header("Graph Structure", level=2)
    example_graph = (Graph("Example")
        .add(expensive_load_data(0.2), name="Load_Data")
        .add(expensive_preprocessing(0.15), name="Preprocess")
        .variant(lambda f: cheap_operation("variant", f), [1.0, 1.5, 2.0], names=["x1.0", "x1.5", "x2.0"]))
    
    page.add_syntax(example_graph.to_mermaid(), language='mermaid')
    page.add_text("""
    The diagram above shows how the expensive operations (Load_Data, Preprocess) are shared
    across all variants, while only the cheap variant operation differs.
    """)
    
    # Demo 1: Show the concept
    page.add_header("Understanding Memoization", level=2)
    page.add_text("""
    Consider a graph that:
    1. Loads data (2 seconds - expensive)
    2. Preprocesses data (1.5 seconds - expensive)
    3. Applies different scaling factors (fast, varies per combination)
    
    If we explore 3 scaling factors:
    
    **Without memoization:**
    - Load data: 2s × 3 = 6 seconds
    - Preprocess: 1.5s × 3 = 4.5 seconds
    - Scale: 3 fast operations
    - **Total: ~10.5 seconds**
    
    **With memoization:**
    - Load data: 2s × 1 = 2 seconds (cached!)
    - Preprocess: 1.5s × 1 = 1.5 seconds (cached!)
    - Scale: 3 fast operations
    - **Total: ~3.5 seconds**
    
    **Expected speedup: 3x** (10.5s / 3.5s)
    """)
    
    page.add_header("Code Example", level=2)
    code_example = """
from sigexec import Graph, GraphData
import time
import numpy as np

# Define operations with artificial delays
def expensive_load_data(delay=0.2):
    def load(_):
        time.sleep(delay)  # Simulate expensive operation
        data = np.random.randn(1000)
        return GraphData(data, metadata={'sample_rate': 1e6})
    return load

def expensive_preprocessing(delay=0.15):
    def preprocess(signal_data):
        time.sleep(delay)  # Simulate expensive operation
        return GraphData(signal_data.data * 2, signal_data.metadata)
    return preprocess

def cheap_operation(factor):
    def process(signal_data):
        return GraphData(signal_data.data * factor, signal_data.metadata)
    return process

# With memoization (default)
start = time.time()
results_cached = (Graph("Cached", enable_cache=True)
    .add(expensive_load_data(delay=0.2))      # Runs once (0.2s)
    .add(expensive_preprocessing(delay=0.15))  # Runs once (0.15s)
    .variants(cheap_operation, [1.0, 2.0, 3.0], 
              names=['1x', '2x', '3x'])        # Runs 3 times (fast)
    .run()
)
cached_time = time.time() - start
print(f"With cache: {cached_time:.1f}s")  # ~0.35s

# Without memoization
start = time.time()
results_uncached = (Graph("Uncached", enable_cache=False)
    .add(expensive_load_data(delay=0.2))      # Runs 3 times (0.6s)
    .add(expensive_preprocessing(delay=0.15))  # Runs 3 times (0.45s)
    .variants(cheap_operation, [1.0, 2.0, 3.0],
              names=['1x', '2x', '3x'])        # Runs 3 times (fast)
    .run()
)
uncached_time = time.time() - start
print(f"Without cache: {uncached_time:.1f}s")  # ~1.05s
print(f"Speedup: {uncached_time/cached_time:.1f}x")  # ~3x
"""
    page.add_syntax(code_example, language='python')
    
    # Live performance comparison
    page.add_header("Live Performance Comparison", level=2)
    page.add_text("Running the same parameter exploration with and without memoization:")
    
    # Define delays for demonstration (reduced for faster dashboard generation)
    load_delay = 0.2  # seconds
    preprocess_delay = 0.15  # seconds
    num_variants = 3
    
    # Test with cache enabled
    page.add_text("\\n**Running with memoization enabled...**")
    Graph._global_cache.clear()
    start_cached = time.time()
    results_cached = (Graph("CachedPipeline", enable_cache=True)
        .add(expensive_load_data(delay=load_delay), name="Load Data")
        .add(expensive_preprocessing(delay=preprocess_delay), name="Preprocess")
        .variants(cheap_operation, [1.0, 2.0, 3.0], names=['Scale 1x', 'Scale 2x', 'Scale 3x'])
        .run(verbose=False)
    )
    cached_time = time.time() - start_cached
    
    # Clear cache before uncached run
    Graph._global_cache.clear()
    
    # Test without cache
    page.add_text("\\n**Running without memoization...**")
    start_uncached = time.time()
    results_uncached = (Graph("UncachedPipeline", enable_cache=False)
        .add(expensive_load_data(delay=load_delay), name="Load Data")
        .add(expensive_preprocessing(delay=preprocess_delay), name="Preprocess")
        .variants(cheap_operation, [1.0, 2.0, 3.0], names=['Scale 1x', 'Scale 2x', 'Scale 3x'])
        .run(verbose=False)
    )
    uncached_time = time.time() - start_uncached
    
    # Calculate expected times
    expected_cached = load_delay + preprocess_delay  # Expensive ops run once
    expected_uncached = (load_delay + preprocess_delay) * num_variants  # Run per variant
    expected_speedup = expected_uncached / expected_cached
    
    # Calculate actual results
    speedup = uncached_time / cached_time
    time_saved = uncached_time - cached_time
    
    # Create results table
    results_data = {
        'Configuration': [
            'Expected (Cached)',
            'Actual (Cached)', 
            'Expected (Uncached)',
            'Actual (Uncached)',
            'Expected Speedup',
            'Actual Speedup'
        ],
        'Execution Time': [
            f'{expected_cached:.1f}s',
            f'{cached_time:.1f}s',
            f'{expected_uncached:.1f}s',
            f'{uncached_time:.1f}s',
            f'{expected_speedup:.1f}x',
            f'{speedup:.1f}x'
        ],
        'Description': [
            f'{load_delay}s load + {preprocess_delay}s preprocess (once)',
            'Actual measured time',
            f'({load_delay}s + {preprocess_delay}s) × {num_variants} variants',
            'Actual measured time',
            f'{expected_uncached:.1f}s / {expected_cached:.1f}s',
            'Actual speedup achieved'
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    page.add_table(results_df)
    
    page.add_text(f"""
    **Result**: Memoization provides a **{speedup:.1f}x speedup** (expected: {expected_speedup:.1f}x)!
    
    The speedup comes from:
    - Load data ({load_delay}s) runs **1 time** instead of {num_variants} times → saves {load_delay * (num_variants - 1):.1f}s
    - Preprocess ({preprocess_delay}s) runs **1 time** instead of {num_variants} times → saves {preprocess_delay * (num_variants - 1):.1f}s
    - Scaling operations run {num_variants} times in both cases (can't be shared)
    
    **Total time saved: {time_saved:.1f} seconds** ({time_saved/uncached_time*100:.0f}% reduction)
    """)
    
    # Scaling analysis
    page.add_header("Scaling with Parameter Space Size", level=2)
    page.add_text("""
    The benefits of memoization scale linearly with the number of parameter variants.
    Let's see how speedup changes with different numbers of variants:
    """)
    
    scaling_data = []
    short_delay = 0.5  # Use shorter delays for scaling demo
    
    for n_variants in [2, 3, 4, 5]:
        factors = list(range(1, n_variants + 1))
        names = [f'{f}x' for f in factors]
        
        expected_cached = 2 * short_delay
        expected_uncached = 2 * short_delay * n_variants
        expected_speedup = expected_uncached / expected_cached
        
        # With cache
        Graph._global_cache.clear()
        start = time.time()
        _ = (Graph("Test", enable_cache=True)
            .add(expensive_load_data(delay=short_delay))
            .add(expensive_preprocessing(delay=short_delay))
            .variants(cheap_operation, factors, names=names)
            .run(verbose=False)
        )
        cached = time.time() - start
        
        # Without cache
        Graph._global_cache.clear()
        start = time.time()
        _ = (Graph("Test", enable_cache=False)
            .add(expensive_load_data(delay=short_delay))
            .add(expensive_preprocessing(delay=short_delay))
            .variants(cheap_operation, factors, names=names)
            .run(verbose=False)
        )
        uncached = time.time() - start
        
        scaling_data.append({
            'Variants': n_variants,
            'Time (Cached)': f'{cached:.2f}s',
            'Time (Uncached)': f'{uncached:.2f}s',
            'Expected Speedup': f'{expected_speedup:.1f}x',
            'Actual Speedup': f'{uncached/cached:.1f}x',
            'Time Saved': f'{uncached - cached:.2f}s'
        })
    
    scaling_df = pd.DataFrame(scaling_data)
    page.add_table(scaling_df)
    
    page.add_text("""
    As you can see, the speedup scales linearly with the number of variants:
    - 2 variants → ~2x speedup
    - 3 variants → ~3x speedup
    - 4 variants → ~4x speedup
    - 5 variants → ~5x speedup
    
    This is because:
    - **Cached version**: Expensive operations run once regardless of variant count
    - **Uncached version**: Expensive operations run N times for N variants
    - **Speedup = N** (number of variants)
    """)
    
    # Show execution details
    page.add_header("Execution Details", level=2)
    page.add_text("""
    Here's what happens during execution:
    """)
    
    execution_comparison = {
        'Stage': ['Load Data', 'Preprocess', 'Scale 1x', 'Scale 2x', 'Scale 3x', 'TOTAL'],
        'With Cache': [
            f'{load_delay}s (executed)',
            f'{preprocess_delay}s (executed)',
            '<0.1s (executed)',
            '<0.1s (executed)',
            '<0.1s (executed)',
            f'~{expected_cached:.1f}s'
        ],
        'Without Cache': [
            f'{load_delay}s × 3 = {load_delay*3:.1f}s',
            f'{preprocess_delay}s × 3 = {preprocess_delay*3:.1f}s',
            '<0.1s (executed)',
            '<0.1s (executed)',
            '<0.1s (executed)',
            f'~{expected_uncached:.1f}s'
        ]
    }
    
    exec_df = pd.DataFrame(execution_comparison)
    page.add_table(exec_df)
    
    page.add_text("""
    The cached version skips re-execution of expensive stages by reusing stored results.
    Each variant (Scale 1x, 2x, 3x) gets the same preprocessed data without waiting!
    """)
    
    # Best practices
    page.add_header("Best Practices", level=2)
    page.add_text("""
    To maximize the benefits of memoization:
    
    1. **Structure pipelines strategically**: Put expensive operations that are common 
       across variants early in the graph, before the first `.variants()` call.
    
    2. **Use `.variants()` not loops**: Instead of manually looping over parameters,
       use `.variants()` to let the framework handle caching automatically.
    
    3. **Group similar explorations**: If exploring multiple parameter dimensions,
       do them in one graph rather than separate runs.
    
    4. **Clear cache when needed**: The cache is shared across all Graph instances.
       Clear it with `Graph._global_cache.clear()` when starting a new experiment.
    
    5. **Disable cache for debugging**: Use `enable_cache=False` when debugging or when
       you explicitly want fresh execution (e.g., testing randomness).
    """)
    
    page.add_header("When Memoization Helps Most", level=2)
    page.add_text("""
    Memoization provides the biggest benefits when:
    
    - **Exploring parameter spaces**: Testing different processing parameters
    - **Expensive early stages**: Signal generation, filtering, or other costly operations
    - **Many combinations**: Large cartesian products of variants
    - **Branching pipelines**: Multiple processing paths from the same source
    - **Iterative development**: Re-running similar experiments during development
    
    It's less helpful when:
    - Running a single linear graph (no variants)
    - Every stage is unique (no shared computation)
    - Intermediate results are very large (cache memory overhead)
    """)
    
    dashboard.add_page(page)
    return dashboard


if __name__ == "__main__":
    if not STATICDASH_AVAILABLE:
        print("staticdash not available. Install with: pip install staticdash")
    else:
        print("Creating memoization demo dashboard...")
        dashboard = create_dashboard()
        
        # Publish standalone
        directory = sd.Directory(title='Memoization Demo', page_width=1000)
        directory.add_dashboard(dashboard, slug='memoization-demo')
        directory.publish('staticdash')
        
        print("✓ Dashboard published to staticdash/")
        print("  Open staticdash/index.html in a web browser")
