"""
Demonstration of graph visualization without execution.

Shows how to visualize graph structure using Mermaid diagrams
before running the graph.
"""

from sigexec import Graph, GraphData
import numpy as np


def generate_signal(g):
    """Generate a test signal."""
    g.data = np.random.randn(1000)
    g.sample_rate = 1000.0
    return g


def apply_filter_a(g):
    """Apply filter A."""
    g.data = g.data * 0.9
    return g


def apply_filter_b(g):
    """Apply filter B."""
    g.data = g.data * 1.1
    return g


def compute_fft(g):
    """Compute FFT."""
    g.fft = np.fft.fft(g.data)
    return g


def compute_power(g):
    """Compute power spectrum."""
    g.power = np.abs(g.fft) ** 2
    return g


def merge_branches(branches):
    """Average the two branches."""
    result = GraphData()
    result.data = (branches[0].data + branches[1].data) / 2
    result.merged = True
    return result


def main():
    print("=" * 60)
    print("Graph Visualization Demo")
    print("=" * 60)
    print()
    
    # Example 1: Simple linear graph
    print("Example 1: Simple Linear Graph")
    print("-" * 60)
    
    simple_graph = (Graph("SimpleProcess")
        .add(generate_signal, name="Generate")
        .add(apply_filter_a, name="Filter")
        .add(compute_fft, name="FFT")
        .add(compute_power, name="Power"))
    
    print(simple_graph.to_mermaid())
    print()
    
    # Example 2: Graph with branches
    print("\nExample 2: Graph with Branches")
    print("-" * 60)
    
    branched_graph = (Graph("BranchedProcess")
        .add(generate_signal, name="Generate_Signal")
        .branch(["filter_a", "filter_b"])
        .add(apply_filter_a, branch="filter_a", name="Apply_Filter_A")
        .add(apply_filter_b, branch="filter_b", name="Apply_Filter_B")
        .add(compute_fft, name="Compute_FFT")
        .merge(merge_branches, branches=["filter_a", "filter_b"], name="Merge_Results"))
    
    print(branched_graph.to_mermaid())
    print()
    
    # Example 3: Graph with variants
    print("\nExample 3: Graph with Variants")
    print("-" * 60)
    
    variant_graph = (Graph("VariantProcess")
        .add(generate_signal, name="Generate")
        .variant(
            lambda w: lambda g: apply_window(g, w),
            ['hamming', 'hann', 'blackman'],
            names=['Hamming', 'Hann', 'Blackman']
        )
        .add(compute_fft, name="FFT"))
    
    print(variant_graph.to_mermaid())
    print()
    
    # Example 4: Complex graph with multiple branches and operations
    print("\nExample 4: Complex Multi-Branch Graph")
    print("-" * 60)
    
    complex_graph = (Graph("ComplexProcess")
        .add(generate_signal, name="Generate_Signal")
        .add(lambda g: normalize(g), name="Normalize")
        .branch(["path_a", "path_b", "path_c"])
        .add(apply_filter_a, branch="path_a", name="Filter_A")
        .add(apply_filter_b, branch="path_b", name="Filter_B")
        .add(lambda g: g, branch="path_c", name="No_Filter")
        .add(compute_fft, name="FFT_All_Paths")
        .merge(average_merge, branches=["path_a", "path_b", "path_c"], name="Average_Merge")
        .add(compute_power, name="Final_Power"))
    
    print(complex_graph.to_mermaid())
    print()
    
    # Save examples to files
    print("\nSaving visualizations to files...")
    simple_graph.visualize("simple_graph.md")
    branched_graph.visualize("branched_graph.md")
    variant_graph.visualize("variant_graph.md")
    complex_graph.visualize("complex_graph.md")
    
    print("\n✓ All visualizations saved!")
    print("\nYou can view these .md files in VS Code or any Markdown viewer")
    print("that supports Mermaid diagrams.")


def apply_window(g, window_type):
    """Apply windowing function."""
    g.data = g.data * get_window(window_type, len(g.data))
    g.window_type = window_type
    return g


def get_window(window_type, length):
    """Get window function."""
    if window_type == 'hamming':
        return np.hamming(length)
    elif window_type == 'hann':
        return np.hanning(length)
    elif window_type == 'blackman':
        return np.blackman(length)
    return np.ones(length)


def normalize(g):
    """Normalize signal."""
    g.data = g.data / np.max(np.abs(g.data))
    return g


def average_merge(branches):
    """Average multiple branches."""
    result = GraphData()
    arrays = [b.fft for b in branches if hasattr(b, 'fft')]
    if arrays:
        result.fft = np.mean(arrays, axis=0)
    return result


if __name__ == "__main__":
    main()
