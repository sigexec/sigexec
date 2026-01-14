"""
Port Flow Visualization Demo

Demonstrates how the visualization shows actual port (data attribute) flow
through a graph with branches, including:
- Operations that create multiple ports
- Operations that only access some ports
- Operations that add new ports mid-pipeline
"""

from sigexec import Graph, GraphData
import numpy as np


def load_data(g):
    """Creates initial ports: data, sample_rate, timestamps"""
    g.data = np.random.randn(1000)
    g.sample_rate = 1000.0
    g.timestamps = np.arange(1000) / 1000.0
    return g


def calibrate(g):
    """Uses all three initial ports"""
    g.data = g.data * g.sample_rate / 1000
    return g


def fast_filter(g):
    """Only needs data and sample_rate"""
    g.data = g.data * 0.9
    return g


def slow_filter(g):
    """Only needs data and timestamps"""
    for i, t in enumerate(g.timestamps):
        g.data[i] = g.data[i] * (1 + t * 0.1)
    return g


def annotate(g):
    """Adds a new 'metadata' port that wasn't in the input"""
    g.metadata = {"processed": True, "filter": "slow"}
    return g


def merge_results(branches):
    """Combines results from both branches"""
    result = branches['fast_path'].copy()
    # Get metadata from slow path if available
    if hasattr(branches['slow_path'], 'metadata'):
        result.metadata = branches['slow_path'].metadata
    # Get timestamps from slow path
    result.timestamps = branches['slow_path'].timestamps
    return result


def final_analysis(g):
    """Uses all available ports"""
    print(f"Data shape: {g.data.shape}")
    print(f"Sample rate: {g.sample_rate}")
    print(f"Timestamps: {len(g.timestamps)}")
    if hasattr(g, 'metadata'):
        print(f"Metadata: {g.metadata}")
    return g


def main():
    print("=" * 70)
    print("Port Flow Visualization Demo")
    print("=" * 70)
    print()
    
    # Create empty sample for port analysis (no actual data needed)
    sample = GraphData()
    
    # Build the graph
    graph = (
        Graph("PortFlowDemo")
        .add(load_data, name="Load Data")
        .add(calibrate, name="Calibrate")
        .branch(["fast_path", "slow_path"])
        .add(fast_filter, name="Fast Filter", branch="fast_path")
        .add(slow_filter, name="Slow Filter", branch="slow_path")
        .add(annotate, name="Annotate", branch="slow_path")
        .merge(merge_results, branches=["fast_path", "slow_path"], name="Merge Results")
        .add(final_analysis, name="Final Analysis")
    )
    
    print("Graph Visualization WITH Port Information:")
    print("-" * 70)
    print(graph.to_mermaid(show_ports=True, sample_input=sample))
    print()
    
    print("Graph Visualization WITHOUT Port Information:")
    print("-" * 70)
    print(graph.to_mermaid(show_ports=False))
    print()
    
    print("Key Observations:")
    print("-" * 70)
    print("1. Load Data creates 3 ports: data, sample_rate, timestamps")
    print("2. Fast Filter only uses data and sample_rate")
    print("3. Slow Filter only uses data and timestamps")
    print("4. Annotate adds a NEW port 'metadata' that wasn't in inputs")
    print("5. Merge Results combines ports from both branches")
    print("6. Final Analysis receives all ports from the merged result")
    print()
    
    # Actually run the graph to show it works
    print("Running the graph:")
    print("-" * 70)
    initial_data = GraphData()  # Start with empty GraphData
    result = graph.run(initial_data)
    print()
    print(f"Available ports in result: {list(result.ports.keys())}")
    

if __name__ == "__main__":
    main()
