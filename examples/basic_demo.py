"""
Basic Demo - Core sigexec features with inline blocks
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigexec import Graph, GraphData

print("="*60)
print("BASIC SIGEXEC DEMO")
print("="*60)

# Define simple operations inline
def multiply(gdata):
    """Multiply data by 2."""
    gdata.data = gdata.data * 2
    gdata.multiplied = True
    return gdata

def add_offset(gdata):
    """Add an offset."""
    gdata.data = gdata.data + 10
    gdata.offset_applied = True
    return gdata

def compute_stats(gdata):
    """Compute statistics."""
    gdata.mean = gdata.data.mean()
    gdata.max = gdata.data.max()
    return gdata

# Create and run a simple graph
print("\n1. Simple Sequential Graph")
print("-" * 40)

result = (Graph("BasicProcessing")
    .add(multiply, name="Multiply")
    .add(add_offset, name="AddOffset")
    .add(compute_stats, name="ComputeStats")
    .run(GraphData(data=[1, 2, 3, 4, 5]), verbose=True))

print(f"\nResult data: {result.data}")
print(f"Mean: {result.mean:.2f}, Max: {result.max:.2f}")
print(f"Ports: {list(result.ports.keys())}")

# Show the execution graph visualization
print("\n2. Execution Graph Visualization")
print("-" * 40)

graph = (Graph("BasicProcessing")
    .add(multiply, name="Multiply")
    .add(add_offset, name="AddOffset") 
    .add(compute_stats, name="ComputeStats"))

print(graph.to_mermaid())

print("\n✓ Basic demo completed!")
