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


# Staticdash dashboard for publishing
def create_dashboard():
    try:
        import staticdash as sd
    except ImportError:
        raise RuntimeError("staticdash is required for dashboard publishing")

    dashboard = sd.Dashboard("Basic sigexec demo")
    page = sd.Page("basic-demo", "Basic sigexec demo")
    page.add_header("Basic sigexec demo", level=1)
    page.add_text("""
    This demo shows a simple sequential graph with multiply, offset, and stats operations.
    """)
    # Show the execution graph
    graph = (Graph("BasicProcessing")
        .add(multiply, name="Multiply")
        .add(add_offset, name="AddOffset")
        .add(compute_stats, name="ComputeStats"))
    page.add_header("Execution Graph", level=2)
    page.add_syntax(graph.to_mermaid(), language="mermaid")
    dashboard.add_page(page)
    return dashboard

print("\n✓ Basic demo completed!")
