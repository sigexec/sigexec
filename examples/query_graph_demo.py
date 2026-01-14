"""
Example: Query graph structure without running it.

This shows how to inspect a graph's structure, count operations,
detect branches, and generate visualizations - all before execution.
"""

from sigexec import Graph, GraphData
import numpy as np


def step_a(g):
    g.data = np.array([1, 2, 3])
    return g


def step_b(g):
    g.data = g.data * 2
    return g


def step_c(g):
    g.data = g.data + 1
    return g


def merge_fn(branches):
    result = GraphData()
    result.data = branches[0].data + branches[1].data
    return result


# Build a complex graph
print("Building a complex graph...")
print("=" * 70)

graph = (Graph("Complex Pipeline")
    .add(step_a, name="Load_Data")
    .add(step_b, name="Preprocess")
    .branch(["fast_path", "slow_path"])
    .add(step_c, branch="fast_path", name="Fast_Filter")
    .add(lambda g: step_c(step_c(g)), branch="slow_path", name="Slow_Filter")
    .merge(merge_fn, branches=["fast_path", "slow_path"], name="Combine_Results")
    .add(step_b, name="Final_Processing"))

print(f"Graph name: {graph.name}")
print(f"Number of operations: {len(graph)}")
print(f"Caching enabled: {graph._enable_cache}")
print()

# Query structure
print("Analyzing structure...")
print("-" * 70)
has_branches = any(op.get('type') == 'branch' for op in graph.operations)
has_merges = any(op.get('type') == 'merge' for op in graph.operations)
has_variants = any(op.get('variant_spec') for op in graph.operations)

print(f"Has branches: {has_branches}")
print(f"Has merges: {has_merges}")
print(f"Has variants: {has_variants}")
print()

# List all operations
print("Operations list:")
print("-" * 70)
for i, op in enumerate(graph.operations, 1):
    op_type = op.get('type', 'operation')
    name = op.get('name', f'op{i}')
    branch = op.get('branch')
    
    if op_type == 'branch':
        print(f"  {i}. BRANCH: {op['names']}")
    elif op_type == 'merge':
        print(f"  {i}. MERGE: {name} (branches: {op['branches']})")
    else:
        branch_str = f" [on: {branch}]" if branch else ""
        print(f"  {i}. {name}{branch_str}")
print()

# Generate visualization
print("Mermaid diagram:")
print("-" * 70)
print(graph.to_mermaid())
print()

# Save visualization
graph.visualize("query_example.md")
print("✓ Visualization saved to query_example.md")
print()

# Show summary
print("Summary:")
print("-" * 70)
print(repr(graph))
print()

# Now actually run it to show it works
print("Running the graph...")
print("-" * 70)
result = graph.run(initial_data=GraphData(), verbose=True)
print(f"\nFinal result: {result.data}")
