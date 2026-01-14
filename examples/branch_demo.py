"""
Branch execution demo showing isolated port namespaces.
"""

from sigexec import Graph, GraphData
import numpy as np

def demo_basic_branch():
    """Show basic branch/merge with port isolation."""
    print("=" * 70)
    print("DEMO 1: Basic Branch with Merge Function")
    print("=" * 70)
    
    def source(data: GraphData) -> GraphData:
        """Generate initial data."""
        data.data = np.array([1, 2, 3, 4, 5])
        data.set('generated', True)
        return data
    
    def process_a(data: GraphData) -> GraphData:
        """Process in branch A - uses port 'result'."""
        data.set('result', data.data * 2)
        data.set('method', 'multiply')
        return data
    
    def process_b(data: GraphData) -> GraphData:
        """Process in branch B - also uses port 'result' (no collision!)."""
        data.set('result', data.data + 10)
        data.set('method', 'add')
        return data
    
    def merge_branches(branches: dict) -> GraphData:
        """Merge function that combines branch results."""
        # Get results from both branches
        result_a = branches['path_a'].get('result')
        result_b = branches['path_b'].get('result')
        
        # Create merged output
        merged = GraphData()
        merged.data = np.concatenate([result_a, result_b])
        merged.set('branch_a_result', result_a)
        merged.set('branch_b_result', result_b)
        merged.set('merged', True)
        return merged
    
    # Create graph with branches
    graph = (Graph("Branch Demo")
        .add(source, name="Source")
        .branch(["path_a", "path_b"])
        .add(process_a, name="Process A", branch="path_a")
        .add(process_b, name="Process B", branch="path_b")
        .merge(merge_branches, branches=['path_a','path_b']))
    
    # Run and check results
    result = graph.run(GraphData())
    
    print(f"\nMerged result shape: {result.data.shape}")
    print(f"Branch A results: {result.get('branch_a_result')}")
    print(f"Branch B results: {result.get('branch_b_result')}")
    print(f"Combined data: {result.data}")
    print()


def demo_merge_keep():
    """Show averaging results from multiple branches."""
    print("=" * 70)
    print("DEMO 2: Averaging Multiple Filter Outputs")
    print("=" * 70)
    
    def gen(data: GraphData) -> GraphData:
        data.data = np.random.randn(100)
        data.set('generated', True)
        return data
    
    def filter_a(data: GraphData) -> GraphData:
        # Smooth with factor 0.3
        data.data = data.data * 0.3
        data.set('filter_type', 'smooth_low')
        return data
    
    def filter_b(data: GraphData) -> GraphData:
        # Smooth with factor 0.5
        data.data = data.data * 0.5
        data.set('filter_type', 'smooth_mid')
        return data
    
    def filter_c(data: GraphData) -> GraphData:
        # Smooth with factor 0.7
        data.data = data.data * 0.7
        data.set('filter_type', 'smooth_high')
        return data
    
    def average_merge(branches: dict) -> GraphData:
        """Average the filtered data from all branches."""
        arrays = [b.data for b in branches.values()]
        
        result = GraphData()
        result.data = np.mean(arrays, axis=0)
        result.set('merged_from', list(branches.keys()))
        result.set('num_branches', len(branches))
        return result
    
    graph = (Graph("Filter Averaging")
        .add(gen, name="Generate")
        .branch(["a", "b", "c"])
        .add(filter_a, branch="a")
        .add(filter_b, branch="b")
        .add(filter_c, branch="c")
        .merge(average_merge, branches=['a','b','c']))
    
    result = graph.run(GraphData())
    
    print(f"\nMerged from branches: {result.get('merged_from')}")
    print(f"Number of branches averaged: {result.get('num_branches')}")
    print(f"Result shape: {result.data.shape}")
    print()


if __name__ == "__main__":
    demo_basic_branch()
    demo_merge_keep()
    
    print("\n" + "=" * 70)
    print("Branch/Merge Summary:")
    print("=" * 70)
    print("- .branch([names]) creates isolated port namespaces")
    print("- .add(func, branch='name') adds operation to specific branch")
    print("- .merge(merge_func, branches=[...]) combines branches with custom logic (branches must be specified)")
    print("- merge_func receives Dict[branch_name, GraphData]")
    print("- merge_func must return a single GraphData")
    print()
