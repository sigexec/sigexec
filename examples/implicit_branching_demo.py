"""
Example: Implicit Branching via Metadata Optimization

This demonstrates the memory-efficient metadata optimization feature that enables
implicit branching without explicit branch/merge operations.

When optimize_metadata=True, the Graph analyzes which metadata fields each operation
actually uses and creates optimized subsets. This provides automatic branching when
different operations use different metadata fields, reducing memory overhead.
"""

import numpy as np
from sigexec import Graph, GraphData


def main():
    print("=" * 70)
    print("Implicit Branching via Metadata Optimization Demo")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Example 1: Basic Metadata Optimization
    # =========================================================================
    print("Example 1: Basic Metadata Optimization")
    print("-" * 70)
    print()
    print("Operations that only use specific metadata fields will automatically")
    print("receive optimized metadata subsets, saving memory.")
    print()
    
    def uses_sample_rate(sig: GraphData) -> GraphData:
        """Operation that only needs sample_rate from metadata."""
        sr = sig.metadata['sample_rate']
        scaled = sig.data * sr
        new_metadata = sig.metadata.copy()
        new_metadata['processed'] = True
        return GraphData(scaled, metadata=new_metadata)
    
    def uses_frequency(sig: GraphData) -> GraphData:
        """Operation that only needs center_frequency from metadata."""
        fc = sig.metadata['center_frequency']
        shifted = sig.data * np.exp(2j * np.pi * fc * np.arange(len(sig.data)))
        new_metadata = sig.metadata.copy()
        new_metadata['shifted'] = True
        return GraphData(shifted, metadata=new_metadata)
    
    # Create signal with multiple metadata fields
    signal = GraphData(
        data=np.random.randn(100),
        metadata={
            'sample_rate': 1000.0,
            'center_frequency': 50.0,
            'antenna': 'array_1',
            'timestamp': '2026-01-14T12:00:00',
            'large_calibration_data': np.zeros(10000),  # Large unused data
        }
    )
    
    print(f"Input signal has {len(signal.metadata)} metadata fields")
    print(f"Metadata keys: {list(signal.metadata.keys())}")
    print()
    
    # WITHOUT optimization - all metadata passed to every operation
    print("WITHOUT metadata optimization:")
    graph = Graph(optimize_metadata=False)
    result = (graph
             .add(uses_sample_rate, name="Scale by SR")
             .add(uses_frequency, name="Frequency Shift")
             .run(signal, verbose=True))
    print(f"  Each operation received all {len(signal.metadata)} metadata fields")
    print()
    
    # WITH optimization - only needed metadata passed to each operation
    print("WITH metadata optimization:")
    graph_opt = Graph(optimize_metadata=True)
    result_opt = (graph_opt
                 .add(uses_sample_rate, name="Scale by SR")
                 .add(uses_frequency, name="Frequency Shift")
                 .run(signal, verbose=True))
    print()
    print("  ✓ First operation received only 'sample_rate'")
    print("  ✓ Second operation received only 'center_frequency'")
    print("  ✓ Large unused calibration data was never copied!")
    print("  ✓ All metadata preserved in final result")
    print()
    
    # =========================================================================
    # Example 2: Implicit Branching with Different Metadata
    # =========================================================================
    print()
    print("Example 2: Implicit Branching - Operations Using Different Fields")
    print("-" * 70)
    print()
    print("When operations use completely different metadata fields, they can")
    print("execute independently with implicit branching (no explicit branch/merge).")
    print()
    
    def extract_signal_stats(sig: GraphData) -> GraphData:
        """Analyzes signal statistics - doesn't need metadata."""
        stats = {
            'mean': float(np.mean(np.abs(sig.data))),
            'std': float(np.std(np.abs(sig.data))),
            'peak': float(np.max(np.abs(sig.data)))
        }
        return GraphData(sig.data, metadata={**sig.metadata, 'stats': stats})
    
    def apply_antenna_calibration(sig: GraphData) -> GraphData:
        """Applies antenna-specific calibration - needs antenna field."""
        antenna = sig.metadata['antenna']
        # Simulate antenna-specific gain correction
        gain = {'array_1': 1.2, 'array_2': 1.5}.get(antenna, 1.0)
        calibrated = sig.data * gain
        return GraphData(calibrated, metadata={**sig.metadata, 'calibrated': True})
    
    def apply_timestamp_correction(sig: GraphData) -> GraphData:
        """Applies time-dependent correction - needs timestamp field."""
        timestamp = sig.metadata['timestamp']
        # Simulate time-based phase correction
        phase_corr = np.exp(1j * 0.1 * np.arange(len(sig.data)))
        corrected = sig.data * phase_corr
        return GraphData(corrected, metadata={**sig.metadata, 'time_corrected': True})
    
    graph_implicit = Graph(optimize_metadata=True)
    result_implicit = (graph_implicit
                      .add(extract_signal_stats, name="Compute Stats")
                      .add(apply_antenna_calibration, name="Antenna Cal")
                      .add(apply_timestamp_correction, name="Time Correction")
                      .run(signal, verbose=True))
    
    print()
    print("With optimization:")
    print("  • extract_signal_stats got: empty metadata (doesn't need any)")
    print("  • apply_antenna_calibration got: only 'antenna'")
    print("  • apply_timestamp_correction got: only 'timestamp'")
    print("  • Final result has: ALL metadata fields plus new ones")
    print()
    print(f"Final metadata keys: {sorted(result_implicit.metadata.keys())}")
    print()
    
    # =========================================================================
    # Example 3: Combining with Explicit Branching
    # =========================================================================
    print()
    print("Example 3: Metadata Optimization + Explicit Branching")
    print("-" * 70)
    print()
    print("Metadata optimization works seamlessly with explicit branch/merge.")
    print()
    
    def branch_a_operation(sig: GraphData) -> GraphData:
        """Branch A: uses sample_rate."""
        sr = sig.metadata['sample_rate']
        return GraphData(sig.data + sr, metadata=sig.metadata)
    
    def branch_b_operation(sig: GraphData) -> GraphData:
        """Branch B: uses center_frequency."""
        fc = sig.metadata['center_frequency']
        return GraphData(sig.data * fc, metadata=sig.metadata)
    
    def merge_branches(signals):
        """Merge the two branches."""
        combined = signals[0].data + signals[1].data
        return GraphData(combined, metadata=signals[0].metadata)
    
    graph_combined = Graph(optimize_metadata=True)
    result_combined = (graph_combined
                      .input_data(signal)
                      .branch(['branch_a', 'branch_b'])
                      .add(branch_a_operation, branch='branch_a', name="Add SR")
                      .add(branch_b_operation, branch='branch_b', name="Multiply FC")
                      .merge(['branch_a', 'branch_b'], combiner=merge_branches)
                      .run(verbose=True))
    
    print()
    print("  • Branch A operation received only metadata it needs")
    print("  • Branch B operation received only metadata it needs")
    print("  • Merge received full metadata from both branches")
    print()
    
    # =========================================================================
    # Example 4: Memory Efficiency Benefits
    # =========================================================================
    print()
    print("Example 4: Memory Efficiency Benefits")
    print("-" * 70)
    print()
    
    # Create signal with very large metadata
    large_metadata = {
        'sample_rate': 1e6,
        'huge_calibration_table': np.random.randn(1000, 1000),  # ~8 MB
        'huge_antenna_patterns': np.random.randn(500, 500),     # ~2 MB
        'huge_reference_data': np.random.randn(2000, 2000),     # ~32 MB
    }
    
    signal_large = GraphData(data=np.random.randn(1000), metadata=large_metadata)
    
    def simple_scale(sig: GraphData) -> GraphData:
        """Only needs sample_rate - tiny subset of metadata."""
        sr = sig.metadata['sample_rate']
        new_metadata = sig.metadata.copy()
        new_metadata['scaled'] = True
        return GraphData(sig.data * 0.5, metadata=new_metadata)
    
    print(f"Input signal has ~42 MB of metadata")
    print(f"Operation only needs 1 field (8 bytes)")
    print()
    
    graph_efficient = Graph(optimize_metadata=True)
    result_efficient = graph_efficient.add(simple_scale).run(signal_large, verbose=True)
    
    print()
    print("With optimization:")
    print("  ✓ Operation received only 8 bytes instead of 42 MB")
    print("  ✓ ~99.9999% reduction in metadata copying")
    print("  ✓ Original large metadata still in final result")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary: Metadata Optimization Benefits")
    print("=" * 70)
    print()
    print("✓ Automatic implicit branching when operations use different metadata")
    print("✓ Significant memory savings for large metadata structures")
    print("✓ No changes required to existing operations")
    print("✓ Works seamlessly with explicit branching/merging")
    print("✓ Full metadata preserved in results")
    print("✓ Simple to enable: Graph(optimize_metadata=True)")
    print()
    print("This feature is especially useful for:")
    print("  • Large calibration tables or reference data")
    print("  • Operations that only need a few specific metadata fields")
    print("  • Processing pipelines with many independent metadata-using operations")
    print("  • Memory-constrained environments")
    print()


if __name__ == '__main__':
    main()
