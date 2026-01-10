"""
Example: Pipeline Memoization

This example demonstrates how memoization automatically caches results
from common pipeline stages, avoiding redundant computation.
"""

import numpy as np
import time
from sigchain import Pipeline
from sigchain.blocks import (
    LFMGenerator,
    StackPulses,
    RangeCompress,
    DopplerCompress
)


def example_basic_memoization():
    """
    Example 1: Basic memoization with branching.
    
    When you branch a pipeline, common stages are only executed once!
    """
    print("=" * 70)
    print("Example 1: Basic Memoization")
    print("=" * 70)
    
    # Build base pipeline (common stages)
    print("\nDefining base pipeline...")
    base = (Pipeline("Base")
        .add(LFMGenerator(num_pulses=64), name="Generate")
        .add(StackPulses(), name="Stack")
        .add(RangeCompress(), name="RangeCompress")
    )
    
    print(f"  {base}")
    
    # Create branches with different final stages
    print("\nCreating branches...")
    branch1 = base.branch("Branch1").add(DopplerCompress(window='hann'), name="Doppler_Hann")
    branch2 = base.branch("Branch2").add(DopplerCompress(window='hamming'), name="Doppler_Hamming")
    branch3 = base.branch("Branch3").add(DopplerCompress(window='none'), name="Doppler_Rect")
    
    print(f"  {branch1}")
    print(f"  {branch2}")
    print(f"  {branch3}")
    
    # Run first branch
    print("\n Running Branch1 (executes all stages)...")
    result1 = branch1.run(verbose=True)
    
    # Run second branch - common stages are CACHED!
    print("\n  Running Branch2 (reuses cached stages!)...")
    result2 = branch2.run(verbose=True)
    
    # Run third branch - also uses cache!
    print("\n  Running Branch3 (also reuses cache!)...")
    result3 = branch3.run(verbose=True)
    
    print("\n✓ Common stages (Generate, Stack, RangeCompress) ran only ONCE!")
    print("  Branch-specific stages (Doppler) ran for each branch.")


def example_composition():
    """
    Example 2: Compose pipelines naturally.
    
    Define multiple pipelines that share common operations.
    They automatically share cached results!
    """
    print("\n" + "=" * 70)
    print("Example 2: Pipeline Composition with Memoization")
    print("=" * 70)
    
    # Define separate pipelines that share initial stages
    print("\nDefining pipeline 1...")
    pipeline1 = (Pipeline("WithHann")
        .add(LFMGenerator(num_pulses=64), name="Generate")
        .add(StackPulses(), name="Stack")
        .add(RangeCompress(), name="RangeCompress")
        .add(DopplerCompress(window='hann'), name="Doppler_Hann")
    )
    
    print("\nDefining pipeline 2 (shares first 3 stages)...")
    pipeline2 = (Pipeline("WithHamming")
        .add(LFMGenerator(num_pulses=64), name="Generate")
        .add(StackPulses(), name="Stack")
        .add(RangeCompress(), name="RangeCompress")
        .add(DopplerCompress(window='hamming'), name="Doppler_Hamming")
    )
    
    print("\nRunning pipeline 1...")
    result1 = pipeline1.run(verbose=True)
    
    print("\nRunning pipeline 2 (uses cached results!)...")
    result2 = pipeline2.run(verbose=True)
    
    print("\n✓ Pipeline 2 only executed its unique stage (Doppler_Hamming)!")
    print("  The first 3 stages were retrieved from cache.")


def example_variants_with_memoization():
    """
    Example 3: Using variants() for parameter sweeps.
    
    variants() automatically uses memoization, so base stages run once!
    """
    print("\n" + "=" * 70)
    print("Example 3: Variants with Automatic Memoization")
    print("=" * 70)
    
    # Build base
    base = (Pipeline("VariantBase")
        .add(LFMGenerator(num_pulses=64), name="Generate")
        .add(StackPulses(), name="Stack")
        .add(RangeCompress(), name="RangeCompress")
    )
    
    print("\nTrying multiple window functions...")
    print("(Base stages run once, then each variant runs its own configuration)")
    print()
    
    # Test multiple configurations - base runs once!
    results = base.variants(
        operation_factory=lambda window: DopplerCompress(window=window),
        configs=['hann', 'hamming', 'blackman', 'bartlett'],
        names=['Hann', 'Hamming', 'Blackman', 'Bartlett']
    )
    
    print(f"\n✓ Got {len(results)} results!")
    print("  Base pipeline stages (Generate, Stack, RangeCompress) executed once")
    print("  Each variant's Doppler stage executed separately")
    
    # Compare results
    print("\nPeak magnitudes:")
    for i, (name, result) in enumerate(zip(['Hann', 'Hamming', 'Blackman', 'Bartlett'], results)):
        peak = np.max(np.abs(result.data))
        print(f"  {name:12s}: {peak:7.2f}")


def example_manual_cache_control():
    """
    Example 4: Manual cache control.
    
    You can disable caching if needed, or clear the cache manually.
    """
    print("\n" + "=" * 70)
    print("Example 4: Manual Cache Control")
    print("=" * 70)
    
    # Create pipeline with cache disabled
    print("\nPipeline with caching DISABLED:")
    no_cache = Pipeline("NoCache", enable_cache=False)
    no_cache.add(LFMGenerator(num_pulses=32), name="Generate")
    no_cache.add(StackPulses(), name="Stack")
    
    branch1 = no_cache.branch("Branch1").add(RangeCompress(), name="Compress")
    branch2 = no_cache.branch("Branch2").add(RangeCompress(), name="Compress")
    
    print("  Running branch1...")
    branch1.run(verbose=True)
    
    print("\n  Running branch2 (will re-execute everything)...")
    branch2.run(verbose=True)
    
    print("\n  Without caching, all stages execute for each branch.")
    
    # Clear cache
    print("\nClearing global cache...")
    Pipeline.clear_cache(Pipeline())
    print("  Cache cleared!")


def example_timing_comparison():
    """
    Example 5: Demonstrate performance benefit of memoization.
    """
    print("\n" + "=" * 70)
    print("Example 5: Performance Benefit")
    print("=" * 70)
    
    # Larger problem for timing
    base = (Pipeline("Timing")
        .add(LFMGenerator(num_pulses=128, pulse_duration=10e-6, sample_rate=10e6), name="Generate")
        .add(StackPulses(), name="Stack")
        .add(RangeCompress(), name="RangeCompress")
    )
    
    # Create multiple branches
    branches = [
        base.branch(f"Branch{i}").add(DopplerCompress(window='hann'), name=f"Doppler{i}")
        for i in range(5)
    ]
    
    # Time execution with cache
    print("\nWith memoization:")
    start = time.time()
    for i, branch in enumerate(branches):
        branch.run(verbose=False)
    with_cache_time = time.time() - start
    print(f"  Executed {len(branches)} branches in {with_cache_time:.3f}s")
    
    # Clear cache and time without
    Pipeline.clear_cache(Pipeline())
    
    print("\nWithout memoization (cache disabled):")
    branches_no_cache = [
        Pipeline(f"NoCacheBranch{i}", enable_cache=False)
        .add(LFMGenerator(num_pulses=128, pulse_duration=10e-6, sample_rate=10e6), name="Generate")
        .add(StackPulses(), name="Stack")
        .add(RangeCompress(), name="RangeCompress")
        .add(DopplerCompress(window='hann'), name=f"Doppler{i}")
        for i in range(5)
    ]
    
    start = time.time()
    for branch in branches_no_cache:
        branch.run(verbose=False)
    without_cache_time = time.time() - start
    print(f"  Executed {len(branches_no_cache)} branches in {without_cache_time:.3f}s")
    
    speedup = without_cache_time / with_cache_time
    print(f"\n✓ Memoization speedup: {speedup:.1f}x faster!")


def main():
    """Run all examples."""
    example_basic_memoization()
    example_composition()
    example_variants_with_memoization()
    example_manual_cache_control()
    example_timing_comparison()
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("""
Memoization Benefits:
- Common stages execute only once across branches
- Automatic caching based on operation sequence
- Significant performance improvement for parameter sweeps
- Natural composition of pipelines
- Can be disabled if needed (enable_cache=False)

Best Practices:
- Define base pipeline with common stages
- Use .branch() to create variants
- Use .variants() for parameter sweeps
- Results are cached automatically
""")


if __name__ == "__main__":
    main()
