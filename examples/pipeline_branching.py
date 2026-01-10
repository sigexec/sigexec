"""
Example: Pipeline Branching and Variants

This example demonstrates:
1. Creating multiple branches from a common base pipeline
2. Testing different parameter configurations (variants)
3. Intermediate plotting at each stage
4. Comparing results from different approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from sigchain import Pipeline
from sigchain.blocks import (
    LFMGenerator,
    StackPulses,
    RangeCompress,
    DopplerCompress
)


def plot_signal(signal_data, title="Signal", filename=None):
    """Helper function to plot signal data."""
    data = signal_data.data
    
    # Convert to magnitude dB
    if np.iscomplexobj(data):
        data_db = 20 * np.log10(np.abs(data) + 1e-10)
    else:
        data_db = data
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data_db, aspect='auto', cmap='jet',
               vmin=np.max(data_db) - 50, vmax=np.max(data_db))
    plt.colorbar(label='Magnitude (dB)')
    plt.title(title)
    plt.xlabel('Range Bin')
    plt.ylabel('Pulse / Doppler Bin')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    plt.close()


def example_1_branching():
    """
    Example 1: Branch pipelines to try different processing paths.
    
    We'll create a base pipeline, then branch it to try different
    window functions for Doppler compression.
    """
    print("=" * 70)
    print("Example 1: Pipeline Branching")
    print("=" * 70)
    
    # Create base pipeline (common stages)
    base = (Pipeline("Base")
        .add(LFMGenerator(num_pulses=64, target_delay=15e-6, target_doppler=500.0))
        .add(StackPulses())
        .add(RangeCompress())
    )
    
    print("\nBase pipeline created:")
    print(f"  {base}")
    
    # Create branches with different window functions
    print("\nCreating branches...")
    
    branch_hann = base.branch(name="Hann").add(
        DopplerCompress(window='hann'),
        name="DopplerHann"
    )
    
    branch_hamming = base.branch(name="Hamming").add(
        DopplerCompress(window='hamming'),
        name="DopplerHamming"
    )
    
    branch_none = base.branch(name="Rectangular").add(
        DopplerCompress(window='none'),
        name="DopplerRect"
    )
    
    print(f"  Branch 1: {branch_hann}")
    print(f"  Branch 2: {branch_hamming}")
    print(f"  Branch 3: {branch_none}")
    
    # Execute all branches
    print("\nExecuting branches...")
    result_hann = branch_hann.run(verbose=True)
    print()
    result_hamming = branch_hamming.run(verbose=True)
    print()
    result_none = branch_none.run(verbose=True)
    
    # Plot results for comparison
    print("\nPlotting results...")
    plot_signal(result_hann, "Range-Doppler Map (Hann Window)", "branch_hann.png")
    plot_signal(result_hamming, "Range-Doppler Map (Hamming Window)", "branch_hamming.png")
    plot_signal(result_none, "Range-Doppler Map (Rectangular Window)", "branch_rect.png")
    
    # Compare peak magnitudes
    print("\nComparison:")
    print(f"  Hann window peak:        {np.max(np.abs(result_hann.data)):.2f}")
    print(f"  Hamming window peak:     {np.max(np.abs(result_hamming.data)):.2f}")
    print(f"  Rectangular window peak: {np.max(np.abs(result_none.data)):.2f}")


def example_2_variants():
    """
    Example 2: Use variants() to automatically test multiple configurations.
    
    This is more concise than manual branching for parameter sweeps.
    """
    print("\n" + "=" * 70)
    print("Example 2: Pipeline Variants (Parameter Sweep)")
    print("=" * 70)
    
    # Build base pipeline
    base = (Pipeline("VariantBase")
        .add(LFMGenerator(num_pulses=64, target_delay=15e-6, target_doppler=500.0))
        .add(StackPulses())
        .add(RangeCompress())
    )
    
    print("\nTesting different window functions using variants()...")
    
    # Test multiple window functions automatically
    windows = ['hann', 'hamming', 'blackman', 'bartlett', 'none']
    
    results = base.variants(
        operation_factory=lambda window: DopplerCompress(window=window),
        configs=windows,
        names=[w.capitalize() for w in windows]
    )
    
    # Compare results
    print("\nResults from variants:")
    for i, (window, result) in enumerate(zip(windows, results)):
        peak = np.max(np.abs(result.data))
        mean = np.mean(np.abs(result.data))
        snr = 20 * np.log10(peak / mean)
        print(f"  {window:12s}: Peak={peak:7.2f}, Mean={mean:7.2f}, SNR={snr:5.2f} dB")
    
    # Plot the best one
    best_idx = np.argmax([np.max(np.abs(r.data)) for r in results])
    print(f"\nBest window function: {windows[best_idx]}")
    plot_signal(results[best_idx], 
                f"Best Result ({windows[best_idx].capitalize()} Window)",
                "variant_best.png")


def example_3_intermediate_plotting():
    """
    Example 3: Add plotting at each stage to visualize the pipeline.
    
    Use .plot() to add visualization without breaking the pipeline flow.
    """
    print("\n" + "=" * 70)
    print("Example 3: Intermediate Plotting")
    print("=" * 70)
    
    print("\nBuilding pipeline with intermediate plots...")
    
    # Create pipeline with plotting at each stage
    result = (Pipeline("PlottedPipeline")
        .add(LFMGenerator(num_pulses=64, target_delay=15e-6, target_doppler=500.0))
        .plot(lambda sig: plot_signal(sig, "Stage 1: Raw Signal", "stage1_raw.png"))
        
        .add(StackPulses())
        .plot(lambda sig: plot_signal(sig, "Stage 2: Stacked Pulses", "stage2_stacked.png"))
        
        .add(RangeCompress())
        .plot(lambda sig: plot_signal(sig, "Stage 3: Range Compressed", "stage3_range.png"))
        
        .add(DopplerCompress(window='hann'))
        .plot(lambda sig: plot_signal(sig, "Stage 4: Range-Doppler Map", "stage4_rdm.png"))
        
        .run(verbose=True)
    )
    
    print("\nAll stages plotted!")


def example_4_advanced_branching():
    """
    Example 4: Advanced branching with different processing approaches.
    
    Test completely different processing paths from the same base.
    """
    print("\n" + "=" * 70)
    print("Example 4: Advanced Branching (Different Processing Paths)")
    print("=" * 70)
    
    # Base: just generate and stack
    base = (Pipeline("AdvancedBase")
        .add(LFMGenerator(num_pulses=64, target_delay=15e-6, target_doppler=500.0))
        .add(StackPulses())
    )
    
    print("\nCreating different processing paths...")
    
    # Path 1: Standard processing
    path1 = (base.branch("Standard")
        .add(RangeCompress())
        .add(DopplerCompress(window='hann'))
    )
    
    # Path 2: Doppler first, then range (non-standard)
    # Note: This won't work well for radar, but demonstrates flexibility
    path2 = (base.branch("Alternate")
        .add(DopplerCompress(window='hann'))
        # Can't do range compress after Doppler, but shows the concept
    )
    
    print(f"  Path 1: {path1}")
    print(f"  Path 2: {path2}")
    
    # Execute
    print("\nExecuting standard path...")
    result1 = path1.run()
    
    print("Executing alternate path...")
    result2 = path2.run()
    
    print(f"\nStandard result peak:  {np.max(np.abs(result1.data)):.2f}")
    print(f"Alternate result peak: {np.max(np.abs(result2.data)):.2f}")


def example_5_save_intermediate():
    """
    Example 5: Save and inspect intermediate results.
    
    Use save_intermediate to capture results at every stage.
    """
    print("\n" + "=" * 70)
    print("Example 5: Save Intermediate Results")
    print("=" * 70)
    
    pipeline = (Pipeline("Intermediate")
        .add(LFMGenerator(num_pulses=32, target_delay=10e-6, target_doppler=300.0))
        .add(StackPulses())
        .add(RangeCompress())
        .add(DopplerCompress(window='hann'))
    )
    
    print("\nRunning pipeline with intermediate results saved...")
    result = pipeline.run(save_intermediate=True, verbose=True)
    
    # Get all intermediate results
    intermediates = pipeline.get_intermediate_results()
    
    print(f"\nCaptured {len(intermediates)} intermediate results:")
    for i, sig in enumerate(intermediates):
        print(f"  Stage {i+1}: shape={sig.shape}, "
              f"peak={np.max(np.abs(sig.data)):.2f}")
    
    # Plot them all
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    stage_names = ["Raw Signal", "Stacked", "Range Compressed", "Range-Doppler Map"]
    
    for i, (sig, ax, name) in enumerate(zip(intermediates, axes, stage_names)):
        data_db = 20 * np.log10(np.abs(sig.data) + 1e-10)
        im = ax.imshow(data_db, aspect='auto', cmap='jet',
                      vmin=np.max(data_db) - 50, vmax=np.max(data_db))
        ax.set_title(name)
        ax.set_xlabel('Range Bin')
        ax.set_ylabel('Pulse/Doppler Bin')
        plt.colorbar(im, ax=ax, label='dB')
    
    plt.tight_layout()
    plt.savefig('intermediate_stages.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: intermediate_stages.png")
    plt.close()


def main():
    """Run all examples."""
    example_1_branching()
    example_2_variants()
    example_3_intermediate_plotting()
    example_4_advanced_branching()
    example_5_save_intermediate()
    
    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
