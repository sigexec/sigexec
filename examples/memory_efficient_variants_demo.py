"""
Example: Memory-efficient variant processing with incremental saving.

This demonstrates how to process many variants without accumulating
all results in memory by saving each one as it completes.
"""

import numpy as np
from sigexec import Graph, SignalData


def save_variant_result(params, result):
    """
    Callback that saves each variant result to disk as it completes.
    
    This prevents memory buildup when processing many variants.
    """
    # Create filename from variant names
    variant_names = params['variant']
    filename = f"variant_{'_'.join(str(v).replace('.', 'p') for v in variant_names)}.npy"
    
    # Save to disk
    np.save(filename, result.data)
    
    # Print progress
    print(f"✓ Saved {filename} (shape: {result.shape})")


def main():
    print("=" * 70)
    print("Memory-Efficient Variant Processing Demo")
    print("=" * 70)
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    data = SignalData(data=signal, metadata={'sample_rate': 1000.0})
    
    # Define processing functions
    def apply_window(window_type):
        """Factory for window application."""
        def _apply(sig):
            if window_type == 'hann':
                window = np.hanning(len(sig.data))
            elif window_type == 'hamming':
                window = np.hamming(len(sig.data))
            elif window_type == 'blackman':
                window = np.blackman(len(sig.data))
            else:
                window = np.ones(len(sig.data))
            
            windowed = sig.data * window
            return SignalData(data=windowed, metadata=sig.metadata)
        return _apply
    
    def amplify(gain):
        """Factory for amplification."""
        def _amplify(sig):
            amplified = sig.data * gain
            return SignalData(data=amplified, metadata=sig.metadata)
        return _amplify
    
    # Create graph with multiple variants
    # This creates 3 × 3 = 9 combinations
    graph = (
        Graph("memory_efficient")
        .input_data(data)
        .variants(apply_window, ['hann', 'hamming', 'blackman'], 
                 names=['Hann', 'Hamming', 'Blackman'])
        .variants(amplify, [1.0, 2.0, 5.0],
                 names=['1.0x', '2.0x', '5.0x'])
    )
    
    print(f"\nProcessing {3 * 3} = 9 variant combinations...")
    print("Saving each result to disk as it completes...")
    print()
    
    # Run with callback - results are saved incrementally
    # Setting return_results=False means we don't accumulate in memory
    results = graph.run(
        verbose=False,
        on_variant_complete=save_variant_result,
        return_results=False  # Don't accumulate - saves memory!
    )
    
    print()
    print("=" * 70)
    print("✓ All variants processed and saved!")
    print(f"  Results accumulated in memory: {len(results)}")
    print(f"  (With return_results=False, list is empty to save memory)")
    print()
    print("Files created:")
    import os
    for f in sorted(os.listdir('.')):
        if f.startswith('variant_') and f.endswith('.npy'):
            print(f"  - {f}")
    print()
    
    # Example: If you DO want to keep results in memory for immediate use
    print("Alternative: Keep results in memory for immediate analysis")
    print("=" * 70)
    
    # Smaller example
    small_pipeline = (
        Graph("with_results")
        .input_data(data)
        .variants(amplify, [1.0, 2.0], names=['1x', '2x'])
    )
    
    # Track what was saved via callback AND keep in memory
    saved_files = []
    def track_and_save(params, result):
        filename = f"temp_{'_'.join(params['variant'])}.npy"
        np.save(filename, result.data)
        saved_files.append(filename)
    
    # return_results=True (default) keeps them in memory too
    results_with_memory = small_pipeline.run(
        on_variant_complete=track_and_save,
        return_results=True  # Default - keeps results
    )
    
    print(f"Results in memory: {len(results_with_memory)}")
    print(f"Files saved: {len(saved_files)}")
    
    # Clean up temp files
    for f in saved_files:
        os.remove(f)
    
    print("\n✓ Demo complete!")


if __name__ == '__main__':
    main()
