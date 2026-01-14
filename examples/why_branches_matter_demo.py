"""
Real-world example showing why branches are necessary.

This demonstrates the problem: when you write blocks without knowledge of the
graph context, you need branch isolation to prevent port collisions.
"""

from sigexec import Graph, GraphData
import numpy as np

def demo_without_branches_problem():
    """Show the problem that branches solve."""
    print("=" * 70)
    print("THE PROBLEM: Without branches, blocks must know their context")
    print("=" * 70)
    
    def source(data: GraphData) -> GraphData:
        """Generate data."""
        data.data = np.array([1, 2, 3, 4, 5])
        return data
    
    def lowpass_filter(data: GraphData) -> GraphData:
        """Apply lowpass filter - would use port 'filtered'."""
        data.set('filtered', data.data * 0.5)
        data.set('filter_type', 'lowpass')
        return data
    
    def highpass_filter(data: GraphData) -> GraphData:
        """Apply highpass filter - also wants to use port 'filtered'!"""
        data.set('filtered', data.data * 2.0)
        data.set('filter_type', 'highpass')
        return data
    
    # WITHOUT branches, if we wanted both filters, we'd need to:
    # 1. Rename ports manually ('filtered_lowpass', 'filtered_highpass')
    # 2. Make blocks context-aware (know which variant they're in)
    # 3. Use complex port naming schemes
    
    print("\nWithout branches, blocks must coordinate port names manually")
    print("This breaks the 'write once, use anywhere' principle")
    print()


def demo_with_branches_solution():
    """Show how branches solve the problem."""
    print("=" * 70)
    print("THE SOLUTION: Branches provide isolated port namespaces")
    print("=" * 70)
    
    def source(data: GraphData) -> GraphData:
        """Generate data."""
        data.data = np.array([1, 2, 3, 4, 5])
        data.set('original', data.data.copy())
        return data
    
    def lowpass_filter(data: GraphData) -> GraphData:
        """Apply lowpass filter."""
        data.set('filtered', data.data * 0.5)
        data.set('filter_type', 'lowpass')
        return data
    
    def highpass_filter(data: GraphData) -> GraphData:
        """Apply highpass filter - same port names, no collision!"""
        data.set('filtered', data.data * 2.0)
        data.set('filter_type', 'highpass')
        return data
    
    def analyze(data: GraphData) -> GraphData:
        """Analyze filtered data."""
        filtered = data.get('filtered')
        data.set('mean', float(np.mean(filtered)))
        data.set('std', float(np.std(filtered)))
        return data
    
    def merge_filters(branches: dict) -> GraphData:
        """Compare filter outputs."""
        result = GraphData()
        result.data = branches['lowpass'].data  # Use one as primary
        
        # Store comparisons
        for name, branch_data in branches.items():
            result.set(f'{name}_mean', float(np.mean(branch_data.get('filtered'))))
            result.set(f'{name}_type', branch_data.get('filter_type'))
        
        return result
    
    # WITH branches, each filter operates in isolated namespace
    graph = (Graph("Signal Processing")
        .add(source, name="Source")
        .branch(["lowpass", "highpass"])
        .add(lowpass_filter, branch="lowpass")
        .add(highpass_filter, branch="highpass")
        .add(analyze, name="Analyze")  # Runs on both branches
        .merge(merge_filters, branches=['lowpass', 'highpass']))  # Combine results
    
    result = graph.run(GraphData(), verbose=True)
    
    print("\n" + "=" * 70)
    print("Merged results:")
    print("=" * 70)
    print(f"Lowpass mean: {result.get('lowpass_mean'):.2f}, type: {result.get('lowpass_type')}")
    print(f"Highpass mean: {result.get('highpass_mean'):.2f}, type: {result.get('highpass_type')}")
    
    print("\n" + "=" * 70)
    print("Key insight: Both blocks used the same port names ('filtered')")
    print("but they didn't collide because branches isolate namespaces!")
    print("The merge function combined them into a single result.")
    print("=" * 70)
    print()


def demo_practical_radar_example():
    """Real-world radar processing with multiple window types."""


def create_dashboard() -> 'sd.Dashboard':
    """Create a minimal staticdash dashboard showing why branches matter."""
    try:
        import staticdash as sd
    except Exception:
        raise

    from sigexec.diagnostics import plot_pulse_matrix

    dashboard = sd.Dashboard('Why Branches Matter')
    page = sd.Page('why-branches-matter', 'Why Branches Matter')

    # Use the radar window comparison pipeline from the demo
    from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress

    graph = (Graph("Radar Window Comparison")
        .add(LFMGenerator(), name="Generate LFM")
        .add(StackPulses(), name="Stack Pulses")
        .add(RangeCompress(), name="Range FFT")
        .branch(["hann", "hamming", "blackman"])
        .add(DopplerCompress(window='hann'), branch="hann")
        .add(DopplerCompress(window='hamming'), branch="hamming")
        .add(DopplerCompress(window='blackman'), branch="blackman")
        .merge(lambda branches: GraphData(), branches=['hann','hamming','blackman']) )

    # Run graph to collect branch outputs individually
    # We will run the doppler steps individually to get branch pulse matrices
    # Run generator stages first to ensure reference_pulse is available
    base = Graph().add(LFMGenerator()).add(StackPulses())
    base_result = base.run()

    # Now apply range compression + doppler for each window using the generated base
    hann = Graph().input_data(base_result).add(RangeCompress()).add(DopplerCompress(window='hann')).run()
    hamming = Graph().input_data(base_result).add(RangeCompress()).add(DopplerCompress(window='hamming')).run()
    blackman = Graph().input_data(base_result).add(RangeCompress()).add(DopplerCompress(window='blackman')).run()

    page.add_header('Window Comparison (Range-Doppler Maps)', level=1)
    page.add_text('Compare range-doppler maps produced with different window functions')

    page.add_text('Hann window:')
    page.add_plot(plot_pulse_matrix(hann, title='Hann Window'))

    page.add_text('Hamming window:')
    page.add_plot(plot_pulse_matrix(hamming, title='Hamming Window'))

    page.add_text('Blackman window:')
    page.add_plot(plot_pulse_matrix(blackman, title='Blackman Window'))

    dashboard.add_page(page)
    return dashboard
    print("=" * 70)
    print("PRACTICAL EXAMPLE: Comparing window functions in radar processing")
    print("=" * 70)
    
    from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
    
    # Each DopplerCompress block writes to the same ports:
    # - 'doppler_compressed'
    # - 'doppler_frequencies'
    # - 'range_doppler_map'
    #
    # Without branches, we couldn't compare different windows side-by-side!
    
    def merge_windows(branches: dict) -> GraphData:
        """Merge window comparison results."""
        result = GraphData()
        result.data = branches['hann'].data  # Use hann as primary
        
        # Store peak values from each window
        for name, branch_data in branches.items():
            peak = float(np.max(np.abs(branch_data.data)))
            result.set(f'{name}_peak', peak)
        
        result.set('compared_windows', list(branches.keys()))
        return result
    
    graph = (Graph("Radar Window Comparison")
        .add(LFMGenerator(), name="Generate LFM")
        .add(StackPulses(), name="Stack Pulses")
        .add(RangeCompress(), name="Range FFT")
        .branch(["hann", "hamming", "blackman"])
        .add(DopplerCompress(window='hann'), branch="hann")
        .add(DopplerCompress(window='hamming'), branch="hamming")
        .add(DopplerCompress(window='blackman'), branch="blackman")
        .merge(merge_windows, branches=['hann','hamming','blackman']))
    
    result = graph.run(GraphData())
    
    print(f"\nCompared windows: {result.get('compared_windows')}")
    print("Peak values:")
    for window in result.get('compared_windows'):
        print(f"  {window}: {result.get(f'{window}_peak'):.2f}")
    
    print("\n" + "=" * 70)
    print("All three window functions used identical port names,")
    print("but branches kept them isolated, then merge combined them!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    demo_without_branches_problem()
    demo_with_branches_solution()
    demo_practical_radar_example()
    
    print("\n" + "=" * 70)
    print("Summary: Why branches are essential")
    print("=" * 70)
    print("1. Blocks don't need to know graph context")
    print("2. Same port names can be used in parallel paths")
    print("3. Natural way to compare variations side-by-side")
    print("4. Each branch gets its own isolated GraphData copy")
    print("5. Merge function combines branches with custom logic")
    print()
