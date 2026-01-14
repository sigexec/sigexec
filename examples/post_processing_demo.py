"""
Post-Processing Plots Demo

Shows how to save intermediate results and plot them after the fact,
and how to selectively plot variant results.
"""

import numpy as np
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigexec.diagnostics import plot_pulse_matrix, plot_range_profile, plot_range_doppler_map

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False


def create_dashboard() -> sd.Dashboard:
    """
    Create post-processing plots demo dashboard.
    
    Shows how to save intermediate results and plot them after execution,
    and how to selectively plot variant results.
    
    Returns:
        Dashboard object ready to be added to a Directory
    """
    dashboard = sd.Dashboard('Post-Processing Plots')
    page = sd.Page('post-plots', 'Plotting After Execution')
    
    page.add_header("Plotting After Graph Execution", level=1)
    page.add_text("""
    All plotting functions are pure functions that work on SignalData objects.
    This means you can plot whenever you want - during graph execution with .tap(),
    or after the fact with saved results.
    """)
    
    # Demo 1: Intermediate results
    page.add_header("Method 1: Save and Plot Intermediate Results", level=2)
    page.add_text("""
    Use `save_intermediate=True` to capture the output of every graph stage.
    Then use `get_intermediate_results()` to retrieve them and plot any stage you want.
    """)
    
    code_example_1 = """
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigexec.diagnostics.visualization import (
    plot_timeseries, plot_pulse_matrix, 
    plot_range_profile, plot_range_doppler_map
)

# Build and run graph, saving all intermediate results
graph = (Graph("Radar")
    .add(LFMGenerator(num_pulses=64, target_delay=2e-6, target_doppler=200.0))
    .add(StackPulses())
    .add(RangeCompress(window='hamming'))
    .add(DopplerCompress(window='hann'))
)

# Run with intermediate results saved
result = graph.run(save_intermediate=True)
intermediates = graph.get_intermediate_results()

# Now plot any stage you want!
fig1 = plot_timeseries(intermediates[0], title="Stage 1: Generated Signal")
fig2 = plot_pulse_matrix(intermediates[1], title="Stage 2: Stacked Pulses")
fig3 = plot_range_profile(intermediates[2], title="Stage 3: After Range Compression")
fig4 = plot_range_doppler_map(intermediates[3], title="Stage 4: Final Range-Doppler Map")

# Or plot specific stages based on what you're debugging
if need_to_check_range_compression:
    fig = plot_range_profile(intermediates[2], title="Range Compression Check")
    page.add_plot(fig)
"""
    page.add_syntax(code_example_1, language='python')
    
    # Actually run this demo
    page.add_header("Live Example: Intermediate Results", level=3)
    
    # Run the generator stages first to ensure required ports (e.g., reference_pulse)
    base = Graph().add(LFMGenerator(
            num_pulses=32,
            target_delay=2e-6,
            target_doppler=200.0,
            noise_power=0.01,
        )).add(StackPulses())

    # Run base stages and collect their intermediates so we have full-stage history
    base_result = base.run(save_intermediate=True)
    base_intermediates = base.get_intermediate_results()

    graph = (Graph("IntermediateDemo")
        .input_data(base_result)
        .add(RangeCompress(window='hamming', oversample_factor=2))
        .add(DopplerCompress(window='hann', oversample_factor=2))
    )

    result = graph.run(save_intermediate=True)
    # Combine base + later intermediates so indices match the original expectations
    intermediates = base_intermediates + graph.get_intermediate_results()
    
    # Plot selected stages
    fig1 = plot_pulse_matrix(intermediates[1], title="After Stacking", height=400)
    page.add_plot(fig1, height=400)
    
    fig2 = plot_range_profile(intermediates[2], title="After Range Compression", height=400)
    page.add_plot(fig2, height=400)
    
    fig3 = plot_range_doppler_map(intermediates[3], title="Final Result", 
                                   height=500, use_db=True, mark_target=True)
    page.add_plot(fig3, height=500)
    
    # Demo 2: Filtering variant results
    page.add_header("Method 2: Filter and Plot Variant Results", level=2)
    page.add_text("""
    When exploring parameter spaces with variants, you can selectively plot
    only the combinations you're interested in after running all of them.
    """)
    
    code_example_2 = """
from sigexec import Graph
from sigexec.blocks import LFMGenerator, StackPulses, RangeCompress, DopplerCompress
from sigexec.diagnostics.visualization import plot_range_doppler_map

# Run all variant combinations
results = (Graph("Radar")
    .add(LFMGenerator(num_pulses=64, target_delay=2e-6, target_doppler=200.0))
    .add(StackPulses())
    .variants(lambda w: RangeCompress(window=w), 
              ['hamming', 'hann', 'blackman'],
              names=['Hamming', 'Hann', 'Blackman'])
    .variants(lambda w: DopplerCompress(window=w), 
              ['hamming', 'hann'],
              names=['Hamming', 'Hann'])
    .run()
)

# Now selectively plot what you want:

# Example 1: Plot only Hamming range window results
for params, result in results:
    if params['variant'][0] == 'Hamming':
        title = f"Range: {params['variant'][0]}, Doppler: {params['variant'][1]}"
        fig = plot_range_doppler_map(result, title=title)
        page.add_plot(fig)

# Example 2: Compare two specific combinations
combo1 = next((r for p, r in results if p['variant'] == ['Hamming', 'Hamming']), None)
combo2 = next((r for p, r in results if p['variant'] == ['Blackman', 'Hann']), None)

fig1 = plot_range_doppler_map(combo1, title="Hamming + Hamming")
fig2 = plot_range_doppler_map(combo2, title="Blackman + Hann")

# Example 3: Find and plot best performing combination
best_params, best_result = max(results, 
                                key=lambda x: np.max(np.abs(x[1].data)))
title = f"Best: {' + '.join(best_params['variant'])}"
fig = plot_range_doppler_map(best_result, title=title)
"""
    page.add_syntax(code_example_2, language='python')
    
    # Actually run filtered variant demo
    page.add_header("Live Example: Filtered Variant Results", level=3)
    page.add_text("Running 2×2 variant combinations and plotting only selected results:")
    
    results = (Graph("FilteredDemo")
        .add(LFMGenerator(
            num_pulses=32,
            target_delay=2e-6,
            target_doppler=200.0,
            noise_power=0.01,
        ))
        .add(StackPulses())
        .variants(lambda w: RangeCompress(window=w, oversample_factor=2), 
                 ['hamming', 'blackman'],
                 names=['Hamming', 'Blackman'])
        .variants(lambda w: DopplerCompress(window=w, oversample_factor=2), 
                 ['hamming', 'hann'],
                 names=['Hamming', 'Hann'])
        .run()
    )
    
    # Only plot Hamming range window results
    page.add_header("Filtered Results: Only Hamming Range Window", level=4)
    for params, result in results:
        if params['variant'][0] == 'Hamming':
            title = f"Range: {params['variant'][0]}, Doppler: {params['variant'][1]}"
            fig = plot_range_doppler_map(result, title=title, height=450, 
                                         use_db=True, mark_target=True)
            page.add_plot(fig, height=450)
    
    # Find best combination
    page.add_header("Best Performing Combination", level=4)
    best_params, best_result = max(results, 
                                    key=lambda x: np.max(np.abs(x[1].data)))
    title = f"Best SNR: {' + '.join(best_params['variant'])}"
    fig = plot_range_doppler_map(best_result, title=title, height=450,
                                 use_db=True, mark_target=True)
    page.add_plot(fig, height=450)
    
    page.add_header("Summary", level=2)
    page.add_text("""
    Key takeaways:
    
    1. **Intermediate Results**: Use `save_intermediate=True` and `get_intermediate_results()` 
       to capture and plot any stage of your graph after execution.
       
    2. **Variant Filtering**: Run all variants, then selectively plot or analyze only the 
       combinations you're interested in.
       
    3. **Pure Functions**: All plot functions work on SignalData objects, so you have complete 
       flexibility in when and how you visualize your results.
       
    4. **Interactive Exploration**: You can load results, explore them, and create new 
       visualizations without re-running expensive processing steps.
    """)
    
    dashboard.add_page(page)
    return dashboard


if __name__ == "__main__":
    if not STATICDASH_AVAILABLE:
        print("staticdash not available. Install with: pip install staticdash")
    else:
        print("Creating post-processing plots dashboard...")
        dashboard = create_dashboard()
        
        # Publish standalone
        directory = sd.Directory(title='Post-Processing Plots', page_width=1000)
        directory.add_dashboard(dashboard, slug='post-processing-plots')
        directory.publish('staticdash')
        
        print("✓ Dashboard published to staticdash/")
        print("  Open staticdash/index.html in a web browser")
