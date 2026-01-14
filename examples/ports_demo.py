"""
Port-Based Data Flow Demo

Demonstrates the pure port-based architecture where all data flows
through named ports with no privileged "data" port.
"""

import numpy as np
from sigexec import GraphData, Graph, SignalData


def main():
    print("=" * 70)
    print("Port-Based Data Flow Demo")


def create_dashboard() -> 'sd.Dashboard':
    """Create a minimal staticdash dashboard for the ports demo."""
    try:
        import staticdash as sd
    except Exception:
        raise

    from sigexec.diagnostics import plot_timeseries

    dashboard = sd.Dashboard('Ports Demo')
    page = sd.Page('ports', 'Ports Demo')

    # Simple example: generate and scale signal
    t = np.linspace(0, 1, 200)
    signal = np.sin(2 * np.pi * 10 * t)

    gdata = GraphData()
    gdata.signal = signal
    gdata.sample_rate = 100.0

    # Run scaling operation
    def scale_signal(gdata: GraphData) -> GraphData:
        signal = gdata.signal
        sr = gdata.sample_rate
        gdata.signal = signal * sr
        return gdata

    result = Graph().add(lambda g: g).add(scale_signal).run(gdata)

    page.add_header('Port-based data flow', level=1)
    page.add_text('Demonstrates basic port read/write and a scaled signal')

    page.add_text('Original signal:')
    page.add_plot(plot_timeseries(SignalData(data=signal, metadata={'sample_rate':100.0}), title='Original'))
    page.add_text('Scaled signal:')
    page.add_plot(plot_timeseries(SignalData(data=result.signal, metadata={'sample_rate':100.0}), title='Scaled'))

    dashboard.add_page(page)
    return dashboard
    print("=" * 70)
    print()
    
    # ==========================================================================
    # Example 1: Basic Port Usage
    # ==========================================================================
    print("Example 1: Basic Port Reading and Writing")
    print("-" * 70)
    
    def generate_signal(gdata: GraphData) -> GraphData:
        """Generate a signal and write to 'signal' port."""
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 10 * t)
        
        gdata.signal = signal  # Write to port
        gdata.sample_rate = 100.0
        gdata.time = t
        
        print(f"  Generated signal (shape: {signal.shape})")
        return gdata
    
    def scale_signal(gdata: GraphData) -> GraphData:
        """Scale signal by sample rate."""
        # Read from ports - errors if missing
        signal = gdata.signal
        sr = gdata.sample_rate
        
        # Process
        scaled = signal * sr
        
        # Write result back
        gdata.signal = scaled
        print(f"  Scaled signal by {sr}")
        return gdata
    
    # Run pipeline
    result = Graph().add(generate_signal).add(scale_signal).run(GraphData())
    print(f"  Final signal range: [{result.signal.min():.2f}, {result.signal.max():.2f}]")
    print()
    
    # ==========================================================================
    # Example 2: Using get() with Defaults
    # ==========================================================================
    print("Example 2: Safe Port Access with Defaults")
    print("-" * 70)
    
    def process_with_defaults(gdata: GraphData) -> GraphData:
        """Process signal with optional configuration ports."""
        signal = gdata.signal
        
        # Use get() for optional ports with defaults
        gain = gdata.get('gain', 1.0)  # Default to 1.0 if not present
        offset = gdata.get('offset', 0.0)  # Default to 0.0
        
        result = signal * gain + offset
        gdata.signal = result
        
        print(f"  Applied gain={gain}, offset={offset}")
        return gdata
    
    # Without optional ports
    gdata1 = GraphData()
    gdata1.signal = np.array([1, 2, 3])
    result1 = process_with_defaults(gdata1)
    print(f"  Result without options: {result1.signal}")
    
    # With optional ports
    gdata2 = GraphData()
    gdata2.signal = np.array([1, 2, 3])
    gdata2.gain = 2.0
    gdata2.offset = 10.0
    result2 = process_with_defaults(gdata2)
    print(f"  Result with options: {result2.signal}")
    print()
    
    # ==========================================================================
    # Example 3: Multiple Independent Ports
    # ==========================================================================
    print("Example 3: Multiple Independent Data Ports")
    print("-" * 70)
    
    # Clear caches to avoid interference
    Graph._global_cache.clear()
    
    def generate_dual_signals(gdata: GraphData) -> GraphData:
        """Generate two independent signals on different ports."""
        t = np.linspace(0, 1, 50)
        
        gdata.signal_a = np.sin(2 * np.pi * 5 * t)
        gdata.signal_b = np.cos(2 * np.pi * 3 * t)
        gdata.time = t
        
        print(f"  Generated signal_a and signal_b")
        return gdata
    
    def process_signal_a(gdata: GraphData) -> GraphData:
        """Only touches signal_a - doesn't need signal_b."""
        sig_a = gdata.signal_a
        gdata.signal_a = sig_a * 2
        print(f"  Processed signal_a (doubled)")
        return gdata
    
    def process_signal_b(gdata: GraphData) -> GraphData:
        """Only touches signal_b - doesn't need signal_a."""
        sig_b = gdata.signal_b
        gdata.signal_b = sig_b + 1
        print(f"  Processed signal_b (added 1)")
        return gdata
    
    result = (Graph()
             .add(generate_dual_signals)
             .add(process_signal_a)
             .add(process_signal_b)
             .run(GraphData()))
    
    print(f"  Final signal_a range: [{result.signal_a.min():.2f}, {result.signal_a.max():.2f}]")
    print(f"  Final signal_b range: [{result.signal_b.min():.2f}, {result.signal_b.max():.2f}]")
    print()
    
    # ==========================================================================
    # Example 4: Port Validation with require()
    # ==========================================================================
    print("Example 4: Required Port Validation")
    print("-" * 70)
    
    def process_requires_ports(gdata: GraphData) -> GraphData:
        """Operation that requires specific ports to exist."""
        # Validate required ports upfront
        gdata.require('signal', 'sample_rate', 'calibration')
        
        sig = gdata.signal
        sr = gdata.sample_rate
        cal = gdata.calibration
        
        gdata.signal = sig * cal / sr
        print(f"  Applied calibration with SR={sr}")
        return gdata
    
    # This will work
    gdata_ok = GraphData()
    gdata_ok.signal = np.array([1, 2, 3])
    gdata_ok.sample_rate = 1000.0
    gdata_ok.calibration = 1.5
    
    result_ok = process_requires_ports(gdata_ok)
    print(f"  ✓ Processing succeeded")
    
    # This will fail
    print("\n  Trying with missing port...")
    gdata_bad = GraphData()
    gdata_bad.signal = np.array([1, 2, 3])
    gdata_bad.sample_rate = 1000.0
    # Missing calibration!
    
    try:
        result_bad = process_requires_ports(gdata_bad)
        print(f"  ✗ Should have failed!")
    except ValueError as e:
        print(f"  ✓ Correctly failed: {e}")
    print()
    
    # ==========================================================================
    # Example 5: Flexible Port Names (No "data" Required)
    # ==========================================================================
    print("Example 5: Flexible Port Names - No Required 'data' Port")
    print("-" * 70)
    
    def process_anything(gdata: GraphData) -> GraphData:
        """Works with any port name - no assumptions."""
        print(f"  Available ports: {list(gdata.ports.keys())}")
        
        # Process whatever ports exist
        if gdata.has_port('amplitude'):
            gdata.amplitude = gdata.amplitude * 2
            print(f"  Doubled amplitude")
        
        if gdata.has_port('frequency'):
            gdata.frequency = gdata.frequency + 10
            print(f"  Increased frequency by 10")
        
        return gdata
    
    # Use custom port names
    gdata_custom = GraphData()
    gdata_custom.amplitude = 5.0
    gdata_custom.frequency = 440.0
    gdata_custom.note = "A4"
    
    result_custom = process_anything(gdata_custom)
    print(f"  Final amplitude: {result_custom.amplitude}")
    print(f"  Final frequency: {result_custom.frequency}")
    print(f"  Note unchanged: {result_custom.note}")
    print()
    
    # ==========================================================================
    # Example 6: Explicit set() Method
    # ==========================================================================
    print("Example 6: Explicit Port Setting with set()")
    print("-" * 70)
    
    def compute_statistics(gdata: GraphData) -> GraphData:
        """Compute stats and write to multiple ports explicitly."""
        signal = gdata.signal
        
        # Explicit port setting
        gdata.set('mean', float(np.mean(signal)))
        gdata.set('std', float(np.std(signal)))
        gdata.set('min', float(np.min(signal)))
        gdata.set('max', float(np.max(signal)))
        
        print(f"  Computed statistics:")
        print(f"    mean={gdata.mean:.3f}, std={gdata.std:.3f}")
        print(f"    range=[{gdata.min:.3f}, {gdata.max:.3f}]")
        
        return gdata
    
    gdata_stats = GraphData()
    gdata_stats.signal = np.random.randn(1000)
    result_stats = compute_statistics(gdata_stats)
    print()
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("Port-Based Architecture Benefits")
    print("=" * 70)
    print()
    print("✓ No privileged 'data' port - use any names you want")
    print("✓ Explicit errors when accessing missing ports")
    print("✓ Safe defaults with get() method")
    print("✓ Required port validation with require()")
    print("✓ Clean attribute syntax: gdata.signal, gdata.sample_rate")
    print("✓ Explicit setting with set() method")
    print("✓ Perfect for graph-based dataflow processing")
    print()


if __name__ == '__main__':
    main()
