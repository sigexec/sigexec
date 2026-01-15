"""
Variant Demo
============
Demonstrates execution with parameter variants, showing hexagon shapes in mermaid.
"""

from sigexec import Graph, GraphData
import numpy as np

print("=" * 60)
print("VARIANT DEMONSTRATION")
print("=" * 60)

# Create a graph with variant operations
graph = Graph(name="VariantDemo")

# Regular operation - will show as rectangle
def generate(gdata):
    """Generate some data"""
    return GraphData(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        metadata=gdata.metadata
    )

# Operation factory for scaling variants
def make_scaler(factor):
    """Factory that creates a scaling operation"""
    def scale(gdata):
        return GraphData(
            gdata.data * factor,
            metadata={**gdata.metadata, 'scale_factor': factor}
        )
    return scale

# Operation factory for offset variants
def make_offsetter(amount):
    """Factory that creates an offset operation"""
    def offset(gdata):
        return GraphData(
            gdata.data + amount,
            metadata={**gdata.metadata, 'offset': amount}
        )
    return offset

# Regular operation - will show as rectangle
def compute_stats(gdata):
    """Compute statistics"""
    return GraphData(
        gdata.data,
        metadata={
            **gdata.metadata,
            'mean': float(np.mean(gdata.data)),
            'max': float(np.max(gdata.data))
        }
    )

print("\n1. Building Graph with Variants")
print("-" * 60)

# Add operations, some with variants
graph.add(generate, name="Generate")
graph.variant(make_scaler, [2.0, 3.0, 5.0], names=['Scale_2x', 'Scale_3x', 'Scale_5x'], name='Scale')
graph.variant(make_offsetter, [10.0, 20.0], names=['Offset_10', 'Offset_20'], name='Offset')
graph.add(compute_stats, name="Stats")

print("[OK] Graph created with 2 variant operations")
print("  - Scale: 3 variants (factor = 2.0, 3.0, 5.0)")
print("  - Offset: 2 variants (amount = 10.0, 20.0)")
print("  - Total combinations: 3 x 2 = 6 execution paths")

print("\n2. Execution Graph (Mermaid)")
print("-" * 60)
print("Notice: Operations with variants show as hexagons {{Name}}")
print("        Regular operations show as rectangles [Name]")
print()
print(graph.to_mermaid())


# Staticdash dashboard for publishing
def create_dashboard():
    try:
        import staticdash as sd
    except ImportError:
        raise RuntimeError("staticdash is required for dashboard publishing")

    dashboard = sd.Dashboard("Variant demo")
    page = sd.Page("variant-demo", "Variant demo")
    page.add_header("Variant demo", level=1)
    page.add_text("""
    This demo shows a graph with parameter variants (hexagons in mermaid).
    """)
    page.add_header("Execution Graph", level=2)
    page.add_syntax(graph.to_mermaid(), language="mermaid")
    dashboard.add_page(page)
    return dashboard

print("\n[OK] Variant demo completed!")
print("\nKey visualization:")
print("  [Rectangle] = Regular operation (single execution)")
print("  {{Hexagon}}  = Variant operation (multiple parameter values)")
