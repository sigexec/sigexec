"""
Demonstration of optimize_ports=True vs False in Mermaid visualization

This shows how port optimization affects data flow visualization:
- optimize_ports=False: All ports flow through every operation (wasteful)
- optimize_ports=True: Each operation receives only the ports it needs (efficient)
"""

from sigexec import Graph, GraphData

def source(g):
    """Creates three ports: a, b, c"""
    g.a = 100
    g.b = 200
    g.c = 300
    return g

def use_a(g):
    """Only uses port 'a'"""
    g.result_a = g.a * 2
    return g

def use_b(g):
    """Only uses port 'b'"""
    g.result_b = g.b * 3
    return g

def use_c(g):
    """Only uses port 'c'"""
    g.result_c = g.c * 4
    return g

def combine(g):
    """Uses all result ports"""
    g.final = g.result_a + g.result_b + g.result_c
    return g

# Create sample for port analysis
sample = GraphData()

print("="*80)
print("GRAPH WITH optimize_ports=False (wasteful - all ports flow everywhere)")
print("="*80)
g1 = (Graph("Unoptimized", optimize_ports=False)
      .add(source, name='source')
      .add(use_a, name='use_a')
      .add(use_b, name='use_b')
      .add(use_c, name='use_c')
      .add(combine, name='combine'))

# Run with verbose to see port flow
result1 = g1.run(sample, verbose=True)
print(f"Result: {result1.final}")

print("\n" + "="*80)
print("GRAPH WITH optimize_ports=True (efficient - only needed ports flow)")
print("="*80)
g2 = (Graph("Optimized", optimize_ports=True)
      .add(source, name='source')
      .add(use_a, name='use_a')
      .add(use_b, name='use_b')
      .add(use_c, name='use_c')
      .add(combine, name='combine'))

# Run with verbose to see optimized port flow
result2 = g2.run(sample, verbose=True)
print(f"Result: {result2.final}")

print("\n" + "="*80)
print("KEY DIFFERENCES:")
print("="*80)
print("Without optimization:")
print("  - 'use_a' receives ports [a, b, c] even though it only uses 'a'")
print("  - 'use_b' receives ports [a, b, c, result_a] even though it only uses 'b'")
print("  - Unused ports (b, c) are copied through 'use_a' unnecessarily")
print()
print("With optimization:")
print("  - 'use_a' receives ONLY port 'a' (ports b, c bypass it)")
print("  - 'use_b' receives ONLY port 'b' (other ports bypass it)")
print("  - 'use_c' receives ONLY port 'c' (other ports bypass it)")
print("  - 'combine' receives all the result ports it needs")
print()
print("This reduces memory usage and makes data flow more explicit!")
