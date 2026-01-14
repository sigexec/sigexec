"""
Mermaid Diagram Comparison: Port Optimization On vs Off

This shows the difference in data flow between optimize_ports=True (bypass behavior)
and optimize_ports=False (all ports flow through everything).
"""

from sigexec import Graph, GraphData, requires_ports
import numpy as np

# Define simple operations with clear port usage
@requires_ports('a')
def use_a(g):
    """Only uses port 'a'"""
    g.result_a = g.a * 2
    return g

@requires_ports('b')
def use_b(g):
    """Only uses port 'b'"""
    g.result_b = g.b * 3
    return g

@requires_ports('result_a', 'result_b')
def combine(g):
    """Combines both results"""
    g.final = g.result_a + g.result_b
    return g

def source(g):
    """Creates two ports: a and b"""
    g.a = 10
    g.b = 20
    return g

print("="*80)
print("MERMAID DIAGRAMS: Port Flow Comparison")
print("="*80)
print()

# Create sample
sample = GraphData()

print("Graph Structure:")
print("  source -> use_a -> use_b -> combine")
print()
print("Port Usage:")
print("  source: creates [a, b]")
print("  use_a: needs [a]")
print("  use_b: needs [b]")
print("  combine: needs [result_a, result_b]")
print()

print("-"*80)
print("WITHOUT Port Optimization (optimize_ports=False)")
print("-"*80)
print()
print("Mermaid Conceptual Diagram:")
print()
print("```mermaid")
print("flowchart LR")
print("    source([source]) --|a, b|--> use_a")
print("    use_a[use_a] --|a, b, result_a|--> use_b")
print("    use_b[use_b] --|a, b, result_a, result_b|--> combine")
print("    combine[combine] --> final([final])")
print("```")
print()
print("Note: ALL ports flow through EVERY operation (wasteful)")
print("  - use_a receives [a, b] but only uses [a]")
print("  - use_b receives [a, b, result_a] but only uses [b]")
print("  - Unused ports (b through use_a, a through use_b) are copied unnecessarily")
print()

g1 = Graph("Unoptimized", optimize_ports=False)
g1.add(source, name='source')
g1.add(use_a, name='use_a')
g1.add(use_b, name='use_b')
g1.add(combine, name='combine')

print("Actual execution with verbose=True:")
result1 = g1.run(sample, verbose=True)
print(f"Result: {result1.final}")
print()

print("-"*80)
print("WITH Port Optimization (optimize_ports=True) - DEFAULT")
print("-"*80)
print()
print("Mermaid Conceptual Diagram:")
print()
print("```mermaid")
print("flowchart LR")
print("    source([source]) --|a|--> use_a")
print("    source -.b bypasses use_a.-> use_b")
print("    use_a[use_a] --|result_a|--> combine")
print("    use_a -.b continues to bypass.-> use_b")
print("    use_b[use_b] --|b, result_b|--> combine")
print("    use_b -.result_a bypasses use_b.-> combine")
print("    combine[combine] --> final([final])")
print("```")
print()
print("Note: Only NEEDED ports flow to each operation (efficient)")
print("  - use_a receives ONLY [a] (port b bypasses it)")
print("  - use_b receives ONLY [b] (ports a, result_a bypass it)")
print("  - combine receives ALL accumulated ports [result_a, result_b] from merge")
print("  - Bypassed ports are efficiently routed around operations that don't need them")
print()

g2 = Graph("Optimized", optimize_ports=True)
g2.add(source, name='source')
g2.add(use_a, name='use_a')
g2.add(use_b, name='use_b')
g2.add(combine, name='combine')

print("Actual execution with verbose=True:")
result2 = g2.run(sample, verbose=True)
print(f"Result: {result2.final}")
print()

print("="*80)
print("VISUAL SUMMARY")
print("="*80)
print()
print("Without optimization: Solid arrows everywhere (all ports to all ops)")
print("  source --[a,b]--> use_a --[a,b,result_a]--> use_b --[...]--> combine")
print()
print("With optimization: Solid arrows for used ports, dotted for bypasses")
print("  source --[a]--> use_a              (b bypasses)")
print("  source ········[b]·······> use_b   (b bypasses use_a)")
print("  use_a --[result_a]--> combine      (result_a bypasses use_b)")
print("  use_b --[b, result_b]--> combine")
print()
print(f"Both produce same result: {result1.final} == {result2.final}")
