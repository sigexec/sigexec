"""
Visual Demo: Port Optimization Comparison

This demonstrates the difference between optimize_ports=True (efficient, explicit)
and optimize_ports=False (simple, but wasteful) through actual execution.
"""

from sigexec import Graph, GraphData, requires_ports
import numpy as np

print("="*80)
print("DEMONSTRATION: optimize_ports=True vs False")
print("="*80)
print()

# Define operations that use specific ports
@requires_ports('a')
def use_a_only(g):
    """Operation that only uses port 'a'"""
    g.result_a = g.a * 2
    return g

@requires_ports('b')
def use_b_only(g):
    """Operation that only uses port 'b'"""
    g.result_b = g.b * 3
    return g

@requires_ports('c')
def use_c_only(g):
    """Operation that only uses port 'c'"""
    g.result_c = g.c * 4
    return g

def source(g):
    """Creates three separate ports"""
    g.a = 100
    g.b = 200
    g.c = 300
    return g

def combine(g):
    """Combines all result ports"""
    g.final = g.result_a + g.result_b + g.result_c
    return g

# Create sample for analysis
sample = GraphData()

print("SCENARIO: Source creates ports [a, b, c]")
print("  - use_a_only: needs only 'a'")
print("  - use_b_only: needs only 'b'")
print("  - use_c_only: needs only 'c'")
print("  - combine: needs all results")
print()

print("-"*80)
print("WITH optimize_ports=False (old behavior - wasteful)")
print("-"*80)
print()
print("Data flow:")
print("  source -> use_a_only:   ports [a, b, c] all copied (b, c wasted!)")
print("  use_a_only -> use_b_only:   ports [a, b, c, result_a] (a, c wasted!)")
print("  use_b_only -> use_c_only:   ports [a, b, c, result_a, result_b] (a, b wasted!)")
print("  use_c_only -> combine:   all ports needed here ✓")
print()

g1 = (Graph("Unoptimized", optimize_ports=False)
      .add(source, name='source')
      .add(use_a_only, name='use_a_only')
      .add(use_b_only, name='use_b_only')
      .add(use_c_only, name='use_c_only')
      .add(combine, name='combine'))

result1 = g1.run(sample, verbose=True)
print(f"Final result: {result1.final} ✓")
print()

print("-"*80)
print("WITH optimize_ports=True (new default - efficient!)")
print("-"*80)
print()
print("Data flow:")
print("  source -> use_a_only:   port [a] only (b, c bypass!)")
print("  use_a_only -> use_b_only:   port [b] only (a, result_a bypass!)")
print("  use_b_only -> use_c_only:   port [c] only (result_b bypasses!)")
print("  use_c_only -> combine:   all bypassed ports rejoin + results ✓")
print()

g2 = (Graph("Optimized", optimize_ports=True)
      .add(source, name='source')
      .add(use_a_only, name='use_a_only')
      .add(use_b_only, name='use_b_only')
      .add(use_c_only, name='use_c_only')
      .add(combine, name='combine'))

result2 = g2.run(sample, verbose=True)
print(f"Final result: {result2.final} ✓")
print()

print("="*80)
print("BENEFITS OF optimize_ports=True (Default)")
print("="*80)
print("✓ Memory efficient: Only copy ports that are actually used")
print("✓ Explicit data flow: Clear which ports each operation needs")
print("✓ Implicit branching: Operations using different ports naturally branch")
print("✓ Same results: Both approaches produce identical output")
print()
print(f"Both graphs produced: {result1.final} == {result2.final}")
print()
