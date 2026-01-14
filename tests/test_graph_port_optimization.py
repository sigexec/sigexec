import logging
from sigexec import Graph, GraphData


def op_a_min(g):
    # Should only receive 'a' and not 'b'
    assert hasattr(g, 'a')
    assert not hasattr(g, 'b')
    g.a_out = g.a + 1
    return g


def op_b_min(g):
    # Should only receive 'b' and not 'a'
    assert hasattr(g, 'b')
    assert not hasattr(g, 'a')
    g.b_out = g.b * 2
    return g


def test_graph_uses_port_optimizer_and_logs(caplog):
    caplog.set_level(logging.DEBUG)

    sample = GraphData()
    sample.a = 1
    sample.b = 2

    g = (Graph(optimize_ports=True)
         .add(op_a_min, name='op_a')
         .add(op_b_min, name='op_b'))

    result = g.run(sample)

    # Ensure results are computed
    assert result.a_out == 2
    assert result.b_out == 4

    # Ensure PortAnalyzer logging ran for the ops
    assert any('static analysis determined keys' in r.message.lower() or 'runtime analysis determined keys' in r.message.lower() for r in caplog.records)


def test_graph_passes_minimal_ports(caplog):
    """Explicit test: stages should receive only the ports they need."""
    caplog.set_level(logging.DEBUG)

    # Reuse module-level ops op_a_min and op_b_min
    sample = GraphData()
    sample.a = 5
    sample.b = 7

    g = (Graph(optimize_ports=True)
         .add(op_a_min, name='op_a')
         .add(op_b_min, name='op_b'))

    result = g.run(sample)

    assert result.a_out == 6
    assert result.b_out == 14
