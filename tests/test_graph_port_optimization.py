import logging
from sigexec.core.data import GraphData
from sigexec.core.graph import Graph
from sigexec.core.port_optimizer import PortAnalyzer


def test_graph_uses_port_optimizer_and_logs(caplog):
    caplog.set_level(logging.DEBUG)

    # Define two ops touching different ports
    def op_a(g):
        g.a_out = g.a + 1
        return g

    def op_b(g):
        g.b_out = g.b * 2
        return g

    sample = GraphData()
    sample.a = 1
    sample.b = 2

    g = (Graph(optimize_ports=True)
         .add(op_a, name='op_a')
         .add(op_b, name='op_b'))

    result = g.run(sample)

    # Ensure results are computed
    assert result.a_out == 2
    assert result.b_out == 4

    # Ensure PortAnalyzer logging ran for the ops
    assert any('static analysis determined keys' in r.message.lower() or 'runtime analysis determined keys' in r.message.lower() for r in caplog.records)
