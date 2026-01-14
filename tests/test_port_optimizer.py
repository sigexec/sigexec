import logging
from sigexec import Graph, GraphData, requires_ports


def op_attr(g):
    g.data = g.data + 1
    return g


def op_dyn(g):
    g.result = g.custom + (g.get('maybe', 0) or 0)
    return g


def test_static_detection_via_graph_logs(caplog):
    caplog.set_level(logging.DEBUG)

    data = GraphData(data=[1, 2, 3])

    g = Graph(optimize_ports=True).add(op_attr, name='op_attr')
    g.run(data)

    assert "static analysis determined keys" in caplog.text.lower()
    assert "data" in caplog.text


def test_runtime_detection_via_graph_logs(caplog):
    caplog.set_level(logging.DEBUG)

    sample = GraphData()
    sample.custom = 42

    g = Graph(optimize_ports=True).add(op_dyn, name='op_dyn')
    g.run(sample)

    assert ("runtime analysis determined keys" in caplog.text.lower() or
            "static analysis determined keys" in caplog.text.lower())
    assert "custom" in caplog.text or "maybe" in caplog.text or "get" in caplog.text


def test_decorator_integration_and_logs(caplog):
    caplog.set_level(logging.DEBUG)

    @requires_ports('x')
    def op_decl(g):
        g.x_out = g.x + 1
        return g

    g = Graph(optimize_ports=True).add(op_decl, name='op_decl')
    g.run(GraphData(x=3))

    assert any('using decorator-declared keys' in r.message.lower() for r in caplog.records)
