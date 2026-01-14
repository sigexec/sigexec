import pytest
import logging
from sigexec import requires_ports, GraphData, Graph


def test_decorator_precedence_and_logging(caplog):
    caplog.set_level(logging.DEBUG)

    @requires_ports('a')
    def op_declared(g):
        # Access declared port
        g.a_out = g.a + 1
        return g

    # Analyzer should pick up decorator and we should see a log entry when running
    g = Graph(optimize_ports=True).add(op_declared, name='op_declared')
    g.run(GraphData(a=1))
    assert any('using decorator-declared keys' in r.message.lower() for r in caplog.records)


def test_strict_mode_raises_on_undeclared_access():
    # Function declares only 'a' but tries to access 'b' via get()
    @requires_ports('a')
    def op_bad(g):
        # Access declared port
        _ = g.a
        # Attempt to access undeclared port (via get)
        _ = g.get('b', None)
        return g

    sample = GraphData()
    sample.a = 1
    sample.b = 2

    g = (Graph(optimize_ports=True, optimize_ports_strict=True)
         .add(op_bad, name='op_bad'))

    with pytest.raises(ValueError) as exc:
        g.run(sample)

    assert "undeclared" in str(exc.value).lower()


def test_decorator_allows_minimal_passing_when_not_strict():
    @requires_ports('a')
    def op_ok(g):
        # Should be fine accessing a only
        assert hasattr(g, 'a')
        # Using get on undeclared is allowed when not strict (returns default)
        assert g.get('b', None) is None
        g.a_out = g.a + 1
        return g

    sample = GraphData()
    sample.a = 5
    sample.b = 7

    g = (Graph(optimize_ports=True, optimize_ports_strict=False)
         .add(op_ok, name='op_ok'))

    result = g.run(sample)
    assert result.a_out == 6
