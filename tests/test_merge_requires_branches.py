import pytest
from sigexec import Graph, GraphData


def test_merge_without_branches_raises():
    def merge_fn(branches):
        return GraphData()

    g = (Graph()
         .add(lambda d: d)
         .branch(['a', 'b'])
         .add(lambda d: d, branch='a')
         .add(lambda d: d, branch='b'))

    with pytest.raises(ValueError):
        g.merge(merge_fn)  # should raise because branches are not specified


def test_merge_accepts_legacy_list_with_combiner():
    # Legacy API: .merge([branches], combiner=fn) should work
    def combine(branches):
        return GraphData()

    g = (Graph()
         .add(lambda d: d)
         .branch(['x', 'y'])
         .add(lambda d: d, branch='x')
         .add(lambda d: d, branch='y'))

    # Should not raise
    g.merge(['x', 'y'], combiner=combine)
