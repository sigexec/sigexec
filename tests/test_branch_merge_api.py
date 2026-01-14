import numpy as np
from sigexec import Graph, GraphData


def test_merge_receives_ordered_view_by_index():
    # Setup
    def gen(data: GraphData) -> GraphData:
        data.data = np.array([1, 2, 3])
        return data

    def a_proc(data: GraphData) -> GraphData:
        data.set('val', data.data * 2)
        return data

    def b_proc(data: GraphData) -> GraphData:
        data.set('val', data.data + 10)
        return data

    def merge_fn(branches):
        # Index-based access
        a = branches[0].get('val')
        b = branches[1].get('val')
        merged = GraphData()
        merged.data = np.concatenate([a, b])
        return merged

    g = (Graph('t')
         .add(gen)
         .branch(['a', 'b'])
         .add(a_proc, branch='a')
         .add(b_proc, branch='b')
         .merge(merge_fn, branches=['a', 'b']))

    res = g.run(GraphData())
    assert isinstance(res.data, np.ndarray)
    assert res.data.shape[0] == 6


def test_merge_receives_ordered_view_by_name():
    # Setup
    def gen(data: GraphData) -> GraphData:
        data.data = np.array([1, 2])
        return data

    def a_proc(data: GraphData) -> GraphData:
        data.set('x', 1)
        return data

    def b_proc(data: GraphData) -> GraphData:
        data.set('x', 2)
        return data

    def merge_fn(branches):
        # Name-based access
        assert branches['a'].get('x') == 1
        assert branches['b'].get('x') == 2
        merged = GraphData()
        merged.data = np.array([0])
        return merged

    g = (Graph('t2')
         .add(gen)
         .branch(['a', 'b'])
         .add(a_proc, branch='a')
         .add(b_proc, branch='b')
         .merge(merge_fn, branches=['a', 'b']))

    res = g.run(GraphData())
    assert isinstance(res.data, np.ndarray)
