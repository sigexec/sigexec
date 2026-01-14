import pytest
from sigexec.core.port_optimizer import PortAnalyzer, PortAccessTracker, create_port_subset, merge_port_subsets
from sigexec.core.data import GraphData


def _op_attr(g):
    return g.data + 1


def _op_ports_sub(g):
    return g.ports['foo']


def test_static_attribute_access():
    keys = PortAnalyzer.get_operation_metadata_keys(_op_attr)
    assert keys == {"data"}


def test_static_ports_subscript():
    keys = PortAnalyzer.get_operation_metadata_keys(_op_ports_sub)
    assert keys == {"foo"}


def test_runtime_detection_attribute():
    def op(g):
        # dynamic access by attribute (not visible statically if name differs)
        return g.custom + (g.get('maybe', 0) or 0)

    sample = GraphData(data=[0])
    sample.custom = 42
    keys = PortAnalyzer.get_operation_metadata_keys(op, sample)
    assert keys == {"custom", "maybe"} or keys == {"custom"}


def test_port_access_tracker_reads_and_writes():
    base = {"a": 1, "b": 2}
    tracker = PortAccessTracker(base)
    assert tracker.get('a') == 1
    assert 'a' in tracker.accessed_keys
    tracker['c'] = 3
    assert tracker['c'] == 3


def test_create_and_merge_subsets():
    full = {"a": 1, "b": 2, "c": 3}
    subset = create_port_subset(full, {"a", "c"})
    assert subset == {"a": 1, "c": 3}

    unioned = merge_port_subsets([{"a": 1}, {"b": 2}])
    assert unioned == {"a": 1, "b": 2}

    intersected = merge_port_subsets([{"a": 1, "b": 2}, {"b": 2, "c": 3}], strategy='intersection')
    assert intersected == {"b": 2}
