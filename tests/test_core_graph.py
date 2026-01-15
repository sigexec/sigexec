"""
Pytest unit tests for core Graph functionality.
"""

import numpy as np
import pytest
from sigexec import Graph, GraphData


class TestGraphCreation:
    """Test Graph initialization."""
    
    def test_create_default_pipeline(self):
        """Test creating graph with defaults."""
        p = Graph()
        assert p.name == "Graph"
        assert p._enable_cache == True
        assert len(p.operations) == 0
        
    def test_create_named_pipeline(self):
        """Test creating graph with custom name."""
        p = Graph("MyPipeline")
        assert p.name == "MyPipeline"
        
    def test_create_with_cache_disabled(self):
        """Test creating graph with caching disabled."""
        p = Graph(enable_cache=False)
        assert p._enable_cache == False
        
    def test_create_with_input_data(self):
        """Test creating graph with initial data."""
        data = GraphData(np.array([1, 2, 3]))
        p = Graph(input_data=data)
        assert p._input_data is data


class TestGraphAdd:
    """Test adding operations to graph."""
    
    def test_add_simple_operation(self):
        """Test adding a simple operation."""
        p = Graph()
        
        def multiply_by_2(sig):
            return GraphData(sig.data * 2, metadata=sig.metadata.copy())
        
        p.add(multiply_by_2)
        
        assert len(p.operations) == 1
        assert p.operations[0]['func'] is multiply_by_2
        
    def test_add_with_name(self):
        """Test adding operation with custom name."""
        p = Graph()
        p.add(lambda sig: sig, name="CustomOp")
        
        assert p.operations[0]['name'] == "CustomOp"
        
    def test_add_multiple_operations(self):
        """Test chaining multiple add calls."""
        p = Graph()
        p.add(lambda sig: sig, name="Op1")
        p.add(lambda sig: sig, name="Op2")
        p.add(lambda sig: sig, name="Op3")
        
        assert len(p.operations) == 3
        assert p.operations[0]['name'] == "Op1"
        assert p.operations[2]['name'] == "Op3"
        
    def test_add_returns_self(self):
        """Test that add returns self for chaining."""
        p = Graph()
        result = p.add(lambda sig: sig)
        
        assert result is p


class TestGraphRun:
    """Test graph execution."""
    
    def test_run_empty_pipeline(self):
        """Test running graph with no operations."""
        data = GraphData(np.array([1, 2, 3]))
        p = Graph()
        result = p.run(data)
        
        np.testing.assert_array_equal(result.data, data.data)
        
    def test_run_single_operation(self):
        """Test running graph with one operation."""
        data = GraphData(np.array([1.0, 2.0, 3.0]))
        
        def multiply_by_2(sig):
            return GraphData(sig.data * 2, metadata=sig.metadata.copy())
        
        p = Graph()
        p.add(multiply_by_2)
        result = p.run(data)
        
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result.data, expected)
        
    def test_run_multiple_operations(self):
        """Test running graph with multiple operations."""
        data = GraphData(np.array([1.0, 2.0, 3.0]))
        
        def add_10(sig):
            return GraphData(sig.data + 10, metadata=sig.metadata.copy())
        
        def multiply_by_2(sig):
            return GraphData(sig.data * 2, metadata=sig.metadata.copy())
        
        p = Graph()
        p.add(add_10).add(multiply_by_2)
        result = p.run(data)
        
        # (data + 10) * 2 = [22, 24, 26]
        expected = np.array([22.0, 24.0, 26.0])
        np.testing.assert_array_equal(result.data, expected)
        
    def test_run_with_input_data(self):
        """Test running graph with input_data set."""
        data = GraphData(np.array([5.0]))
        
        p = Graph(input_data=data)
        p.add(lambda sig: GraphData(sig.data * 3, metadata=sig.metadata.copy()))
        
        result = p.run()  # No argument needed
        assert result.data[0] == 15.0


class TestGraphCaching:
    """Test graph caching functionality."""
    
    def test_cache_enabled_by_default(self):
        """Test that caching is enabled by default."""
        p = Graph()
        assert p._enable_cache == True
        
    @pytest.mark.skip(reason="Caching not yet implemented in flow graph execution")
    def test_cache_reuse(self):
        """Test that cached results are reused."""
        call_count = {'count': 0}
        
        def counting_op(sig):
            call_count['count'] += 1
            return GraphData(sig.data * 2, metadata=sig.metadata.copy())
        
        # Disable port optimization to avoid counting port analysis calls
        p = Graph(optimize_ports=False)
        p.add(counting_op)
        
        data = GraphData(np.array([1.0]))
        
        # First run
        p.run(data)
        assert call_count['count'] == 1
        
        # Second run should use cache
        p.run(data)
        assert call_count['count'] == 1  # Still 1, not 2
        
    @pytest.mark.skip(reason="Caching not yet implemented in flow graph execution")
    def test_cache_disabled(self):
        """Test running with cache disabled."""
        call_count = {'count': 0}
        
        def counting_op(sig):
            call_count['count'] += 1
            return GraphData(sig.data * 2, metadata=sig.metadata.copy())
        
        # Disable both cache and port optimization to avoid analysis calls
        p = Graph(enable_cache=False, optimize_ports=False)
        p.add(counting_op)
        
        data = GraphData(np.array([1.0]))
        
        # Run twice
        p.run(data)
        p.run(data)
        
        assert call_count['count'] == 2  # Called both times
        
    def test_clear_cache(self):
        """Test clearing global cache."""
        p1 = Graph()
        p1.add(lambda sig: GraphData(sig.data * 2, metadata=sig.metadata.copy()))
        
        data = GraphData(np.array([1.0]))
        p1.run(data)
        
        # Clear cache
        Graph.clear_cache()
        
        # Run again - should execute (not from cache)
        result = p1.run(data)
        assert result.data[0] == 2.0


class TestGraphInputData:
    """Test graph input_data method."""
    
    def test_input_data_method(self):
        """Test setting input data via method."""
        data = GraphData(np.array([7.0]))
        
        p = Graph()
        p.input_data(data)
        p.add(lambda sig: GraphData(sig.data * 2, metadata=sig.metadata.copy()))
        
        result = p.run()
        assert result.data[0] == 14.0
        
    def test_input_data_returns_self(self):
        """Test that input_data returns self for chaining."""
        p = Graph()
        data = GraphData(np.array([1.0]))
        result = p.input_data(data)
        
        assert result is p


class TestGraphHelperMethods:
    """Test graph helper methods."""
    
    def test_map_alias(self):
        """Test that map is alias for add."""
        p = Graph()
        p.map(lambda sig: GraphData(sig.data * 2, metadata=sig.metadata.copy()))
        
        assert len(p.operations) == 1
        
    def test_transform(self):
        """Test transform method that operates on data array."""
        data = GraphData(np.array([1.0, 2.0, 3.0]))
        
        p = Graph()
        p.transform(lambda arr: arr * 10)
        result = p.run(data)
        
        expected = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_equal(result.data, expected)
        
    def test_tap_for_inspection(self):
        """Test tap method for inspecting signal without modification."""
        inspected_values = []
        
        def inspector(sig):
            inspected_values.append(sig.data.copy())
        
        data = GraphData(np.array([1.0, 2.0]))
        
        # Disable port optimization to avoid analysis side effects
        p = Graph(optimize_ports=False)
        p.add(lambda sig: GraphData(sig.data * 2, metadata=sig.metadata.copy()))
        p.tap(inspector)
        p.add(lambda sig: GraphData(sig.data + 10, metadata=sig.metadata.copy()))
        
        result = p.run(data)
        
        # Tap should have captured intermediate value
        assert len(inspected_values) == 1
        np.testing.assert_array_equal(inspected_values[0], np.array([2.0, 4.0]))
        
        # Final result should be (data * 2) + 10
        expected = np.array([12.0, 14.0])
        np.testing.assert_array_equal(result.data, expected)


class TestGraphLength:
    """Test graph length."""
    
    def test_empty_pipeline_length(self):
        """Test length of empty graph."""
        p = Graph()
        assert len(p) == 0
        
    def test_pipeline_length_after_adds(self):
        """Test length after adding operations."""
        p = Graph()
        p.add(lambda sig: sig)
        p.add(lambda sig: sig)
        p.add(lambda sig: sig)
        
        assert len(p) == 3


class TestGraphRepr:
    """Test graph string representation."""
    
    def test_repr_includes_name(self):
        """Test that repr includes graph name."""
        p = Graph("TestGraph")
        repr_str = repr(p)
        
        assert "TestGraph" in repr_str
        
    def test_repr_shows_cache_status(self):
        """Test that repr shows cache status."""
        p_cached = Graph(enable_cache=True)
        p_uncached = Graph(enable_cache=False)
        
        assert "cached" in repr(p_cached)
        assert "cached" not in repr(p_uncached)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
