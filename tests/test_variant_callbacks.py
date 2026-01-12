"""
Tests for variant callback and memory-efficient processing features.
"""

import numpy as np
import pytest
from sigexec import Graph, SignalData


class TestVariantCallbacks:
    """Test on_variant_complete callback functionality."""
    
    def test_callback_invoked_for_each_variant(self):
        """Callback should be called once for each variant combination."""
        data = SignalData(data=np.array([1.0, 2.0, 3.0]), metadata={})
        
        def multiply(factor):
            def _multiply(sig):
                return SignalData(data=sig.data * factor, metadata=sig.metadata)
            return _multiply
        
        # Track callback invocations
        invocations = []
        def track_variant(params, result):
            invocations.append((params.copy(), result.data.copy()))
        
        # Run with 3 variants
        graph = (
            Graph("callback_test")
            .input_data(data)
            .variants(multiply, [2.0, 3.0, 5.0], names=['2x', '3x', '5x'])
        )
        
        results = graph.run(on_variant_complete=track_variant)
        
        # Should be called 3 times
        assert len(invocations) == 3
        
        # Check params and results match
        assert invocations[0][0]['variant'] == ['2x']
        np.testing.assert_array_equal(invocations[0][1], np.array([2.0, 4.0, 6.0]))
        
        assert invocations[1][0]['variant'] == ['3x']
        np.testing.assert_array_equal(invocations[1][1], np.array([3.0, 6.0, 9.0]))
        
        assert invocations[2][0]['variant'] == ['5x']
        np.testing.assert_array_equal(invocations[2][1], np.array([5.0, 10.0, 15.0]))
    
    def test_callback_with_multiple_variants(self):
        """Callback should work with cartesian product of multiple variants."""
        data = SignalData(data=np.array([10.0]), metadata={})
        
        def multiply(factor):
            def _mult(sig):
                return SignalData(data=sig.data * factor, metadata=sig.metadata)
            return _mult
        
        def add(value):
            def _add(sig):
                return SignalData(data=sig.data + value, metadata=sig.metadata)
            return _add
        
        invocations = []
        def track(params, result):
            invocations.append(params['variant'])
        
        # 2 × 2 = 4 combinations
        graph = (
            Graph()
            .input_data(data)
            .variants(multiply, [2.0, 3.0], names=['2x', '3x'])
            .variants(add, [1.0, 5.0], names=['+1', '+5'])
        )
        
        graph.run(on_variant_complete=track)
        
        assert len(invocations) == 4
        assert invocations[0] == ['2x', '+1']
        assert invocations[1] == ['2x', '+5']
        assert invocations[2] == ['3x', '+1']
        assert invocations[3] == ['3x', '+5']
    
    def test_return_results_false_saves_memory(self):
        """With return_results=False, should return empty list."""
        data = SignalData(data=np.array([1.0]), metadata={})
        
        def identity(x):
            def _id(sig):
                return sig
            return _id
        
        graph = (
            Graph()
            .input_data(data)
            .variants(identity, [1, 2, 3])
        )
        
        # With return_results=False, list should be empty
        results = graph.run(return_results=False)
        assert len(results) == 0
        
        # With return_results=True (default), list should have all results
        results = graph.run(return_results=True)
        assert len(results) == 3
    
    def test_callback_and_return_results_both_work(self):
        """Can use callback AND keep results in memory if desired."""
        data = SignalData(data=np.array([5.0]), metadata={})
        
        def double(sig):
            return SignalData(data=sig.data * 2, metadata=sig.metadata)
        
        callback_count = [0]
        def count_variants(params, result):
            callback_count[0] += 1
        
        graph = (
            Graph()
            .input_data(data)
            .variants(lambda _: double, [1, 2], names=['a', 'b'])
        )
        
        results = graph.run(
            on_variant_complete=count_variants,
            return_results=True
        )
        
        # Both should work
        assert callback_count[0] == 2
        assert len(results) == 2
    
    def test_callback_without_variants_not_called(self):
        """Callback should not be called when there are no variants."""
        data = SignalData(data=np.array([1.0]), metadata={})
        
        called = [False]
        def should_not_call(params, result):
            called[0] = True
        
        graph = (
            Graph()
            .input_data(data)
            .add(lambda sig: SignalData(data=sig.data * 2, metadata=sig.metadata))
        )
        
        result = graph.run(on_variant_complete=should_not_call)
        
        # Callback should NOT be called (no variants)
        assert called[0] is False
        # Should return single result, not list
        assert isinstance(result, SignalData)


class TestVariantCallbacksWithBranches:
    """Test callbacks work correctly with DAG branches."""
    
    def test_callback_with_variants_and_branches(self):
        """Callback should work when variants are combined with branches."""
        data = SignalData(data=np.array([10.0]), metadata={})
        
        def multiply(factor):
            def _mult(sig):
                return SignalData(data=sig.data * factor, metadata=sig.metadata)
            return _mult
        
        invocations = []
        def track(params, result):
            invocations.append((params['variant'], float(result.data[0])))
        
        # Graph with variants and branches
        graph = (
            Graph()
            .input_data(data)
            .variants(multiply, [2.0, 3.0], names=['2x', '3x'])
            .branch(['a', 'b'])
            .add(lambda sig: SignalData(data=sig.data + 1, metadata=sig.metadata), 
                 branch='a')
            .add(lambda sig: SignalData(data=sig.data + 10, metadata=sig.metadata),
                 branch='b')
            .merge(['a', 'b'], 
                   combiner=lambda sigs: SignalData(
                       data=sigs[0].data + sigs[1].data,
                       metadata=sigs[0].metadata
                   ))
        )
        
        results = graph.run(on_variant_complete=track)
        
        # Should be called for each variant (2 variants)
        assert len(invocations) == 2
        
        # Verify params and results
        assert invocations[0][0] == ['2x']
        assert invocations[0][1] == (10*2 + 1) + (10*2 + 10)  # branch_a + branch_b
        
        assert invocations[1][0] == ['3x']
        assert invocations[1][1] == (10*3 + 1) + (10*3 + 10)
