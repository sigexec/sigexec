"""
Test interaction between variants and branch operations.

Ensures that branches created after a variant are properly duplicated per variant value
without creating duplicate branch names or conflicts.
"""

import pytest
import numpy as np
from sigexec.core.graph import Graph
from sigexec.core.data import SignalData


class TestVariantBranchInteraction:
    """Test that variants + branch work together correctly."""
    
    def test_branch_after_variant_duplicate_mode(self):
        """Branches after variant should be duplicated per variant value."""
        # Create a variant over a scalar parameter
        data = SignalData(data=np.array([1, 2, 3, 4]))
        
        def multiply_factory(factor):
            def multiply(signal):
                return SignalData(data=signal.data * factor)
            return multiply
        
        def add_one(signal):
            return SignalData(data=signal.data + 1)
        
        def add_two(signal):
            return SignalData(data=signal.data + 2)
        
        graph = (
            Graph("sweep_then_branch")
            .input_data(data)
            .variant(multiply_factory, [2, 3], names=["x2", "x3"])
            .branch(["b1", "b2"])  # Branch after variant
            .add(add_one, name="add_one", branch="b1")
            .add(add_two, name="add_two", branch="b2")
            .merge(["b1", "b2"], combiner=lambda sigs: SignalData(data=sigs[0].data + sigs[1].data))
        )
        
        results = graph.run()
        
        # Should have 2 results (one per variant value)
        assert len(results) == 2
        
        # Check first variant value (x2): [1,2,3,4] * 2 = [2,4,6,8]
        # Then b1: +1 = [3,5,7,9], b2: +2 = [4,6,8,10]
        # Merge: [7,11,15,19]
        params1, result1 = results[0]
        assert params1 == {"variant": ["x2"]}
        expected1 = np.array([7, 11, 15, 19])
        np.testing.assert_array_equal(result1.data, expected1)
        
        # Check second variant value (x3): [1,2,3,4] * 3 = [3,6,9,12]
        # Then b1: +1 = [4,7,10,13], b2: +2 = [5,8,11,14]
        # Merge: [9,15,21,27]
        params2, result2 = results[1]
        assert params2 == {"variant": ["x3"]}
        expected2 = np.array([9, 15, 21, 27])
        np.testing.assert_array_equal(result2.data, expected2)
    
    def test_branch_after_variant_function_mode(self):
        """Branches with functions after variant should work correctly."""
        data = SignalData(data=np.array([10, 20, 30]))
        
        def scale_factory(s):
            def scale(signal):
                return SignalData(data=signal.data * s)
            return scale
        
        def extract_first(signal):
            return SignalData(data=signal.data[:1])
        
        def extract_last(signal):
            return SignalData(data=signal.data[-1:])
        
        graph = (
            Graph("sweep_branch_funcs")
            .input_data(data)
            .variant(scale_factory, [1, 2], names=["s1", "s2"])
            .branch(["first", "last"], functions=[extract_first, extract_last])
            .merge(["first", "last"], combiner=lambda sigs: SignalData(data=np.concatenate([sigs[0].data, sigs[1].data])))
        )
        
        results = graph.run()
        assert len(results) == 2
        
        # s1: [10, 20, 30] -> first=[10], last=[30] -> merge=[10, 30]
        params1, result1 = results[0]
        assert params1 == {"variant": ["s1"]}
        np.testing.assert_array_equal(result1.data, np.array([10, 30]))
        
        # s2: [20, 40, 60] -> first=[20], last=[60] -> merge=[20, 60]
        params2, result2 = results[1]
        assert params2 == {"variant": ["s2"]}
        np.testing.assert_array_equal(result2.data, np.array([20, 60]))
    
    def test_multiple_variants_with_branches(self):
        """Multiple variants create cartesian product, branches work per combination."""
        data = SignalData(data=np.array([1.0, 2.0]))
        
        def multiply_factory(m):
            def multiply(signal):
                return SignalData(data=signal.data * m)
            return multiply
        
        def add_factory(a):
            def add(signal):
                return SignalData(data=signal.data + a)
            return add
        
        def path_a(signal):
            return SignalData(data=signal.data * 10)
        
        def path_b(signal):
            return SignalData(data=signal.data * 100)
        
        graph = (
            Graph("multi_sweep_branch")
            .input_data(data)
            .variant(multiply_factory, [2, 3], names=["m2", "m3"])
            .variant(add_factory, [10, 20], names=["a10", "a20"])
            .branch(["pa", "pb"], functions=[path_a, path_b])
            .merge(["pa", "pb"], combiner=lambda sigs: SignalData(data=sigs[0].data + sigs[1].data))
        )
        
        results = graph.run()
        
        # Should have 2x2=4 combinations
        assert len(results) == 4
        
        # Each result should have unique params
        all_params = [r[0] for r in results]
        assert {"variant": ["m2", "a10"]} in all_params
        assert {"variant": ["m2", "a20"]} in all_params
        assert {"variant": ["m3", "a10"]} in all_params
        assert {"variant": ["m3", "a20"]} in all_params
        
        # Verify one specific case: m2, a10
        # [1, 2] * 2 = [2, 4] + 10 = [12, 14]
        # pa: * 10 = [120, 140], pb: * 100 = [1200, 1400]
        # merge: [1320, 1540]
        result_m2_a10 = [r[1] for r in results if r[0] == {"variant": ["m2", "a10"]}][0]
        expected = np.array([1320.0, 1540.0])
        np.testing.assert_array_equal(result_m2_a10.data, expected)
    
    def test_no_duplicate_branch_names_across_variants(self):
        """Branch names should be properly scoped per variant combination."""
        data = SignalData(data=np.array([5]))
        
        def scale_factory(s):
            def scale(signal):
                return SignalData(data=signal.data * s)
            return scale
        
        def identity(signal):
            return signal
        
        # This should NOT raise an error about duplicate branches
        # Each variant combination has its own branch scope
        graph = (
            Graph("branch_scoping")
            .input_data(data)
            .variant(scale_factory, [1, 2, 3], names=["s1", "s2", "s3"])
            .branch(["same_name", "another"])  # Same name "same_name" per variant
            .add(identity, branch="same_name")
            .add(identity, branch="another")
            .merge(["same_name", "another"], combiner=lambda sigs: sigs[0])
        )
        
        results = graph.run()
        assert len(results) == 3
        
        # All should succeed without conflicts
        for params, result in results:
            assert result.data[0] in [5, 10, 15]  # Depends on scale factor
