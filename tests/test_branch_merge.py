"""
Pytest unit tests for branch/merge DAG functionality.
"""

import numpy as np
import pytest
from sigexec import Graph, SignalData


class TestBranchDuplicate:
    """Test branch with duplication (no functions)."""
    
    def test_simple_duplicate_branch(self):
        """Test simple branch duplication."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        
        def multiply_by_2(sig):
            return SignalData(sig.data * 2, metadata={'sample_rate': sig.sample_rate})
        
        def multiply_by_3(sig):
            return SignalData(sig.data * 3, metadata={'sample_rate': sig.sample_rate})
        
        def combiner(signals):
            # Add the two branches
            return SignalData(signals[0].data + signals[1].data, 
                             metadata={'sample_rate': signals[0].sample_rate})
        
        result = (Graph()
            .input_data(SignalData(data, metadata={'sample_rate': 1000}))
            .branch(["b1", "b2"])
            .add(multiply_by_2, branch="b1")
            .add(multiply_by_3, branch="b2")
            .merge(["b1", "b2"], combiner=combiner)
            .run()
        )
        
        expected = data * 2 + data * 3  # [5, 10, 15, 20]
        np.testing.assert_array_equal(result.data, expected)
        
    def test_branch_returns_pipeline(self):
        """Test that branch returns self for chaining."""
        p = Graph()
        result = p.branch(["b1", "b2"])
        assert result is p
        
    def test_merge_returns_pipeline(self):
        """Test that merge returns self for chaining."""
        p = Graph()
        p.branch(["b1", "b2"])
        result = p.merge(["b1", "b2"], combiner=lambda sigs: sigs[0])
        assert result is p


class TestBranchWithFunctions:
    """Test branch with functions (no duplication)."""
    
    def test_function_branch(self):
        """Test branch with functions."""
        data = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        
        def extract_amplitude(sig):
            return SignalData(np.abs(sig.data), 
                            metadata={'sample_rate': sig.sample_rate, 'type': 'amplitude'})
        
        def extract_phase(sig):
            return SignalData(np.angle(sig.data), 
                            metadata={'sample_rate': sig.sample_rate, 'type': 'phase'})
        
        def reconstruct(signals):
            amp, phase = signals[0], signals[1]
            reconstructed = amp.data * np.exp(1j * phase.data)
            return SignalData(reconstructed, metadata={'sample_rate': amp.sample_rate})
        
        result = (Graph()
            .input_data(SignalData(data, metadata={'sample_rate': 1000}))
            .branch(labels=["amp", "phase"], functions=[extract_amplitude, extract_phase])
            .merge(["amp", "phase"], combiner=reconstruct)
            .run()
        )
        
        np.testing.assert_array_almost_equal(result.data, data)
        
    def test_branch_with_mismatched_functions(self):
        """Test that branch raises error if functions count doesn't match labels."""
        p = Graph()
        
        with pytest.raises(ValueError, match="must match number of labels"):
            p.branch(labels=["b1", "b2"], functions=[lambda sig: sig])


class TestBranchTargeting:
    """Test adding operations to specific branches."""
    
    def test_add_to_specific_branch(self):
        """Test adding operation to specific branch."""
        data = np.array([10.0])
        
        result = (Graph()
            .input_data(SignalData(data, metadata={'sample_rate': 1000}))
            .branch(["b1", "b2"])
            .add(lambda sig: SignalData(sig.data * 2, metadata=sig.metadata.copy()), branch="b1")
            .add(lambda sig: SignalData(sig.data * 3, metadata=sig.metadata.copy()), branch="b2")
            .merge(["b1", "b2"], combiner=lambda sigs: SignalData(
                sigs[0].data + sigs[1].data, 
                metadata=sigs[0].metadata))
            .run()
        )
        
        # b1: 10*2=20, b2: 10*3=30, sum=50
        assert result.data[0] == 50.0
        
    def test_add_to_nonexistent_branch_raises_error(self):
        """Test that adding to nonexistent branch raises error."""
        p = Graph()
        p.branch(["b1", "b2"])
        
        with pytest.raises(ValueError, match="not active"):
            p.add(lambda sig: sig, branch="nonexistent")


class TestScalarParameters:
    """Test using SignalData for scalar parameters."""
    
    def test_scalar_parameters(self):
        """Test using SignalData for scalar parameters."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        scale_factor = 5.0
        
        def identity(sig):
            return sig
        
        def make_scalar(sig):
            # Extract a "parameter" as a scalar SignalData
            return SignalData(np.array([scale_factor]), 
                             metadata={'sample_rate': 1.0, 'type': 'parameter'})
        
        def scale_by_param(signals):
            data_sig, param_sig = signals[0], signals[1]
            factor = param_sig.data[0]
            scaled = data_sig.data * factor
            return SignalData(scaled, metadata={'sample_rate': data_sig.sample_rate})
        
        result = (Graph()
            .input_data(SignalData(data, metadata={'sample_rate': 1000}))
            .branch(labels=["data", "param"], functions=[identity, make_scalar])
            .add(lambda sig: SignalData(sig.data + 1, metadata={'sample_rate': sig.sample_rate}), 
                 branch="data")
            .merge(["data", "param"], combiner=scale_by_param)
            .run()
        )
        
        expected = (data + 1) * scale_factor
        np.testing.assert_array_equal(result.data, expected)


class TestMergeCombiner:
    """Test merge combiner function behavior."""
    
    def test_combiner_receives_list(self):
        """Test that combiner receives list of SignalData."""
        received_types = []
        
        def inspector_combiner(signals):
            received_types.append(type(signals))
            received_types.append(type(signals[0]))
            return signals[0]
        
        data = SignalData(np.array([1.0]))
        
        (Graph()
            .input_data(data)
            .branch(["b1", "b2"])
            .merge(["b1", "b2"], combiner=inspector_combiner)
            .run()
        )
        
        assert received_types[0] == list
        assert received_types[1] == SignalData
        
    def test_combiner_order_matches_branch_names(self):
        """Test that combiner receives signals in order of branch_names."""
        data = np.array([1.0])
        
        def tag_branch(value):
            def tagger(sig):
                return SignalData(np.array([value]), metadata={'tag': value})
            return tagger
        
        def check_order(signals):
            # Branch order: ["b1", "b2", "b3"]
            assert signals[0].metadata['tag'] == 1
            assert signals[1].metadata['tag'] == 2
            assert signals[2].metadata['tag'] == 3
            return signals[0]
        
        (Graph()
            .input_data(SignalData(data))
            .branch(["b1", "b2", "b3"])
            .add(tag_branch(1), branch="b1")
            .add(tag_branch(2), branch="b2")
            .add(tag_branch(3), branch="b3")
            .merge(["b1", "b2", "b3"], combiner=check_order)
            .run()
        )


class TestBranchCaching:
    """Test that branch operations are cached correctly."""
    
    def test_branch_operations_cached(self):
        """Test that branch operations use caching."""
        call_counts = {'b1': 0, 'b2': 0}
        
        def counting_op(branch_name):
            def op(sig):
                call_counts[branch_name] += 1
                return SignalData(sig.data * 2, metadata=sig.metadata.copy())
            return op
        
        p = Graph()
        p.input_data(SignalData(np.array([1.0])))
        p.branch(["b1", "b2"])
        p.add(counting_op('b1'), branch="b1")
        p.add(counting_op('b2'), branch="b2")
        p.merge(["b1", "b2"], combiner=lambda sigs: sigs[0])
        
        # First run
        p.run()
        assert call_counts == {'b1': 1, 'b2': 1}
        
        # Second run should use cache
        p.run()
        assert call_counts == {'b1': 1, 'b2': 1}  # Still 1 each


class TestBranchErrorHandling:
    """Test error handling in branch/merge."""
    
    def test_merge_nonexistent_branch(self):
        """Test merging nonexistent branch raises error."""
        p = Graph()
        p.branch(["b1", "b2"])
        
        with pytest.raises(ValueError, match="not active"):
            p.merge(["b1", "nonexistent"], combiner=lambda sigs: sigs[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    
    expected = (data + 1) * scale_factor
    print(f"\nResult: {result.data}")
    print(f"Expected: {expected}")
    assert np.allclose(result.data, expected), "Scalar parameter test failed!"
    print("✓ Test passed!\n")


if __name__ == "__main__":
    test_simple_duplicate_branch()
    test_function_branch()
    test_scalar_parameters()
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
