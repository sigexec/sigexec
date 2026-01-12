"""
Pytest unit tests for core SignalData functionality.
"""

import numpy as np
import pytest
from sigexec.core.data import SignalData


class TestSignalDataCreation:
    """Test SignalData object creation and initialization."""
    
    def test_create_with_array(self):
        """Test creating SignalData with numpy array."""
        data = np.array([1.0, 2.0, 3.0])
        sig = SignalData(data)
        
        assert sig.data is not None
        assert isinstance(sig.data, np.ndarray)
        assert len(sig.data) == 3
        
    def test_create_with_list(self):
        """Test that lists are converted to numpy arrays."""
        sig = SignalData([1.0, 2.0, 3.0])
        
        assert isinstance(sig.data, np.ndarray)
        assert sig.data.tolist() == [1.0, 2.0, 3.0]
        
    def test_create_with_metadata(self):
        """Test creating SignalData with metadata."""
        data = np.array([1.0, 2.0])
        metadata = {'sample_rate': 1000.0, 'units': 'volts'}
        
        sig = SignalData(data, metadata=metadata)
        
        assert sig.metadata == metadata
        assert sig.metadata['sample_rate'] == 1000.0
        assert sig.metadata['units'] == 'volts'
        
    def test_default_empty_metadata(self):
        """Test that default metadata is empty dict."""
        sig = SignalData(np.array([1.0]))
        
        assert sig.metadata == {}
        assert isinstance(sig.metadata, dict)


class TestSignalDataProperties:
    """Test SignalData properties."""
    
    def test_shape_property(self):
        """Test shape property."""
        data = np.array([[1, 2], [3, 4]])
        sig = SignalData(data)
        
        assert sig.shape == (2, 2)
        
    def test_dtype_property(self):
        """Test dtype property."""
        sig_int = SignalData(np.array([1, 2, 3]))
        sig_float = SignalData(np.array([1.0, 2.0, 3.0]))
        sig_complex = SignalData(np.array([1+2j, 3+4j]))
        
        assert sig_int.dtype == np.int64 or sig_int.dtype == np.int32
        assert sig_float.dtype == np.float64
        assert sig_complex.dtype == np.complex128
        
    def test_sample_rate_property(self):
        """Test sample_rate property extracts from metadata."""
        sig_with_sr = SignalData(np.array([1.0]), metadata={'sample_rate': 44100})
        sig_without_sr = SignalData(np.array([1.0]))
        
        assert sig_with_sr.sample_rate == 44100
        assert sig_without_sr.sample_rate is None


class TestSignalDataCopy:
    """Test SignalData copy functionality."""
    
    def test_copy_creates_new_instance(self):
        """Test that copy creates a new instance."""
        data = np.array([1.0, 2.0, 3.0])
        metadata = {'sample_rate': 1000.0}
        sig1 = SignalData(data, metadata=metadata)
        sig2 = sig1.copy()
        
        assert sig2 is not sig1
        assert sig2.data is not sig1.data
        assert sig2.metadata is not sig1.metadata
        
    def test_copy_preserves_data(self):
        """Test that copy preserves data values."""
        data = np.array([1.0, 2.0, 3.0])
        sig1 = SignalData(data)
        sig2 = sig1.copy()
        
        np.testing.assert_array_equal(sig2.data, sig1.data)
        
    def test_copy_is_deep(self):
        """Test that modifying copy doesn't affect original."""
        data = np.array([1.0, 2.0, 3.0])
        metadata = {'sample_rate': 1000.0, 'value': 42}
        sig1 = SignalData(data, metadata=metadata)
        sig2 = sig1.copy()
        
        # Modify copy
        sig2.data[0] = 999.0
        sig2.metadata['value'] = 100
        
        # Original unchanged
        assert sig1.data[0] == 1.0
        assert sig1.metadata['value'] == 42


class TestSignalDataScalarContainer:
    """Test using SignalData for scalar values (parameters)."""
    
    def test_scalar_as_array(self):
        """Test storing scalar as single-element array."""
        sig = SignalData(np.array([42.0]))
        
        assert sig.data.shape == (1,)
        assert sig.data[0] == 42.0
        
    def test_multiple_scalars(self):
        """Test storing multiple parameters."""
        params = SignalData(
            np.array([1000.0, 5e6, 10e9]),  # sample_rate, bandwidth, carrier_freq
            metadata={'type': 'parameters'}
        )
        
        assert params.data.shape == (3,)
        assert params.data[0] == 1000.0
        assert params.metadata['type'] == 'parameters'


class TestSignalDataComplexData:
    """Test SignalData with complex-valued data."""
    
    def test_complex_array(self):
        """Test with complex-valued array."""
        data = np.array([1+2j, 3+4j, 5+6j])
        sig = SignalData(data)
        
        assert sig.dtype == np.complex128
        assert sig.data[0] == 1+2j
        
    def test_amplitude_phase_reconstruction(self):
        """Test storing amplitude and phase separately."""
        original = np.array([1+2j, 3+4j])
        amp = SignalData(np.abs(original), metadata={'type': 'amplitude'})
        phase = SignalData(np.angle(original), metadata={'type': 'phase'})
        
        # Reconstruct
        reconstructed = amp.data * np.exp(1j * phase.data)
        
        np.testing.assert_array_almost_equal(reconstructed, original)


class TestSignalDataMultidimensional:
    """Test SignalData with multidimensional arrays."""
    
    def test_2d_array(self):
        """Test with 2D array (e.g., pulse matrix)."""
        data = np.random.randn(10, 100)  # 10 pulses, 100 samples each
        sig = SignalData(data, metadata={'type': 'pulse_matrix'})
        
        assert sig.shape == (10, 100)
        assert sig.data.ndim == 2
        
    def test_3d_array(self):
        """Test with 3D array."""
        data = np.random.randn(5, 10, 100)
        sig = SignalData(data)
        
        assert sig.shape == (5, 10, 100)
        assert sig.data.ndim == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
