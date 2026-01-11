"""Data class for signal processing chain."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class SignalData:
    """
    Data class representing signal data in the processing chain.
    
    All processing blocks take SignalData as input and return SignalData as output.
    
    Attributes:
        data: The signal data array (can be 1D, 2D, or higher dimensional)
        metadata: Dictionary to store additional information about the signal
                  (including 'sample_rate' if applicable)
    """
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure data is a numpy array."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
    
    @property
    def shape(self):
        """Return the shape of the data."""
        return self.data.shape
    
    @property
    def dtype(self):
        """Return the data type of the data."""
        return self.data.dtype
    
    @property
    def sample_rate(self):
        """Return the sample rate from metadata if present."""
        return self.metadata.get('sample_rate', None)
    
    def copy(self):
        """Create a deep copy of the SignalData object."""
        return SignalData(
            data=self.data.copy(),
            metadata=self.metadata.copy()
        )
