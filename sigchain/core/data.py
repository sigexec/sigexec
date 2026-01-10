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
        sample_rate: Sampling rate in Hz
        metadata: Dictionary to store additional information about the signal
    """
    data: np.ndarray
    sample_rate: float
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
    
    def copy(self):
        """Create a deep copy of the SignalData object."""
        return SignalData(
            data=self.data.copy(),
            sample_rate=self.sample_rate,
            metadata=self.metadata.copy()
        )
