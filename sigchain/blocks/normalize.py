"""Signal normalization block."""

from dataclasses import dataclass
import numpy as np
from ..core.data import SignalData


@dataclass
class Normalize:
    """
    Normalizes signal data using various methods.
    """
    method: str = 'max'  # 'max', 'mean', 'std'
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Normalize the signal data."""
        data = signal_data.data
        
        if self.method == 'max':
            normalized = data / np.max(np.abs(data))
        elif self.method == 'mean':
            normalized = (data - np.mean(data)) / np.std(data)
        elif self.method == 'std':
            normalized = data / np.std(data)
        else:
            normalized = data
        
        metadata = signal_data.metadata.copy()
        metadata['normalized'] = self.method
        
        return SignalData(
            data=normalized,
            metadata=metadata
        )
