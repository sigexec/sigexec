"""Signal normalization block."""

from dataclasses import dataclass
import numpy as np
from ..core.data import GraphData


@dataclass
class Normalize:
    """
    Normalizes signal data using various methods.
    """
    method: str = 'max'  # 'max', 'mean', 'std'
    
    def __call__(self, gdata: GraphData) -> GraphData:
        """Normalize the signal data."""
        data = gdata.data
        
        if self.method == 'max':
            normalized = data / np.max(np.abs(data))
        elif self.method == 'mean':
            normalized = (data - np.mean(data)) / np.std(data)
        elif self.method == 'std':
            normalized = data / np.std(data)
        else:
            normalized = data
        
        gdata.data = normalized
        gdata.normalized = self.method
        
        return gdata
