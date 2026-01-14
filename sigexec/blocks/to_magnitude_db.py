"""Magnitude to dB conversion block."""

from dataclasses import dataclass
import numpy as np
from ..core.data import GraphData


@dataclass
class ToMagnitudeDB:
    """
    Converts complex data to magnitude in dB scale.
    """
    floor: float = 1e-10
    
    def __call__(self, gdata: GraphData) -> GraphData:
        """Convert to magnitude in dB scale."""
        magnitude_db = 20 * np.log10(np.abs(gdata.data) + self.floor)
        
        gdata.data = magnitude_db
        gdata.magnitude_db = True
        
        return gdata
