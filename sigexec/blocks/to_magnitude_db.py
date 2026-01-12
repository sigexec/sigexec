"""Magnitude to dB conversion block."""

from dataclasses import dataclass
import numpy as np
from ..core.data import SignalData


@dataclass
class ToMagnitudeDB:
    """
    Converts complex data to magnitude in dB scale.
    """
    floor: float = 1e-10
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Convert to magnitude in dB scale."""
        magnitude_db = 20 * np.log10(np.abs(signal_data.data) + self.floor)
        
        metadata = signal_data.metadata.copy()
        metadata['magnitude_db'] = True
        
        return SignalData(
            data=magnitude_db,
            metadata=metadata
        )
