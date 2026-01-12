"""Pulse stacking block."""

from dataclasses import dataclass
import numpy as np
from ..core.data import SignalData


@dataclass
class StackPulses:
    """
    Organizes received pulses into a 2D matrix for coherent processing.
    
    Ensures pulses are in the correct format (num_pulses x num_samples)
    for subsequent range-Doppler processing.
    """
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Stack pulses into 2D matrix."""
        if signal_data.data.ndim == 1:
            data = signal_data.data.reshape(1, -1)
        else:
            data = signal_data.data
        
        metadata = signal_data.metadata.copy()
        metadata['pulse_stacked'] = True
        metadata['shape_after_stacking'] = data.shape
        
        return SignalData(
            data=data,
            metadata=metadata
        )
