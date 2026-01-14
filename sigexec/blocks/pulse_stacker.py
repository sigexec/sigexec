"""Pulse stacking block."""

from dataclasses import dataclass
import numpy as np
from ..core.data import GraphData


@dataclass
class StackPulses:
    """
    Organizes received pulses into a 2D matrix for coherent processing.
    
    Ensures pulses are in the correct format (num_pulses x num_samples)
    for subsequent range-Doppler processing.
    """
    
    def __call__(self, gdata: GraphData) -> GraphData:
        """Stack pulses into 2D matrix."""
        if gdata.data.ndim == 1:
            data = gdata.data.reshape(1, -1)
        else:
            data = gdata.data
        
        gdata.data = data
        gdata.pulse_stacked = True
        gdata.shape_after_stacking = data.shape
        
        return gdata
