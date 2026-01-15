"""Pulse stacking block."""

import numpy as np
from ..core.data import GraphData


def StackPulses():
    """
    Organizes received pulses into a 2D matrix for coherent processing.
    
    Ensures pulses are in the correct format (num_pulses x num_samples)
    for subsequent range-Doppler processing.
    
    Returns:
        A function that stacks pulses into 2D matrix
    """
    def stack_pulses(gdata: GraphData) -> GraphData:
        """Stack pulses into 2D matrix."""
        if gdata.data.ndim == 1:
            data = gdata.data.reshape(1, -1)
        else:
            data = gdata.data
        
        gdata.data = data
        gdata.pulse_stacked = True
        gdata.shape_after_stacking = data.shape
        
        return gdata
    
    return stack_pulses
