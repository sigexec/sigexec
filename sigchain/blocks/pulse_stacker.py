"""Pulse stacking block."""

import numpy as np
from ..core.block import ProcessingBlock
from ..core.data import SignalData


class PulseStacker(ProcessingBlock):
    """
    Stacks multiple radar pulses into a 2D matrix.
    
    This block organizes the received pulses into a matrix format where each row
    represents a pulse. This is typically the first step in coherent processing.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the pulse stacker.
        
        Args:
            name: Optional name for the block
        """
        super().__init__(name)
    
    def process(self, signal_data: SignalData) -> SignalData:
        """
        Stack pulses into a 2D matrix.
        
        Args:
            signal_data: Input signal data (already in matrix form from generator)
            
        Returns:
            SignalData with pulses stacked (same as input for our case)
        """
        # In a real system, this might gather pulses from a stream
        # For our simulation, the data is already in the correct format
        
        # Ensure data is 2D
        if signal_data.data.ndim == 1:
            # If 1D, reshape into a single pulse
            data = signal_data.data.reshape(1, -1)
        else:
            data = signal_data.data
        
        # Create output with updated metadata
        metadata = signal_data.metadata.copy()
        metadata['pulse_stacked'] = True
        metadata['shape_after_stacking'] = data.shape
        
        return SignalData(
            data=data,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )
