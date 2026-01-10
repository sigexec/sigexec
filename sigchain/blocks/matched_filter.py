"""Matched filter block for range compression."""

import numpy as np
from scipy import signal
from ..core.block import ProcessingBlock
from ..core.data import SignalData


class MatchedFilter(ProcessingBlock):
    """
    Performs matched filtering (range compression) on radar data.
    
    This block correlates the received signal with the transmitted waveform
    to compress the signal in range and improve SNR.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the matched filter.
        
        Args:
            name: Optional name for the block
        """
        super().__init__(name)
    
    def process(self, signal_data: SignalData) -> SignalData:
        """
        Apply matched filtering to compress in range.
        
        Uses 'valid' mode correlation which returns only the portion where
        the filter and signal fully overlap. For M input samples and N filter
        samples, output has M-N+1 samples.
        
        Args:
            signal_data: Input signal data with pulse-stacked data
            
        Returns:
            SignalData with range-compressed data
        """
        data = signal_data.data
        
        # Get reference pulse from metadata (conjugate for matched filter)
        if 'reference_pulse' in signal_data.metadata:
            reference_pulse = signal_data.metadata['reference_pulse']
        else:
            raise ValueError("Reference pulse not found in metadata")
        
        # Matched filter is conjugated version of reference
        # Note: scipy.signal.correlate already does time-reversal internally,
        # so we only need to conjugate, not time-reverse
        matched_filter = np.conj(reference_pulse)
        
        # Apply matched filter to each pulse (row)
        num_pulses, num_samples = data.shape
        pulse_length = len(reference_pulse)
        
        # Use 'valid' mode: output length = num_samples - pulse_length + 1
        output_length = num_samples - pulse_length + 1
        filtered_data = np.zeros((num_pulses, output_length), dtype=data.dtype)
        
        for i in range(num_pulses):
            # Use scipy's correlate with 'valid' mode
            filtered_data[i, :] = signal.correlate(
                data[i, :], 
                matched_filter, 
                mode='valid'
            )
        
        # Create output with updated metadata
        metadata = signal_data.metadata.copy()
        metadata['range_compressed'] = True
        metadata['matched_filter_applied'] = True
        metadata['num_range_bins'] = output_length
        
        return SignalData(
            data=filtered_data,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )
