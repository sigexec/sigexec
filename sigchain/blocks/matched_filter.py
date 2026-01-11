"""Matched filter block for range compression."""

import numpy as np
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
        Apply matched filtering to compress in range using FFT-based processing.
        
        Performs elementwise multiplication in frequency domain:
        FFT(output) = FFT(received_signal) * conj(FFT(ideal_waveform))
        
        This is the standard approach for radar range compression.
        
        Args:
            signal_data: Input signal data with pulse-stacked data
            
        Returns:
            SignalData with range-compressed data
        """
        data = signal_data.data
        
        # Get reference pulse from metadata (ideal waveform with no noise/delay)
        if 'reference_pulse' in signal_data.metadata:
            reference_pulse = signal_data.metadata['reference_pulse']
        else:
            raise ValueError("Reference pulse not found in metadata")
        
        # Apply matched filter to each pulse (row)
        num_pulses, num_samples = data.shape
        
        # FFT-based matched filtering
        # 1. Take FFT of each pulse (received signal)
        data_fft = np.fft.fft(data, axis=1)
        
        # 2. Take FFT of the ideal reference waveform (no noise, no delay)
        # Pad reference pulse to match the observation window length
        reference_padded = np.zeros(num_samples, dtype=reference_pulse.dtype)
        reference_padded[:len(reference_pulse)] = reference_pulse
        reference_fft = np.fft.fft(reference_padded)
        
        # 3. Elementwise multiplication with conjugate of reference FFT
        filtered_fft = data_fft * np.conj(reference_fft)
        
        # 4. IFFT to get back to time domain
        filtered_data = np.fft.ifft(filtered_fft, axis=1)
        
        # Create output with updated metadata
        metadata = signal_data.metadata.copy()
        metadata['range_compressed'] = True
        metadata['matched_filter_applied'] = True
        metadata['num_range_bins'] = num_samples
        
        return SignalData(
            data=filtered_data,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )
