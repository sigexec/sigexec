"""Doppler processor block for velocity compression."""

import numpy as np
from ..core.block import ProcessingBlock
from ..core.data import SignalData


class DopplerProcessor(ProcessingBlock):
    """
    Performs FFT along the pulse (slow-time) dimension for Doppler compression.
    
    This block applies an FFT across pulses to resolve targets in velocity/Doppler.
    The result is a Range-Doppler Map.
    """
    
    def __init__(self, name: str = None, window: str = 'hann'):
        """
        Initialize the Doppler processor.
        
        Args:
            name: Optional name for the block
            window: Window function to apply ('hann', 'hamming', 'none')
        """
        super().__init__(name)
        self.window = window
    
    def process(self, signal_data: SignalData) -> SignalData:
        """
        Apply FFT along pulse dimension for Doppler processing.
        
        Args:
            signal_data: Input signal data with range-compressed data
            
        Returns:
            SignalData with Range-Doppler Map
        """
        data = signal_data.data
        num_pulses, num_samples = data.shape
        
        # Apply window function if specified
        if self.window != 'none':
            if self.window == 'hann':
                window_func = np.hanning(num_pulses)  # Note: numpy.hann() available in numpy 2.0+
            elif self.window == 'hamming':
                window_func = np.hamming(num_pulses)
            else:
                window_func = np.ones(num_pulses)
            
            # Apply window along pulse dimension
            windowed_data = data * window_func[:, np.newaxis]
        else:
            windowed_data = data
        
        # Perform FFT along pulse (first) dimension
        # fftshift to center zero Doppler
        range_doppler_map = np.fft.fftshift(
            np.fft.fft(windowed_data, axis=0),
            axes=0
        )
        
        # Calculate Doppler frequency axis
        if 'pulse_repetition_interval' in signal_data.metadata:
            pri = signal_data.metadata['pulse_repetition_interval']
            prf = 1.0 / pri  # Pulse Repetition Frequency
            doppler_freq = np.fft.fftshift(np.fft.fftfreq(num_pulses, pri))
        else:
            doppler_freq = np.arange(num_pulses) - num_pulses // 2
        
        # Create output with updated metadata
        metadata = signal_data.metadata.copy()
        metadata['doppler_compressed'] = True
        metadata['range_doppler_map'] = True
        metadata['doppler_frequencies'] = doppler_freq
        metadata['num_doppler_bins'] = num_pulses
        metadata['num_range_bins'] = num_samples
        metadata['window_applied'] = self.window
        
        return SignalData(
            data=range_doppler_map,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )
