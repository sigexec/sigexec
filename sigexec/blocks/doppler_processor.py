"""Doppler processor block for velocity compression."""

from dataclasses import dataclass
import numpy as np
from ..core.data import SignalData


@dataclass
class DopplerCompress:
    """
    Performs Doppler compression via FFT along the pulse dimension.
    
    Applies FFT across pulses with optional windowing and oversampling
    to create a Range-Doppler map.
    """
    window: str = 'hann'
    oversample_factor: int = 1
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Apply FFT for Doppler compression with optional oversampling."""
        data = signal_data.data
        num_pulses, num_samples = data.shape
        
        # Apply window function if specified
        if self.window != 'none':
            if self.window == 'hann':
                window_func = np.hanning(num_pulses)
            elif self.window == 'hamming':
                window_func = np.hamming(num_pulses)
            elif self.window == 'blackman':
                window_func = np.blackman(num_pulses)
            elif self.window == 'bartlett':
                window_func = np.bartlett(num_pulses)
            else:
                import warnings
                warnings.warn(f"Unknown window type '{self.window}', using rectangular window")
                window_func = np.ones(num_pulses)
            
            windowed_data = data * window_func[:, np.newaxis]
        else:
            windowed_data = data
        
        # Determine FFT length with oversampling
        nfft = num_pulses * self.oversample_factor
        
        # Perform FFT along pulse dimension with zero-padding for oversampling
        range_doppler_map = np.fft.fftshift(
            np.fft.fft(windowed_data, n=nfft, axis=0),
            axes=0
        )
        
        # Calculate Doppler frequency axis
        if 'pulse_repetition_interval' in signal_data.metadata:
            pri = signal_data.metadata['pulse_repetition_interval']
            doppler_freq = np.fft.fftshift(np.fft.fftfreq(nfft, pri))
        else:
            doppler_freq = np.arange(nfft) - nfft // 2
        
        metadata = signal_data.metadata.copy()
        metadata['doppler_compressed'] = True
        metadata['range_doppler_map'] = True
        metadata['doppler_frequencies'] = doppler_freq
        metadata['window_applied'] = self.window
        metadata['doppler_oversample_factor'] = self.oversample_factor
        metadata['num_doppler_bins'] = nfft
        
        return SignalData(
            data=range_doppler_map,
            metadata=metadata
        )
