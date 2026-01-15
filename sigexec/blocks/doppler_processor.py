"""Doppler processor block for velocity compression."""

import numpy as np
from ..core.data import GraphData


def DopplerCompress(window: str = 'hann', oversample_factor: int = 1):
    """
    Performs Doppler compression via FFT along the pulse dimension.
    
    Applies FFT across pulses with optional windowing and oversampling
    to create a Range-Doppler map.
    
    Args:
        window: Window type ('hann', 'hamming', 'blackman', 'bartlett', 'none')
        oversample_factor: Oversampling factor for FFT
    
    Returns:
        A function that performs Doppler compression
    """
    def doppler_compress(gdata: GraphData) -> GraphData:
        """Apply FFT for Doppler compression with optional oversampling."""
        data = gdata.data
        num_pulses, num_samples = data.shape
        
        # Apply window function if specified
        if window != 'none':
            if window == 'hann':
                window_func = np.hanning(num_pulses)
            elif window == 'hamming':
                window_func = np.hamming(num_pulses)
            elif window == 'blackman':
                window_func = np.blackman(num_pulses)
            elif window == 'bartlett':
                window_func = np.bartlett(num_pulses)
            else:
                import warnings
                warnings.warn(f"Unknown window type '{window}', using rectangular window")
                window_func = np.ones(num_pulses)
            
            windowed_data = data * window_func[:, np.newaxis]
        else:
            windowed_data = data
        
        # Determine FFT length with oversampling
        nfft = num_pulses * oversample_factor
        
        # Perform FFT along pulse dimension with zero-padding for oversampling
        range_doppler_map = np.fft.fftshift(
            np.fft.fft(windowed_data, n=nfft, axis=0),
            axes=0
        )
        
        # Calculate Doppler frequency axis
        if hasattr(gdata, 'pulse_repetition_interval'):
            pri = gdata.pulse_repetition_interval
            doppler_freq = np.fft.fftshift(np.fft.fftfreq(nfft, pri))
        else:
            doppler_freq = np.arange(nfft) - nfft // 2
        
        gdata.data = range_doppler_map
        gdata.doppler_compressed = True
        gdata.range_doppler_map = True
        gdata.doppler_frequencies = doppler_freq
        gdata.window_applied = window
        gdata.doppler_oversample_factor = oversample_factor
        gdata.num_doppler_bins = nfft
        
        return gdata
    
    return doppler_compress
