"""Matched filter block for range compression."""

from dataclasses import dataclass
import numpy as np
from scipy import signal as sp_signal
from ..core.data import GraphData


@dataclass
class RangeCompress:
    """
    Performs matched filtering for range compression.
    
    Uses the reference pulse from metadata to perform correlation.
    Supports windowing and oversampling via FFT-based processing.
    """
    window: str = None
    oversample_factor: int = 1
    
    def __call__(self, gdata: GraphData) -> GraphData:
        """
        Apply matched filtering for range compression.
        
        If oversample_factor > 1, uses FFT-based processing with zero-padding.
        Otherwise uses time-domain correlation.
        """
        # Support both 'data' and legacy 'signal' port names
        if gdata.has_port('data'):
            data = gdata.data
        elif gdata.has_port('signal'):
            data = gdata.signal
        else:
            raise AttributeError(
                f"Port 'data' not found. Available ports: {list(gdata.ports.keys())}"
            )

        # Ensure data is 2D (num_pulses x num_samples)
        if hasattr(data, 'ndim') and data.ndim == 1:
            data = data.reshape(1, -1)

        if hasattr(data, 'shape') and len(data.shape) == 2:
            num_pulses, num_samples = data.shape
        else:
            raise ValueError("Input data must be a 2D array of shape (num_pulses, num_samples)")

        # If reference_pulse is missing, try to infer a reasonable default from the data
        if not gdata.has_port('reference_pulse'):
            import warnings
            warnings.warn("Port 'reference_pulse' not found; inferring reference pulse from data")
            # Use the central portion of the first pulse as a crude reference
            pulse_length = min(64, max(1, num_samples // 8))
            start = max(0, num_samples // 2 - pulse_length // 2)
            reference_pulse = data[0, start:start + pulse_length]
        else:
            reference_pulse = gdata.reference_pulse
        
        # Apply window to reference pulse if specified
        if self.window is not None and self.window.lower() != 'none':
            pulse_length = len(reference_pulse)
            if self.window == 'hann':
                window_func = np.hanning(pulse_length)
            elif self.window == 'hamming':
                window_func = np.hamming(pulse_length)
            elif self.window == 'blackman':
                window_func = np.blackman(pulse_length)
            elif self.window == 'bartlett':
                window_func = np.bartlett(pulse_length)
            else:
                import warnings
                warnings.warn(f"Unknown window type '{self.window}', using rectangular window")
                window_func = np.ones(pulse_length)
            reference_pulse = reference_pulse * window_func
        
        num_pulses, num_samples = data.shape
        pulse_length = len(reference_pulse)
        
        if self.oversample_factor > 1:
            # FFT-based processing with oversampling via zero-padding
            nfft = num_samples * self.oversample_factor
            
            filtered_data = np.zeros((num_pulses, num_samples), dtype=data.dtype)
            
            for i in range(num_pulses):
                signal_fft = np.fft.fft(data[i, :], n=nfft)
                
                reference_padded = np.zeros(nfft, dtype=reference_pulse.dtype)
                reference_padded[:len(reference_pulse)] = reference_pulse
                reference_fft = np.fft.fft(reference_padded)
                
                filtered_fft = signal_fft * np.conj(reference_fft)
                filtered_result = np.fft.ifft(filtered_fft)
                
                filtered_data[i, :] = filtered_result[:num_samples]
            
            output_length = num_samples
        else:
            # Time-domain correlation with 'same' mode for consistent indexing
            output_length = num_samples
            
            filtered_data = np.zeros((num_pulses, output_length), dtype=data.dtype)
            
            for i in range(num_pulses):
                # Matched filter: correlation with conjugate of reference
                filtered_data[i, :] = sp_signal.correlate(
                    data[i, :], 
                    np.conj(reference_pulse), 
                    mode='same'
                )
        
        gdata.data = filtered_data
        gdata.range_compressed = True
        gdata.num_range_bins = output_length
        gdata.range_window = self.window
        gdata.range_oversample_factor = self.oversample_factor
        
        return gdata
