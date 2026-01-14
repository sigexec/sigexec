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
        # Read primary signal from the canonical 'data' port
        data = gdata.data

        # Expect reference_pulse to be provided explicitly
        if not gdata.has_port('reference_pulse'):
            raise ValueError("Port 'reference_pulse' is required")

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
            
            # Use complex dtype for correlation output
            output_dtype = np.complex128 if np.iscomplexobj(data) or np.iscomplexobj(reference_pulse) else np.float64
            filtered_data = np.zeros((num_pulses, num_samples), dtype=output_dtype)
            
            for i in range(num_pulses):
                signal_fft = np.fft.fft(data[i, :], n=nfft)
                
                reference_padded = np.zeros(nfft, dtype=reference_pulse.dtype)
                reference_padded[:len(reference_pulse)] = reference_pulse
                reference_fft = np.fft.fft(reference_padded)
                
                filtered_fft = signal_fft * np.conj(reference_fft)
                filtered_result = np.fft.ifft(filtered_fft)
                
                # Keep as complex or extract real part based on output dtype
                if np.iscomplexobj(filtered_data):
                    filtered_data[i, :] = filtered_result[:num_samples]
                else:
                    filtered_data[i, :] = np.real(filtered_result[:num_samples])
            
            output_length = num_samples
        else:
            # Time-domain correlation with 'same' mode for consistent indexing
            output_length = num_samples
            
            # Correlation output is complex if either input is complex
            output_dtype = np.complex128 if np.iscomplexobj(data) or np.iscomplexobj(reference_pulse) else np.float64
            filtered_data = np.zeros((num_pulses, output_length), dtype=output_dtype)
            
            for i in range(num_pulses):
                # Matched filter: correlation with conjugate of reference
                corr_result = sp_signal.correlate(
                    data[i, :], 
                    np.conj(reference_pulse), 
                    mode='same'
                )
                # Assign directly - dtype already matches
                filtered_data[i, :] = corr_result
        
        gdata.data = filtered_data
        gdata.range_compressed = True
        gdata.num_range_bins = output_length
        gdata.range_window = self.window
        gdata.range_oversample_factor = self.oversample_factor
        
        return gdata
