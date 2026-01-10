"""
Enhanced blocks that work as composable data processing units.

Each block is a data class that can be configured and called directly,
always taking and returning SignalData objects.
"""

from dataclasses import dataclass
import numpy as np
from scipy import signal
from ..core.data import SignalData


@dataclass
class LFMGenerator:
    """
    Data class for LFM signal generation configuration.
    
    All parameters are configured at initialization, and the process() method
    generates the signal.
    """
    num_pulses: int = 128
    pulse_duration: float = 10e-6
    pulse_repetition_interval: float = 1e-3
    sample_rate: float = 10e6
    bandwidth: float = 5e6
    carrier_freq: float = 10e9
    target_delay: float = 20e-6
    target_doppler: float = 1000.0
    noise_power: float = 0.1
    
    def __call__(self, signal_data: SignalData = None) -> SignalData:
        """Generate the LFM signal."""
        samples_per_pulse = int(self.pulse_duration * self.sample_rate)
        chirp_rate = self.bandwidth / self.pulse_duration
        
        # Calculate delay in samples
        delay_samples = int(self.target_delay * self.sample_rate)
        
        # Extend the observation window to accommodate delay + full pulse
        # This ensures the full pulse is captured after the delay
        samples_per_window = samples_per_pulse + delay_samples
        
        # Time array for a single pulse
        t_pulse = np.arange(samples_per_pulse) / self.sample_rate
        
        # Generate reference LFM pulse
        phase = np.pi * chirp_rate * t_pulse**2
        reference_pulse = np.exp(1j * phase)
        
        # Initialize output array with extended window
        signal_matrix = np.zeros((self.num_pulses, samples_per_window), dtype=complex)
        
        # Generate each pulse with Doppler shift
        for pulse_idx in range(self.num_pulses):
            pulse_time = pulse_idx * self.pulse_repetition_interval
            doppler_phase = 2 * np.pi * self.target_doppler * pulse_time
            
            # Place the full pulse at the delayed position
            signal_matrix[pulse_idx, delay_samples:delay_samples + samples_per_pulse] = (
                reference_pulse * np.exp(1j * doppler_phase)
            )
            
            # Add noise
            noise = (np.random.randn(samples_per_window) + 
                     1j * np.random.randn(samples_per_window)) * np.sqrt(self.noise_power / 2)
            signal_matrix[pulse_idx, :] += noise
        
        return SignalData(
            data=signal_matrix,
            sample_rate=self.sample_rate,
            metadata={
                'num_pulses': self.num_pulses,
                'pulse_duration': self.pulse_duration,
                'pulse_repetition_interval': self.pulse_repetition_interval,
                'bandwidth': self.bandwidth,
                'carrier_freq': self.carrier_freq,
                'target_delay': self.target_delay,
                'target_doppler': self.target_doppler,
                'samples_per_pulse': samples_per_pulse,
                'samples_per_window': samples_per_window,
                'chirp_rate': chirp_rate,
                'reference_pulse': reference_pulse,
            }
        )


@dataclass
class StackPulses:
    """
    Data class for pulse stacking configuration.
    
    Ensures pulses are organized in 2D matrix format.
    """
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Stack pulses into 2D matrix."""
        if signal_data.data.ndim == 1:
            data = signal_data.data.reshape(1, -1)
        else:
            data = signal_data.data
        
        metadata = signal_data.metadata.copy()
        metadata['pulse_stacked'] = True
        metadata['shape_after_stacking'] = data.shape
        
        return SignalData(
            data=data,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


@dataclass
class RangeCompress:
    """
    Data class for range compression via matched filtering.
    
    Uses the reference pulse from metadata to perform correlation.
    """
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """
        Apply matched filtering for range compression using 'valid' mode.
        """
        data = signal_data.data
        
        if 'reference_pulse' not in signal_data.metadata:
            raise ValueError("Reference pulse not found in metadata")
        
        reference_pulse = signal_data.metadata['reference_pulse']
        # Matched filter is conjugated reference
        # Note: scipy.signal.correlate does time-reversal internally
        matched_filter = np.conj(reference_pulse)
        
        num_pulses, num_samples = data.shape
        pulse_length = len(reference_pulse)
        output_length = num_samples - pulse_length + 1
        
        # Use scipy's correlate for matched filtering with 'valid' mode
        from scipy import signal
        filtered_data = np.zeros((num_pulses, output_length), dtype=data.dtype)
        
        for i in range(num_pulses):
            filtered_data[i, :] = signal.correlate(
                data[i, :], 
                matched_filter, 
                mode='valid'
            )
        
        metadata = signal_data.metadata.copy()
        metadata['range_compressed'] = True
        metadata['num_range_bins'] = output_length
        
        return SignalData(
            data=filtered_data,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


@dataclass
class DopplerCompress:
    """
    Data class for Doppler compression via FFT.
    
    Applies FFT along pulse dimension with optional windowing.
    """
    window: str = 'hann'
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Apply FFT for Doppler compression."""
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
                # Unknown window type - use rectangular (no window)
                import warnings
                warnings.warn(f"Unknown window type '{self.window}', using rectangular window")
                window_func = np.ones(num_pulses)
            
            windowed_data = data * window_func[:, np.newaxis]
        else:
            windowed_data = data
        
        # Perform FFT along pulse dimension
        range_doppler_map = np.fft.fftshift(
            np.fft.fft(windowed_data, axis=0),
            axes=0
        )
        
        # Calculate Doppler frequency axis
        if 'pulse_repetition_interval' in signal_data.metadata:
            pri = signal_data.metadata['pulse_repetition_interval']
            doppler_freq = np.fft.fftshift(np.fft.fftfreq(num_pulses, pri))
        else:
            doppler_freq = np.arange(num_pulses) - num_pulses // 2
        
        metadata = signal_data.metadata.copy()
        metadata['doppler_compressed'] = True
        metadata['range_doppler_map'] = True
        metadata['doppler_frequencies'] = doppler_freq
        metadata['window_applied'] = self.window
        
        return SignalData(
            data=range_doppler_map,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


@dataclass
class ToMagnitudeDB:
    """
    Data class for converting complex data to magnitude in dB.
    """
    floor: float = 1e-10
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Convert to magnitude in dB scale."""
        magnitude_db = 20 * np.log10(np.abs(signal_data.data) + self.floor)
        
        metadata = signal_data.metadata.copy()
        metadata['magnitude_db'] = True
        
        return SignalData(
            data=magnitude_db,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )


@dataclass
class Normalize:
    """
    Data class for normalizing signal data.
    """
    method: str = 'max'  # 'max', 'mean', 'std'
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """Normalize the signal data."""
        data = signal_data.data
        
        if self.method == 'max':
            normalized = data / np.max(np.abs(data))
        elif self.method == 'mean':
            normalized = (data - np.mean(data)) / np.std(data)
        elif self.method == 'std':
            normalized = data / np.std(data)
        else:
            normalized = data
        
        metadata = signal_data.metadata.copy()
        metadata['normalized'] = self.method
        
        return SignalData(
            data=normalized,
            sample_rate=signal_data.sample_rate,
            metadata=metadata
        )
