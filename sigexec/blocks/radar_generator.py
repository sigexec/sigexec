"""Radar signal generator block."""

from dataclasses import dataclass
import numpy as np
from ..core.data import SignalData


@dataclass
class LFMGenerator:
    """
    Generates a Linear Frequency Modulated (LFM) radar signal with delay and Doppler shift.
    
    All parameters are configured at initialization via dataclass fields.
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
        
        # Use a fixed observation window independent of target location
        # This provides consistent range extent regardless of where the target is
        samples_per_window = int(5 * samples_per_pulse)
        
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
            metadata={
                'sample_rate': self.sample_rate,
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
