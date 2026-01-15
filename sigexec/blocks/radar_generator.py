"""Radar signal generator block."""

import numpy as np
from ..core.data import GraphData


def LFMGenerator(
    num_pulses: int = 128,
    pulse_duration: float = 10e-6,
    pulse_repetition_interval: float = 1e-3,
    sample_rate: float = 10e6,
    bandwidth: float = 5e6,
    carrier_freq: float = 10e9,
    target_delay: float = 20e-6,
    target_doppler: float = 1000.0,
    noise_power: float = 0.1
):
    """
    Generates a Linear Frequency Modulated (LFM) radar signal with delay and Doppler shift.
    
    Returns a function that generates the LFM signal.
    """
    def generate_lfm(gdata: GraphData = None) -> GraphData:
        """Generate the LFM signal."""
        if gdata is None:
            gdata = GraphData()
        
        samples_per_pulse = int(pulse_duration * sample_rate)
        chirp_rate = bandwidth / pulse_duration
        
        # Calculate delay in samples
        delay_samples = int(target_delay * sample_rate)
        
        # Use a fixed observation window independent of target location
        # This provides consistent range extent regardless of where the target is
        samples_per_window = int(5 * samples_per_pulse)
        
        # Time array for a single pulse
        t_pulse = np.arange(samples_per_pulse) / sample_rate
        
        # Generate reference LFM pulse
        phase = np.pi * chirp_rate * t_pulse**2
        reference_pulse = np.exp(1j * phase)
        
        # Initialize output array with extended window
        signal_matrix = np.zeros((num_pulses, samples_per_window), dtype=complex)
        
        # Generate each pulse with Doppler shift
        for pulse_idx in range(num_pulses):
            pulse_time = pulse_idx * pulse_repetition_interval
            doppler_phase = 2 * np.pi * target_doppler * pulse_time
            
            # Place the full pulse at the delayed position
            signal_matrix[pulse_idx, delay_samples:delay_samples + samples_per_pulse] = (
                reference_pulse * np.exp(1j * doppler_phase)
            )
            
            # Add noise
            noise = (np.random.randn(samples_per_window) + 
                     1j * np.random.randn(samples_per_window)) * np.sqrt(noise_power / 2)
            signal_matrix[pulse_idx, :] += noise
        
        result = gdata
        result.data = signal_matrix
        result.sample_rate = sample_rate
        result.num_pulses = num_pulses
        result.pulse_duration = pulse_duration
        result.pulse_repetition_interval = pulse_repetition_interval
        result.bandwidth = bandwidth
        result.carrier_freq = carrier_freq
        result.target_delay = target_delay
        result.target_doppler = target_doppler
        result.samples_per_pulse = samples_per_pulse
        result.samples_per_window = samples_per_window
        result.chirp_rate = chirp_rate
        result.reference_pulse = reference_pulse
        return result
    
    return generate_lfm