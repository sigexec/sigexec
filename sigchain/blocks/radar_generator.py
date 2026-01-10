"""Radar signal generator block."""

import numpy as np
from ..core.block import ProcessingBlock
from ..core.data import SignalData


class RadarGenerator(ProcessingBlock):
    """
    Generates a Linear Frequency Modulated (LFM) radar signal with delay and Doppler shift.
    
    This block generates synthetic radar data simulating a target at a specific range
    and velocity.
    """
    
    def __init__(
        self,
        num_pulses: int = 128,
        pulse_duration: float = 10e-6,  # 10 microseconds
        pulse_repetition_interval: float = 1e-3,  # 1 millisecond
        sample_rate: float = 10e6,  # 10 MHz
        bandwidth: float = 5e6,  # 5 MHz
        carrier_freq: float = 10e9,  # 10 GHz
        target_delay: float = 20e-6,  # 20 microseconds (3 km range)
        target_doppler: float = 1000.0,  # 1 kHz Doppler shift
        noise_power: float = 0.1,  # Noise power relative to signal
        name: str = None
    ):
        """
        Initialize the radar generator.
        
        Args:
            num_pulses: Number of pulses to generate
            pulse_duration: Duration of each pulse in seconds
            pulse_repetition_interval: Time between pulses in seconds
            sample_rate: Sampling rate in Hz
            bandwidth: Chirp bandwidth in Hz
            carrier_freq: Carrier frequency in Hz
            target_delay: Target time delay in seconds
            target_doppler: Target Doppler shift in Hz
            noise_power: Noise power relative to signal power
            name: Optional name for the block
        """
        super().__init__(name)
        self.num_pulses = num_pulses
        self.pulse_duration = pulse_duration
        self.pulse_repetition_interval = pulse_repetition_interval
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.carrier_freq = carrier_freq
        self.target_delay = target_delay
        self.target_doppler = target_doppler
        self.noise_power = noise_power
        
        # Calculate derived parameters
        self.samples_per_pulse = int(pulse_duration * sample_rate)
        self.samples_per_pri = int(pulse_repetition_interval * sample_rate)
        self.chirp_rate = bandwidth / pulse_duration
    
    def generate_lfm_pulse(self, t: np.ndarray) -> np.ndarray:
        """
        Generate an LFM (chirp) pulse.
        
        Args:
            t: Time array
            
        Returns:
            Complex LFM signal
        """
        # LFM signal: exp(j * pi * chirp_rate * t^2)
        phase = np.pi * self.chirp_rate * t**2
        return np.exp(1j * phase)
    
    def process(self, signal_data: SignalData = None) -> SignalData:
        """
        Generate radar signal data.
        
        Args:
            signal_data: Ignored for generator (can be None)
            
        Returns:
            SignalData containing the generated radar signal
        """
        # Time array for a single pulse
        t_pulse = np.arange(self.samples_per_pulse) / self.sample_rate
        
        # Generate reference LFM pulse
        reference_pulse = self.generate_lfm_pulse(t_pulse)
        
        # Initialize output array (num_pulses x samples_per_pulse)
        signal_matrix = np.zeros((self.num_pulses, self.samples_per_pulse), dtype=complex)
        
        # Generate each pulse with Doppler shift
        for pulse_idx in range(self.num_pulses):
            # Time of this pulse
            pulse_time = pulse_idx * self.pulse_repetition_interval
            
            # Doppler phase shift across pulses
            doppler_phase = 2 * np.pi * self.target_doppler * pulse_time
            
            # Calculate delay in samples
            delay_samples = int(self.target_delay * self.sample_rate)
            
            # Create delayed and Doppler-shifted signal
            if delay_samples < self.samples_per_pulse:
                signal_length = self.samples_per_pulse - delay_samples
                signal_matrix[pulse_idx, delay_samples:] = (
                    reference_pulse[:signal_length] * np.exp(1j * doppler_phase)
                )
            
            # Add noise
            noise = (np.random.randn(self.samples_per_pulse) + 
                     1j * np.random.randn(self.samples_per_pulse)) * np.sqrt(self.noise_power / 2)
            signal_matrix[pulse_idx, :] += noise
        
        # Create metadata
        metadata = {
            'num_pulses': self.num_pulses,
            'pulse_duration': self.pulse_duration,
            'pulse_repetition_interval': self.pulse_repetition_interval,
            'bandwidth': self.bandwidth,
            'carrier_freq': self.carrier_freq,
            'target_delay': self.target_delay,
            'target_doppler': self.target_doppler,
            'samples_per_pulse': self.samples_per_pulse,
            'chirp_rate': self.chirp_rate,
            'reference_pulse': reference_pulse,  # Store for matched filtering
        }
        
        return SignalData(
            data=signal_matrix,
            sample_rate=self.sample_rate,
            metadata=metadata
        )
