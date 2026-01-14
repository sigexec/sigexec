"""
Signal processing blocks.

NOTE: The blocks in this module are EXAMPLES demonstrating the framework.
They implement radar signal processing but are not core to the framework itself.

Users are encouraged to create their own custom blocks for their specific domains
(audio, medical imaging, communications, etc.). See docs/CUSTOM_BLOCKS.md for guidance.

The framework (GraphData, Graph) is in sigexec.core and is designed to be 
minimal and domain-agnostic.
"""

# Import blocks from individual files
from .radar_generator import LFMGenerator
from .pulse_stacker import StackPulses
from .matched_filter import RangeCompress
from .doppler_processor import DopplerCompress
from .to_magnitude_db import ToMagnitudeDB
from .normalize import Normalize
