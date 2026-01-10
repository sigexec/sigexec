"""
Signal processing blocks.

NOTE: The blocks in this module are EXAMPLES demonstrating the framework.
They implement radar signal processing but are not core to the framework itself.

Users are encouraged to create their own custom blocks for their specific domains
(audio, medical imaging, communications, etc.). See docs/CUSTOM_BLOCKS.md for guidance.

The framework (SignalData, Pipeline, ProcessingBlock) is in sigchain.core and is
designed to be minimal and domain-agnostic.
"""

from .radar_generator import RadarGenerator
from .pulse_stacker import PulseStacker
from .matched_filter import MatchedFilter
from .doppler_processor import DopplerProcessor

# Functional/data class style blocks
from .functional import (
    LFMGenerator,
    StackPulses,
    RangeCompress,
    DopplerCompress,
    ToMagnitudeDB,
    Normalize
)

__all__ = [
    'RadarGenerator', 'PulseStacker', 'MatchedFilter', 'DopplerProcessor',
    'LFMGenerator', 'StackPulses', 'RangeCompress', 'DopplerCompress',
    'ToMagnitudeDB', 'Normalize'
]
