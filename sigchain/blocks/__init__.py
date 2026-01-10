"""Signal processing blocks."""

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
