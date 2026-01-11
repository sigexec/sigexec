"""Signal processing chain with pipeline architecture."""

from .core.data import SignalData
from .core.pipeline import Pipeline, create_pipeline

__all__ = ['SignalData', 'Pipeline', 'create_pipeline']

