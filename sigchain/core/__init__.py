"""Core components for the signal processing chain."""

from .data import SignalData
from .pipeline import Pipeline, create_pipeline

__all__ = ['SignalData', 'Pipeline', 'create_pipeline']

