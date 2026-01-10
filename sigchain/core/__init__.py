"""Core components for the signal processing chain."""

from .data import SignalData
from .block import ProcessingBlock
from .dag import DAG

__all__ = ['SignalData', 'ProcessingBlock', 'DAG']
