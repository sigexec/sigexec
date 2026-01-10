"""Signal processing chain with DAG-based architecture."""

from .core.data import SignalData
from .core.block import ProcessingBlock
from .core.dag import DAG

__all__ = ['SignalData', 'ProcessingBlock', 'DAG']
