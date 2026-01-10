"""Signal processing chain with DAG-based architecture."""

from .core.data import SignalData
from .core.block import ProcessingBlock
from .core.dag import DAG
from .core.pipeline import Pipeline, create_pipeline

__all__ = ['SignalData', 'ProcessingBlock', 'DAG', 'Pipeline', 'create_pipeline']
