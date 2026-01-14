"""Signal processing execution graph framework."""

from .core.data import GraphData
# Compatibility alias for older demos and examples
SignalData = GraphData
from .core.graph import Graph, create_graph

__all__ = ['GraphData', 'SignalData', 'Graph', 'create_graph']

