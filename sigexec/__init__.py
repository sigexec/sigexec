"""Signal processing execution graph framework."""

from .core.data import GraphData
from .core.graph import Graph, create_graph
from .core.port_optimizer import PortAnalyzer

# Expose decorator as top-level API
requires_ports = PortAnalyzer.requires_ports

__all__ = ['GraphData', 'Graph', 'create_graph', 'requires_ports']

