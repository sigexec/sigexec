"""Data class for graph processing with port-based architecture."""

from typing import Dict, Any, Optional, Union
import numpy as np


class GraphData:
    """
    Pure port-based data container for graph processing.
    
    All data flows through named ports. The most common convention is to use
    a 'data' port for the main signal, but this is just convention - you can
    use any port names.
    
    Access ports as attributes: gdata.data, gdata.sample_rate, etc.
    
    This enables automatic port mapping and optimization: the Graph analyzes
    which ports each operation uses and only routes the necessary ports.
    
    Example:
        >>> gdata = GraphData()
        >>> gdata.data = np.array([1, 2, 3])
        >>> gdata.sample_rate = 1000.0
        >>> 
        >>> # Or initialize with data directly
        >>> gdata = GraphData(np.array([1, 2, 3]))
        >>> 
        >>> def process(g):
        ...     sig = g.data  # Read from port - errors if missing
        ...     g.processed = True  # Write to port
        ...     return g
    """
    
    def __init__(self, data: Optional[Union[np.ndarray, list]] = None, metadata: Optional[Dict[str, Any]] = None, **ports):
        """
        Initialize GraphData with optional data and port values.
        
        Args:
            data: Optional numpy array or list to set as 'data' port
            metadata: Optional dict of port values (for backward compatibility)
            **ports: Additional port values to initialize
        """
        object.__setattr__(self, 'ports', {})
        
        # If data is provided, set it on the 'data' port
        if data is not None:
            if isinstance(data, list):
                data = np.array(data)
            self.ports['data'] = data
        
        # If metadata dict is provided, unpack it into ports
        if metadata is not None:
            for key, value in metadata.items():
                self.ports[key] = value
        
        # Set any additional ports
        for key, value in ports.items():
            self.ports[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """Access ports as attributes. Raises AttributeError if port doesn't exist."""
        if name == 'ports':
            # Avoid recursion for the ports dict itself
            return object.__getattribute__(self, 'ports')
        
        ports = object.__getattribute__(self, 'ports')
        if name in ports:
            return ports[name]
        
        raise AttributeError(
            f"Port '{name}' not found. Available ports: {list(ports.keys())}"
        )
    
    def __setattr__(self, name: str, value: Any):
        """Set ports as attributes."""
        if name == 'ports':
            # Special case: setting the ports dict itself
            object.__setattr__(self, name, value)
        else:
            # Everything else goes into ports
            if not hasattr(self, 'ports'):
                object.__setattr__(self, 'ports', {})
            self.ports[name] = value
    
    def get(self, name: str, default=None) -> Any:
        """
        Get a port value with optional default.
        
        Args:
            name: Port name
            default: Value to return if port doesn't exist (default: None)
            
        Returns:
            Port value or default
        """
        return self.ports.get(name, default)
    
    def set(self, name: str, value: Any):
        """
        Explicitly set a port value.
        
        Args:
            name: Port name
            value: Port value
        """
        self.ports[name] = value
    
    def has_port(self, name: str) -> bool:
        """Check if a port exists."""
        return name in self.ports
    
    def require(self, *names: str):
        """
        Ensure required ports exist, raising error if any are missing.
        
        Args:
            *names: Port names to require
            
        Raises:
            ValueError: If any required port is missing
            
        Example:
            >>> gdata.require('signal', 'sample_rate')
        """
        missing = [name for name in names if name not in self.ports]
        if missing:
            raise ValueError(
                f"Required ports missing: {missing}. "
                f"Available ports: {list(self.ports.keys())}"
            )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Return all ports as a metadata dict (for backward compatibility).
        
        Returns:
            Dict of all port values except 'data'
        """
        return {k: v for k, v in self.ports.items() if k != 'data'}
    
    @property
    def shape(self):
        """Get shape of data array."""
        if 'data' in self.ports:
            return self.ports['data'].shape
        raise AttributeError("No 'data' port available")
    
    @property
    def dtype(self):
        """Get dtype of data array."""
        if 'data' in self.ports:
            return self.ports['data'].dtype
        raise AttributeError("No 'data' port available")
    
    def __repr__(self) -> str:
        """String representation showing available ports."""
        port_names = list(self.ports.keys())
        return f"GraphData(ports={port_names})"

    
    def copy(self):
        """Create a deep copy of the GraphData object."""
        new_gdata = GraphData()
        for key, value in self.ports.items():
            if isinstance(value, np.ndarray):
                new_gdata.ports[key] = value.copy()
            elif isinstance(value, dict):
                new_gdata.ports[key] = value.copy()
            elif isinstance(value, list):
                new_gdata.ports[key] = value.copy()
            else:
                new_gdata.ports[key] = value
        return new_gdata

    
    def __repr__(self) -> str:
        """String representation showing available ports."""
        port_names = list(self.ports.keys())
        if len(port_names) > 5:
            port_display = f"{port_names[:5]}... ({len(port_names)} total)"
        else:
            port_display = str(port_names)
        return f"GraphData(ports={port_display})"
