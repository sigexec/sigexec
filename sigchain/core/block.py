"""Base class for processing blocks."""

from abc import ABC, abstractmethod
from typing import Optional, List
from .data import SignalData


class ProcessingBlock(ABC):
    """
    Abstract base class for all signal processing blocks.
    
    Each block must implement the process() method which takes a SignalData
    object as input and returns a SignalData object as output.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the processing block.
        
        Args:
            name: Optional name for the block
        """
        self.name = name or self.__class__.__name__
        self.inputs: List['ProcessingBlock'] = []
        self.outputs: List['ProcessingBlock'] = []
    
    @abstractmethod
    def process(self, signal_data: SignalData) -> SignalData:
        """
        Process the input signal data.
        
        Args:
            signal_data: Input signal data
            
        Returns:
            Processed signal data
        """
        pass
    
    def __call__(self, signal_data: SignalData) -> SignalData:
        """
        Make the block callable. Calls process() method.
        
        Args:
            signal_data: Input signal data
            
        Returns:
            Processed signal data
        """
        return self.process(signal_data)
    
    def connect(self, next_block: 'ProcessingBlock') -> 'ProcessingBlock':
        """
        Connect this block to another block.
        
        Args:
            next_block: The block to connect to
            
        Returns:
            The next block (for chaining)
        """
        self.outputs.append(next_block)
        next_block.inputs.append(self)
        return next_block
    
    def __repr__(self):
        """Return string representation of the block."""
        return f"{self.__class__.__name__}(name='{self.name}')"
