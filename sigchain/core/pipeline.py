"""
Fluent pipeline interface for signal processing.

This module provides a cleaner, more functional approach to building
signal processing pipelines using method chaining and lambda functions.
"""

from typing import Callable, Optional, List, Any, Dict
from .data import SignalData
from .block import ProcessingBlock


class Pipeline:
    """
    Fluent pipeline for signal processing with method chaining.
    
    This class provides a cleaner, more DAG-like interface where you can:
    - Chain operations fluently
    - Use lambda functions for custom operations
    - Modify a single SignalData object through the pipeline
    - Specify dependencies implicitly through chaining
    
    Example:
        >>> pipeline = (Pipeline()
        ...     .add(lambda sig: generate_signal(sig))
        ...     .add(lambda sig: stack_pulses(sig))
        ...     .add(lambda sig: matched_filter(sig))
        ...     .add(lambda sig: doppler_fft(sig)))
        >>> result = pipeline.run(initial_data)
    """
    
    def __init__(self, name: str = "Pipeline"):
        """
        Initialize a new pipeline.
        
        Args:
            name: Optional name for the pipeline
        """
        self.name = name
        self.operations: List[Dict[str, Any]] = []
        self._cached_result: Optional[SignalData] = None
    
    def add(
        self,
        operation: Callable[[SignalData], SignalData],
        name: Optional[str] = None
    ) -> 'Pipeline':
        """
        Add an operation to the pipeline.
        
        Args:
            operation: A function that takes SignalData and returns SignalData
            name: Optional name for this operation
            
        Returns:
            Self for method chaining
        """
        op_name = name or f"Op{len(self.operations)}"
        self.operations.append({
            'name': op_name,
            'func': operation
        })
        return self
    
    def add_block(self, block: ProcessingBlock) -> 'Pipeline':
        """
        Add a ProcessingBlock to the pipeline.
        
        Args:
            block: A ProcessingBlock instance
            
        Returns:
            Self for method chaining
        """
        return self.add(block.process, name=block.name)
    
    def map(
        self,
        func: Callable[[SignalData], SignalData],
        name: Optional[str] = None
    ) -> 'Pipeline':
        """
        Alias for add() with a more functional name.
        
        Args:
            func: A function that takes SignalData and returns SignalData
            name: Optional name for this operation
            
        Returns:
            Self for method chaining
        """
        return self.add(func, name)
    
    def transform(
        self,
        data_transformer: Callable[[Any], Any],
        name: Optional[str] = None
    ) -> 'Pipeline':
        """
        Add a transformation that operates directly on the data array.
        
        Args:
            data_transformer: Function that transforms the data array
            name: Optional name for this operation
            
        Returns:
            Self for method chaining
        """
        def wrapper(signal_data: SignalData) -> SignalData:
            transformed_data = data_transformer(signal_data.data)
            return SignalData(
                data=transformed_data,
                sample_rate=signal_data.sample_rate,
                metadata=signal_data.metadata.copy()
            )
        
        return self.add(wrapper, name=name or "transform")
    
    def tap(
        self,
        callback: Callable[[SignalData], None],
        name: Optional[str] = None
    ) -> 'Pipeline':
        """
        Add a callback that inspects the signal without modifying it.
        
        Useful for debugging, logging, or visualization.
        
        Args:
            callback: Function that receives SignalData but doesn't return anything
            name: Optional name for this operation
            
        Returns:
            Self for method chaining
        """
        def wrapper(signal_data: SignalData) -> SignalData:
            callback(signal_data)
            return signal_data
        
        return self.add(wrapper, name=name or "tap")
    
    def run(
        self,
        initial_data: Optional[SignalData] = None,
        verbose: bool = False
    ) -> SignalData:
        """
        Execute the pipeline.
        
        Args:
            initial_data: Initial signal data (can be None for generators)
            verbose: Print execution progress
            
        Returns:
            Final processed SignalData
        """
        current_data = initial_data
        
        for i, op in enumerate(self.operations):
            if verbose:
                print(f"Executing: {op['name']}...")
            
            current_data = op['func'](current_data)
            
            if verbose and current_data is not None:
                print(f"  Output shape: {current_data.shape}")
        
        self._cached_result = current_data
        return current_data
    
    def __call__(self, initial_data: Optional[SignalData] = None) -> SignalData:
        """
        Make the pipeline callable.
        
        Args:
            initial_data: Initial signal data
            
        Returns:
            Final processed SignalData
        """
        return self.run(initial_data)
    
    def __len__(self) -> int:
        """Return the number of operations in the pipeline."""
        return len(self.operations)
    
    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        op_names = [op['name'] for op in self.operations]
        return f"Pipeline('{self.name}', ops={' -> '.join(op_names)})"


def create_pipeline(name: str = "Pipeline") -> Pipeline:
    """
    Factory function to create a new pipeline.
    
    Args:
        name: Optional name for the pipeline
        
    Returns:
        A new Pipeline instance
    """
    return Pipeline(name)
