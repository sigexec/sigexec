"""
Fluent pipeline interface for signal processing.

This module provides a cleaner, more functional approach to building
signal processing pipelines using method chaining and lambda functions.
"""

from typing import Callable, Optional, List, Any, Dict, Tuple
import hashlib
import pickle
from .data import SignalData
from .block import ProcessingBlock


class Pipeline:
    """
    Fluent pipeline for signal processing with method chaining and memoization.
    
    This class provides a cleaner, more DAG-like interface where you can:
    - Chain operations fluently
    - Use lambda functions for custom operations
    - Modify a single SignalData object through the pipeline
    - Specify dependencies implicitly through chaining
    - Branch pipelines with automatic memoization (shared stages run once)
    - Create variants with different configurations
    
    Memoization means if you create two pipelines that share common initial stages,
    those stages only execute once and the result is cached:
    
    Example:
        >>> base = Pipeline().add(gen).add(stack).add(compress_range)
        >>> branch1 = base.branch().add(doppler_hann)
        >>> branch2 = base.branch().add(doppler_hamming)
        >>> 
        >>> # When you run branch1, it executes gen -> stack -> compress_range
        >>> result1 = branch1.run()
        >>> 
        >>> # When you run branch2, it reuses the cached result after compress_range!
        >>> result2 = branch2.run()  # Only executes doppler_hamming
    """
    
    # Class-level cache shared across all pipeline instances
    _global_cache: Dict[str, SignalData] = {}
    
    def __init__(self, name: str = "Pipeline", enable_cache: bool = True):
        """
        Initialize a new pipeline.
        
        Args:
            name: Optional name for the pipeline
            enable_cache: Whether to enable memoization (default: True)
        """
        self.name = name
        self.operations: List[Dict[str, Any]] = []
        self._enable_cache = enable_cache
        self._intermediate_results: List[SignalData] = []
        self._parent_pipeline: Optional['Pipeline'] = None
    
    def _get_cache_key(self, up_to_index: int) -> str:
        """
        Generate a cache key for operations up to a certain index.
        
        Args:
            up_to_index: Index of operations to include in key
            
        Returns:
            Cache key string
        """
        # Create a key based on operation names and their configuration
        # We use operation names as a simple hash
        ops_repr = []
        for i in range(min(up_to_index + 1, len(self.operations))):
            op = self.operations[i]
            # Try to get a repr of the function
            func_name = op.get('name', f"op{i}")
            ops_repr.append(func_name)
        
        key = "->".join(ops_repr)
        return key
    
    def add(
        self,
        operation: Callable[[SignalData], SignalData],
        name: Optional[str] = None,
        cacheable: bool = True
    ) -> 'Pipeline':
        """
        Add an operation to the pipeline.
        
        Args:
            operation: A function that takes SignalData and returns SignalData
            name: Optional name for this operation
            cacheable: Whether this operation's result can be cached
            
        Returns:
            Self for method chaining
        """
        op_name = name or f"Op{len(self.operations)}"
        self.operations.append({
            'name': op_name,
            'func': operation,
            'cacheable': cacheable
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
        
        # Tap operations shouldn't be cached (they're for side effects)
        return self.add(wrapper, name=name or "tap", cacheable=False)
    
    def plot(
        self,
        page=None,
        plotter: Optional[Callable[[SignalData], None]] = None,
        plot_type: Optional[str] = None,
        title: str = "Plot",
        name: Optional[str] = None,
        **plot_kwargs
    ) -> 'Pipeline':
        """
        Add a plotting function that visualizes the signal without modifying it.
        
        Can be used in two ways:
        1. With a staticdash page and plot_type:
           .plot(page, plot_type='timeseries', title="My Plot")
        2. With a custom plotter function:
           .plot(plotter=my_plot_function)
        
        Args:
            page: staticdash Page object to add plot to
            plotter: Custom function that plots SignalData
            plot_type: Type of plot ('timeseries', 'pulse_matrix', 'range_profile', 
                      'range_doppler_map', 'spectrum')
            title: Plot title
            name: Optional name for this operation
            **plot_kwargs: Additional keyword arguments for the plot function
            
        Returns:
            Self for method chaining
        """
        if page is not None and plot_type is not None:
            # Import here to avoid circular dependency
            from ..diagnostics.plot_blocks import (
                add_timeseries_plot,
                add_pulse_matrix_plot,
                add_range_profile_plot,
                add_range_doppler_map_plot,
                add_spectrum_plot,
            )
            
            plot_functions = {
                'timeseries': add_timeseries_plot,
                'pulse_matrix': add_pulse_matrix_plot,
                'range_profile': add_range_profile_plot,
                'range_doppler_map': add_range_doppler_map_plot,
                'spectrum': add_spectrum_plot,
            }
            
            if plot_type not in plot_functions:
                raise ValueError(f"Unknown plot_type '{plot_type}'. Must be one of: {list(plot_functions.keys())}")
            
            plot_func = plot_functions[plot_type]
            
            def plotter_wrapper(signal_data: SignalData) -> None:
                plot_func(page, signal_data, title=title, **plot_kwargs)
            
            return self.tap(plotter_wrapper, name=name or f"plot_{plot_type}")
        
        elif plotter is not None:
            # Use custom plotter function
            return self.tap(plotter, name=name or "plot")
        
        else:
            raise ValueError("Must provide either (page and plot_type) or plotter function")
    
    def branch(self, name: Optional[str] = None) -> 'Pipeline':
        """
        Create a new pipeline branch from the current state.
        
        The new branch shares the operation history and will use memoization
        to avoid re-executing common stages.
        
        Args:
            name: Optional name for the new branch
            
        Returns:
            A new Pipeline that shares operation history with this one
        """
        new_pipeline = Pipeline(
            name=name or f"{self.name}_branch",
            enable_cache=self._enable_cache
        )
        # Copy operations (shared history)
        new_pipeline.operations = self.operations.copy()
        new_pipeline._parent_pipeline = self
        return new_pipeline
    
    def variants(
        self,
        operation_factory: Callable[[Any], Callable[[SignalData], SignalData]],
        configs: List[Any],
        names: Optional[List[str]] = None
    ) -> List[SignalData]:
        """
        Create and execute multiple variants of the pipeline with different configurations.
        
        Uses memoization so common stages only execute once.
        
        Args:
            operation_factory: Function that takes a config and returns an operation
            configs: List of configurations to try
            names: Optional names for each variant
            
        Returns:
            List of results from each variant
        
        Example:
            >>> base = Pipeline().add(gen).add(stack)
            >>> # Try different window functions
            >>> results = base.variants(
            ...     lambda window: DopplerCompress(window=window),
            ...     ['hann', 'hamming', 'blackman']
            ... )
        """
        results = []
        
        for i, config in enumerate(configs):
            name = names[i] if names and i < len(names) else f"Variant{i}"
            
            # Create a branch with the specific configuration
            variant = self.branch(name=name)
            variant.add(operation_factory(config), name=f"{name}_op")
            
            # Execute (will use memoization for common stages)
            result = variant.run()
            results.append(result)
        
        return results
    
    def get_intermediate_results(self) -> List[SignalData]:
        """
        Get all intermediate results from the last run.
        
        Returns:
            List of SignalData from each stage
        """
        return self._intermediate_results.copy()
    
    def clear_cache(cls):
        """Clear the global cache for all pipelines. This is a class method."""
        cls._global_cache.clear()
    
    clear_cache = classmethod(clear_cache)
    
    def run(
        self,
        initial_data: Optional[SignalData] = None,
        verbose: bool = False,
        save_intermediate: bool = False,
        use_cache: Optional[bool] = None
    ) -> SignalData:
        """
        Execute the pipeline with memoization.
        
        If this pipeline shares operations with a previously executed pipeline,
        cached results will be reused.
        
        Args:
            initial_data: Initial signal data (can be None for generators)
            verbose: Print execution progress
            save_intermediate: Save intermediate results for inspection
            use_cache: Override cache setting for this run (default: use instance setting)
            
        Returns:
            Final processed SignalData
        """
        current_data = initial_data
        cache_enabled = use_cache if use_cache is not None else self._enable_cache
        
        if save_intermediate:
            self._intermediate_results = []
        
        for i, op in enumerate(self.operations):
            cache_key = self._get_cache_key(i)
            
            # Check cache if enabled and operation is cacheable
            if cache_enabled and op.get('cacheable', True) and cache_key in Pipeline._global_cache:
                if verbose:
                    print(f"[CACHED] {op['name']}...")
                current_data = Pipeline._global_cache[cache_key]
            else:
                # Execute operation
                if verbose:
                    cache_status = "" if cache_enabled else "[NO CACHE]"
                    print(f"Executing: {op['name']}... {cache_status}")
                
                current_data = op['func'](current_data)
                
                # Cache result if enabled and cacheable
                if cache_enabled and op.get('cacheable', True):
                    Pipeline._global_cache[cache_key] = current_data
            
            if save_intermediate and current_data is not None:
                self._intermediate_results.append(current_data.copy())
            
            if verbose and current_data is not None:
                print(f"  Output shape: {current_data.shape}")
        
        return current_data
    
    def run_and_compare(
        self,
        initial_data: Optional[SignalData] = None,
        comparison_func: Optional[Callable[[List[SignalData]], Any]] = None
    ) -> Tuple[SignalData, List[SignalData]]:
        """
        Run pipeline with intermediate results and optionally compare them.
        
        Args:
            initial_data: Initial signal data
            comparison_func: Optional function to compare intermediate results
            
        Returns:
            Tuple of (final result, list of intermediate results)
        """
        result = self.run(initial_data, save_intermediate=True)
        intermediates = self.get_intermediate_results()
        
        if comparison_func:
            comparison_func(intermediates)
        
        return result, intermediates
    
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
        cache_status = " [cached]" if self._enable_cache else ""
        return f"Pipeline('{self.name}'{cache_status}, ops={' -> '.join(op_names)})"


def create_pipeline(name: str = "Pipeline", enable_cache: bool = True) -> Pipeline:
    """
    Factory function to create a new pipeline.
    
    Args:
        name: Optional name for the pipeline
        enable_cache: Whether to enable memoization
        
    Returns:
        A new Pipeline instance
    """
    return Pipeline(name, enable_cache=enable_cache)
