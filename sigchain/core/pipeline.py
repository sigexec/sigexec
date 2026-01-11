"""
Fluent pipeline interface for signal processing.

This module provides a cleaner, more functional approach to building
signal processing pipelines using method chaining and lambda functions.
"""

from typing import Callable, Optional, List, Any, Dict, Tuple, Union
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
    
    def __init__(self, name: str = "Pipeline", enable_cache: bool = True, input_data: Optional[SignalData] = None):
        """
        Initialize a new pipeline.
        
        Args:
            name: Optional name for the pipeline
            enable_cache: Whether to enable memoization (default: True)
            input_data: Optional initial data to process
        """
        self.name = name
        self.operations: List[Dict[str, Any]] = []
        self._enable_cache = enable_cache
        self._intermediate_results: List[SignalData] = []
        self._parent_pipeline: Optional['Pipeline'] = None
        self._input_data = input_data
    
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
    
    def input_data(self, data: SignalData) -> 'Pipeline':
        """
        Set the input data for the pipeline.
        
        Args:
            data: SignalData to process
            
        Returns:
            Self for method chaining
            
        Example:
            >>> my_data = SignalData(data_array, sample_rate=20e6)
            >>> result = (Pipeline()
            ...     .input_data(my_data)
            ...     .add(RangeCompress())
            ...     .add(DopplerCompress())
            ...     .run()
            ... )
        """
        self._input_data = data
        return self
    
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
    ) -> 'Pipeline':
        """
        Add a variant operation that will be explored with multiple configurations.
        
        Can be chained multiple times to create cartesian product of all variants.
        The actual execution happens when run() is called.
        
        Args:
            operation_factory: Function that takes a config and returns an operation
            configs: List of configurations to try
            names: Optional names for each variant (defaults to config values as strings)
            
        Returns:
            Self for method chaining
        
        Example:
            >>> results = (Pipeline()
            ...     .add(gen).add(stack)
            ...     .variants(lambda w: RangeCompress(window=w), 
            ...               ['hamming', 'hann'],
            ...               names=['Hamming', 'Hann'])
            ...     .variants(lambda w: DopplerCompress(window=w), 
            ...               ['hann', 'blackman'],
            ...               names=['Hann', 'Blackman'])
            ...     .run()
            ... )
            >>> # Returns 4 results: all combinations
            >>> # Access with: params['variant'][0], params['variant'][1], etc.
        """
        # Store variant specification
        variant_spec = {
            'type': 'variants',
            'factory': operation_factory,
            'configs': configs,
            'names': names or [str(c) for c in configs]  # Use config values as default names
        }
        
        self.operations.append({
            'name': 'variants',
            'func': None,  # Will be expanded during run
            'cacheable': True,
            'variant_spec': variant_spec
        })
        
        return self
    
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
    ) -> Union[SignalData, List[Tuple[Dict[str, List[str]], SignalData]]]:
        """
        Execute the pipeline with memoization.
        
        If this pipeline shares operations with a previously executed pipeline,
        cached results will be reused.
        
        If variants have been added via .variants(), returns a list of 
        (params_dict, result) tuples with all combinations explored. 
        Otherwise returns a single SignalData result.
        
        Args:
            initial_data: Initial signal data (can be None for generators)
            verbose: Print execution progress
            save_intermediate: Save intermediate results for inspection
            use_cache: Override cache setting for this run (default: use instance setting)
            
        Returns:
            SignalData if no variants, or List of (params, result) tuples if variants exist.
            When variants are present, params is a dict with:
                params['variant']: List of variant names, e.g. ['Hamming', 'Hann']
        """
        # Check if pipeline has any variant operations
        variant_ops = [op for op in self.operations if op.get('variant_spec')]
        
        # Use provided initial_data, or fall back to self._input_data
        data = initial_data if initial_data is not None else self._input_data
        
        if not variant_ops:
            # No variants - normal execution
            current_data = data
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
        
        # Has variants - explore all combinations
        from itertools import product
        
        # Split operations into segments: normal ops before/between/after variants
        segments = []
        current_segment = []
        
        for op in self.operations:
            if op.get('variant_spec'):
                if current_segment:
                    segments.append(('normal', current_segment))
                segments.append(('variant', op['variant_spec']))
                current_segment = []
            else:
                current_segment.append(op)
        
        if current_segment:
            segments.append(('normal', current_segment))
        
        # Extract all variant specs
        variant_specs = [(s[1]['factory'], s[1]['configs'], s[1]['names']) 
                        for s in segments if s[0] == 'variant']
        
        # Generate all combinations
        all_configs = [spec[1] for spec in variant_specs]  # configs lists
        all_names = [spec[2] for spec in variant_specs]    # names lists
        factories = [spec[0] for spec in variant_specs]    # factories
        
        cache_enabled = use_cache if use_cache is not None else self._enable_cache
        results = []
        
        for config_combo in product(*all_configs):
            # Build params dict for this combination
            # Use a list structure: params["variant"] = [name1, name2, ...]
            variant_names = []
            for i, (names, config) in enumerate(zip(all_names, config_combo)):
                config_idx = all_configs[i].index(config)
                variant_names.append(names[config_idx])
            
            params = {"variant": variant_names}
            
            if verbose:
                print(f"\nExecuting combination: {params}")
            
            # Execute pipeline for this combination
            current_data = data
            variant_idx = 0
            op_global_idx = 0
            
            for seg_type, seg_data in segments:
                if seg_type == 'normal':
                    # Execute normal operations
                    for op in seg_data:
                        cache_key = self._get_cache_key(op_global_idx)
                        
                        if cache_enabled and op.get('cacheable', True) and cache_key in Pipeline._global_cache:
                            if verbose:
                                print(f"[CACHED] {op['name']}...")
                            current_data = Pipeline._global_cache[cache_key]
                        else:
                            if verbose:
                                print(f"Executing: {op['name']}...")
                            current_data = op['func'](current_data)
                            if cache_enabled and op.get('cacheable', True):
                                Pipeline._global_cache[cache_key] = current_data
                        
                        op_global_idx += 1
                
                elif seg_type == 'variant':
                    # Execute variant operation with specific config
                    config = config_combo[variant_idx]
                    factory = factories[variant_idx]
                    operation = factory(config)
                    
                    # Create cache key including the variant config AND input data
                    # This ensures different combinations don't incorrectly reuse cached results
                    input_hash = hash(current_data.data.tobytes() if current_data is not None else "none")
                    cache_key_parts = [
                        str(self._get_cache_key(op_global_idx)),
                        str(hash(str(config))),
                        str(input_hash)
                    ]
                    cache_key = '_'.join(cache_key_parts)
                    
                    if cache_enabled and cache_key in Pipeline._global_cache:
                        if verbose:
                            print(f"[CACHED] variant{variant_idx+1}={config}...")
                        current_data = Pipeline._global_cache[cache_key]
                    else:
                        if verbose:
                            print(f"Executing: variant{variant_idx+1}={config}...")
                        current_data = operation(current_data)
                        if cache_enabled:
                            Pipeline._global_cache[cache_key] = current_data
                    
                    variant_idx += 1
                    op_global_idx += 1
            
            results.append((params, current_data))
        
        return results
    
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


def create_pipeline(name: str = "Pipeline", enable_cache: bool = True, input_data: Optional[SignalData] = None) -> Pipeline:
    """
    Factory function to create a new pipeline.
    
    Args:
        name: Optional name for the pipeline
        enable_cache: Whether to enable memoization
        input_data: Optional initial data to process
        
    Returns:
        A new Pipeline instance
    """
    return Pipeline(name, enable_cache=enable_cache, input_data=input_data)
