"""
Fluent graph interface for data processing.

This module provides a cleaner, more functional approach to building
signal processing graphs using method chaining and lambda functions.
"""

from typing import Callable, Optional, List, Any, Dict, Tuple, Union, Set, Iterable
from collections import OrderedDict
from collections.abc import ItemsView, KeysView, ValuesView, Iterator
import hashlib
import pickle
import numpy as np
from .data import GraphData
from .port_optimizer import (
    PortAnalyzer,
    create_port_subset,
    merge_port_subsets
)


class BranchesView:
    """A small wrapper around an ordered mapping of branch-name -> GraphData.

    Provides both mapping semantics (branches['name']) and sequence semantics
    (branches[0]) so merge functions can treat branch inputs as an ordered set
    without caring about names.
    """
    def __init__(self, ordered: "OrderedDict[str, GraphData]"):
        self._od: "OrderedDict[str, GraphData]" = OrderedDict(ordered)
        self._names: List[str] = list(self._od.keys())

    def __getitem__(self, key: Union[int, str]) -> GraphData:
        """Allow index (0-based) or name lookup."""
        if isinstance(key, int):
            return list(self._od.values())[key]
        return self._od[key]

    def __len__(self) -> int:
        return len(self._od)

    def __iter__(self) -> Iterator[GraphData]:
        return iter(self._od.values())

    def items(self) -> ItemsView:
        return self._od.items()

    def keys(self) -> KeysView:
        return self._od.keys()

    def values(self) -> ValuesView:
        return self._od.values()

    def names(self) -> List[str]:
        return list(self._names)

    def as_list(self) -> List[GraphData]:
        return list(self._od.values())

    def as_dict(self) -> Dict[str, GraphData]:
        return dict(self._od)

    def __repr__(self) -> str:
        return f"BranchesView(names={self.names()})"


class Graph:
    """
    Fluent graph for signal processing with method chaining and memoization.
    
    This class provides a cleaner, more DAG-like interface where you can:
    - Chain operations fluently
    - Use lambda functions for custom operations
    - Modify a single GraphData object through the graph
    - Specify dependencies implicitly through chaining
    - Branch graphs with automatic memoization (shared stages run once)
    - Create variants with different configurations
    
    Memoization means if you create two graphs that share common initial stages,
    those stages only execute once and the result is cached:
    
    Example:
        >>> base = Graph().add(process_a).add(process_b).add(transform)
        >>> branch1 = base.branch().add(filter_hann)
        >>> branch2 = base.branch().add(filter_hamming)
        >>> 
        >>> # When you run branch1, it executes process_a -> process_b -> transform
        >>> result1 = branch1.run()
        >>> 
        >>> # When you run branch2, it reuses the cached result after transform!
        >>> result2 = branch2.run()  # Only executes filter_hamming
    """
    
    # Class-level cache shared across all graph instances
    _global_cache: Dict[str, GraphData] = {}
    
    # Class-level cache for metadata analysis results
    _metadata_analysis_cache: Dict[int, Optional[Set[str]]] = {}
    
    def __init__(
        self, 
        name: str = "Graph", 
        enable_cache: bool = True, 
        input_data: Optional[GraphData] = None,
        optimize_ports: bool = True,
        optimize_ports_strict: bool = False
    ):
        """
        Initialize a new graph.
        
        Args:
            name: Optional name for the graph
            enable_cache: Whether to enable memoization (default: True)
            input_data: Optional initial data to process
            optimize_ports: If True, analyzes operations to determine which ports
                             each operation uses and only passes those ports. Unused ports
                             bypass operations entirely, enabling implicit branching and
                             improving memory efficiency (default: True)
        """
        self.name = name
        self.operations: List[Dict[str, Any]] = []
        self._enable_cache = enable_cache
        self._intermediate_results: List[GraphData] = []
        self._parent_pipeline: Optional['Graph'] = None
        self._input_data = input_data
        self._optimize_ports = optimize_ports
        # If True, enforce that functions with a decorator only access declared ports
        self._optimize_ports_strict = optimize_ports_strict
    
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
    
    def input_data(self, data: GraphData) -> 'Graph':
        """
        Set the input data for the graph.
        
        Args:
            data: GraphData to process
            
        Returns:
            Self for method chaining
            
        Example:
            >>> my_data = GraphData(data_array, sample_rate=20e6)
            >>> result = (Graph()
            ...     .input_data(my_data)
            ...     .add(lambda x: x * 2)  # Double the values
            ...     .add(lambda x: x + 1)  # Add offset
            ...     .run()
            ... )
        """
        self._input_data = data
        return self
    
    def input_variants(
        self,
        signals: List[GraphData],
        names: Optional[List[str]] = None
    ) -> 'Graph':
        """
        Run the same graph over multiple different input signals.
        
        This is a convenience wrapper around .variant() specifically for varying input data.
        Each signal will be processed through the entire graph.
        
        Args:
            signals: List of GraphData objects to process
            names: Optional names for each signal variant
            
        Returns:
            Self for method chaining
            
        Example:
            >>> # Process multiple signals through the same graph
            >>> results = (Graph()
            ...     .input_variants([signal1, signal2, signal3],
            ...                    names=['Dataset A', 'Dataset B', 'Dataset C'])
            ...     .add(lambda x: x * 2)  # Double values
            ...     .add(lambda x: x + 1)  # Add offset
            ...     .run()
            ... )
            >>> # Returns list of (params, result) tuples, one for each input signal
        """
        # Create a factory that returns a function which ignores input and returns the specific signal
        def signal_factory(sig: GraphData) -> Callable[[GraphData], GraphData]:
            return lambda _: sig
        
        return self.variant(signal_factory, signals, names=names)
    
    def add(
        self,
        operation: Callable[[GraphData], GraphData],
        name: Optional[str] = None,
        cacheable: bool = True,
        branch: Optional[str] = None
    ) -> 'Graph':
        """
        Add an operation to the graph.
        
        Args:
            operation: A function that takes GraphData and returns GraphData
            name: Optional name for this operation
            cacheable: Whether this operation's result can be cached
            branch: Optional branch name to add this operation to. If specified,
                   the operation only executes on that branch.
            
        Returns:
            Self for method chaining
        """
        op_name = name or f"Op{len(self.operations)}"
        
        self.operations.append({
            'name': op_name,
            'func': operation,
            'cacheable': cacheable,
            'branch': branch
        })
        return self
    
    def map(
        self,
        func: Callable[[GraphData], GraphData],
        name: Optional[str] = None
    ) -> 'Graph':
        """
        Alias for add() with a more functional name.
        
        Args:
            func: A function that takes GraphData and returns GraphData
            name: Optional name for this operation
            
        Returns:
            Self for method chaining
        """
        return self.add(func, name)
    
    def tap(
        self,
        callback: Callable[[GraphData], None],
        name: Optional[str] = None
    ) -> 'Graph':
        """
        Add a callback that inspects the signal without modifying it.
        
        Useful for debugging, logging, or visualization.
        
        Args:
            callback: Function that receives GraphData but doesn't return anything
            name: Optional name for this operation
            
        Returns:
            Self for method chaining
        """
        def wrapper(signal_data: GraphData) -> GraphData:
            callback(signal_data)
            return signal_data
        
        # Tap operations shouldn't be cached and don't affect port flow
        self.operations.append({
            'func': wrapper,
            'name': name or "tap",
            'cacheable': False,
            'is_tap': True,  # Mark as tap for special handling
            'type': 'operation'
        })
        return self
    
    def transform(
        self,
        data_transformer: Callable[[Any], Any],
        name: Optional[str] = None
    ) -> 'Graph':
        """
        Add a transformation that operates directly on the data array.
        
        Args:
            data_transformer: Function that transforms the data array
            name: Optional name for this operation
            
        Returns:
            Self for method chaining
        """
        def wrapper(signal_data: GraphData) -> GraphData:
            transformed_data = data_transformer(signal_data.data)
            metadata = signal_data.metadata.copy()
            return GraphData(
                data=transformed_data,
                metadata=metadata
            )
        
        return self.add(wrapper, name=name or "transform")
    
    def tap(
        self,
        callback: Callable[[GraphData], None],
        name: Optional[str] = None
    ) -> 'Graph':
        """
        Add a callback that inspects the signal without modifying it.
        
        Useful for debugging, logging, or visualization.
        
        Args:
            callback: Function that receives GraphData but doesn't return anything
            name: Optional name for this operation
            
        Returns:
            Self for method chaining
        """
        def wrapper(signal_data: GraphData) -> GraphData:
            callback(signal_data)
            return signal_data
        
        # Tap operations shouldn't be cached and don't affect port flow
        self.operations.append({
            'func': wrapper,
            'name': name or "tap",
            'cacheable': False,
            'is_tap': True,  # Mark as tap for special handling
            'type': 'operation'
        })
        return self
    
    def variant(
        self,
        operation_factory: Callable[[Any], Callable[[GraphData], GraphData]],
        configs: List[Any],
        names: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> 'Graph':
        """
        Add a variant operation that will be explored with multiple configurations.
        
        Can be chained multiple times to create cartesian product of all variants.
        The actual execution happens when run() is called.
        
        Args:
            operation_factory: Function that takes a config and returns an operation
            configs: List of configurations to try
            names: Optional names for each variant (defaults to config values as strings)
            name: Optional name for the variant operation node (defaults to 'variants')
            
        Returns:
            Self for method chaining
        
        Example:
            >>> results = (Graph()
            ...     .add(process_a).add(process_b)
            ...     .variant(lambda w: lambda x: apply_window(x, w), 
            ...               ['hamming', 'hann'],
            ...               names=['Hamming', 'Hann'],
            ...               name='Window')
            ...     .variant(lambda thresh: lambda x: threshold(x, thresh), 
            ...               [0.1, 0.2],
            ...               names=['Low', 'High'],
            ...               name='Threshold')
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
            'name': name or 'variants',
            'func': None,  # Will be expanded during run
            'cacheable': True,
            'variant_spec': variant_spec
        })
        
        return self
    
    # Alias for more intuitive usage (plural form)
    variants = variant
    
    def _analyze_operation_metadata_usage(
        self,
        op_func: Callable,
        sample_data: Optional[GraphData] = None
    ) -> Optional[Set[str]]:
        """
        Analyze which metadata fields an operation uses.
        
        Uses caching to avoid redundant analysis.
        
        Args:
            op_func: The operation function to analyze
            sample_data: Optional sample data for runtime analysis
            
        Returns:
            Set of metadata keys used, or None if cannot be determined
        """
        # Use function id as cache key
        func_id = id(op_func)
        
        if func_id in Graph._metadata_analysis_cache:
            return Graph._metadata_analysis_cache[func_id]
        
        # For runtime analysis, create a tiny sample to avoid expensive operations
        if sample_data is not None:
            # If the provided sample has a 'data' port, attempt to mimic its shape
            try:
                has_data = 'data' in sample_data.ports
            except Exception:
                has_data = False

            if has_data:
                try:
                    data_shape_len = 1 if sample_data.shape is None else (len(sample_data.shape) if hasattr(sample_data, 'shape') else 1)
                except Exception:
                    data_shape_len = 1

                tiny_data = np.zeros((2, 2) if data_shape_len > 1 else 2)
            else:
                # No data port - use a small 1-D sample to be safe
                tiny_data = np.zeros(2)

            tiny_sample = GraphData(
                data=tiny_data,
                metadata=sample_data.metadata.copy()
            )
            analysis_sample = tiny_sample
        else:
            analysis_sample = None
        
        # Perform analysis
        result = PortAnalyzer.get_operation_metadata_keys(op_func, analysis_sample)
        
        # Cache result
        Graph._metadata_analysis_cache[func_id] = result
        
        return result
    
    def _optimize_ports_for_operation(
        self,
        signal_data: GraphData,
        op_func: Callable,
        verbose: bool = False
    ) -> GraphData:
        """
        Create an optimized version of GraphData with only the metadata fields
        the operation needs.
        
        Args:
            signal_data: The input GraphData
            op_func: The operation function
            verbose: Print optimization info
            
        Returns:
            GraphData with optimized metadata (or original if optimization not possible)
        """
        if not self._optimize_ports:
            return signal_data
        
        # Analyze what metadata keys this operation needs
        needed_keys = self._analyze_operation_metadata_usage(op_func, signal_data)
        
        if needed_keys is None:
            # Cannot determine - use full metadata
            if verbose:
                print(f"    Metadata optimization: Cannot determine, using full metadata")
            return signal_data
        
        if len(needed_keys) == 0:
            # Operation doesn't use metadata
            if verbose:
                print(f"    Metadata optimization: No metadata needed")
            data_val = signal_data.ports.get('data') if hasattr(signal_data, 'ports') else None
            return GraphData(data=data_val, metadata={})
        
        # Create subset
        optimized_metadata = create_port_subset(signal_data.metadata, needed_keys)
        
        if verbose:
            full_size = len(signal_data.metadata)
            opt_size = len(optimized_metadata)
            print(f"    Metadata optimization: {full_size} -> {opt_size} keys ({needed_keys})")
        
        data_val = signal_data.ports.get('data') if hasattr(signal_data, 'ports') else None
        return GraphData(data=data_val, metadata=optimized_metadata)
    
    def _restore_full_metadata(
        self,
        result: GraphData,
        original_metadata: Dict[str, Any]
    ) -> GraphData:
        """
        Restore full metadata to result, merging with any new metadata from operation.
        
        Args:
            result: The result GraphData (may have modified metadata)
            original_metadata: The original full metadata
            
        Returns:
            GraphData with merged metadata
        """
        if not self._optimize_ports:
            return result
        
        # If the operation returned a GraphData, merge its metadata safely
        if isinstance(result, GraphData):
            merged_metadata = original_metadata.copy()
            try:
                merged_metadata.update(result.metadata)
            except Exception:
                # If result.metadata isn't a mapping for any reason, ignore
                pass

            # Safely get 'data' without allowing enforcing trackers to raise
            data_val = None
            if hasattr(result, 'ports'):
                try:
                    data_val = result.ports.get('data')
                except Exception:
                    data_val = None

            return GraphData(data=data_val, metadata=merged_metadata)

        # If the operation returned a raw value (array, scalar, etc.), treat it as data
        return GraphData(data=result, metadata=original_metadata)
    
    def _execute_from_flow_graph(
        self,
        initial_data: Optional[GraphData] = None,
        verbose: bool = False
    ) -> GraphData:
        """
        Execute the graph based on the port flow graph.
        
        This executes operations in order, but passes only the exact ports
        each operation needs, sourced directly from wherever they were produced.
        No filter/restore cycles - true efficient execution.
        
        Args:
            initial_data: Initial GraphData (can be None for generators)
            verbose: Print execution progress
            
        Returns:
            Final GraphData result
        """
        # Determine initial ports
        initial_ports = set(initial_data.ports.keys()) if initial_data else {'data'}
        
        # Build flow graph based on actual initial ports
        flow_graph = self._build_port_flow_graph(initial_ports)
        
        # Map node_id -> GraphData result
        node_results = {}
        
        # Track all available ports and their sources (GraphData containing that port)
        available_ports = {}  # port_name -> GraphData containing that port
        
        # Initialize available ports from initial_data
        if initial_data is not None:
            for port_name in initial_data.ports.keys():
                available_ports[port_name] = initial_data
        
        # Execute each node in order
        for idx, op in enumerate(self.operations):
            # Handle tap operations specially - they don't produce nodes in flow graph
            if op.get('is_tap', False):
                # Execute tap callback on current available ports
                if available_ports:
                    # Build GraphData with all current ports for inspection
                    tap_data = GraphData()
                    for port_name, source_data in available_ports.items():
                        tap_data.set(port_name, source_data.ports.get(port_name))
                    
                    # Execute tap
                    op['func'](tap_data)
                continue
            
            # Find the corresponding node in flow graph
            node = None
            for n in flow_graph['nodes']:
                if n['id'] == f"node{idx}":
                    node = n
                    break
            
            if not node:
                continue
            
            node_id = node['id']
            name = node['name']
            func = node['func']
            consumes = node['consumes']
            
            if verbose:
                print(f"Executing: {name}...")
                if consumes:
                    print(f"  Consumes: {consumes}")
            
            # Build input GraphData with only the ports this operation needs
            input_data = GraphData()
            
            for port_name in consumes:
                if port_name in available_ports:
                    source_data = available_ports[port_name]
                    # Copy the port value from its source
                    input_data.set(port_name, source_data.ports.get(port_name))
            
            # Execute the operation
            result = func(input_data)
            node_results[node_id] = result
            
            # Update available_ports with all ports from result
            for port_name in result.ports.keys():
                available_ports[port_name] = result
            
            if verbose and node['produces']:
                print(f"  Produces: {node['produces']}")
        
        # Return the result from the last node, but merge in all ports from all nodes
        if flow_graph['nodes']:
            last_node_id = flow_graph['nodes'][-1]['id']
            final_result = GraphData()
            
            # Collect all ports from all available_ports (which tracks latest source of each port)
            for port_name, source_data in available_ports.items():
                final_result.set(port_name, source_data.ports.get(port_name))
            
            return final_result
        
        return initial_data if initial_data is not None else GraphData()

    def _execute_with_metadata_optimization(
        self,
        operation: Callable[[GraphData], GraphData],
        signal_data: GraphData,
        verbose: bool = False
    ) -> GraphData:
        """
        Execute an operation with metadata optimization if enabled.
        
        This wraps operation execution to:
        1. Analyze which metadata fields the operation needs
        2. Create an optimized input with only those fields
        3. Execute the operation
        4. Restore the full metadata to the result
        
        Args:
            operation: The operation to execute
            signal_data: The input GraphData (can be None for generators)
            verbose: Print optimization info
            
        Returns:
            Result GraphData with full metadata
        """
        if not self._optimize_ports or signal_data is None:
            # No optimization - execute directly
            return operation(signal_data)
        
        # Save original metadata
        original_metadata = signal_data.metadata.copy()
        
        # Create optimized input
        optimized_input = self._optimize_ports_for_operation(
            signal_data, operation, verbose
        )

        # If strict enforcement is enabled and the operation declared required ports,
        # install an enforcing tracker so any access to undeclared keys raises.
        if getattr(self, '_optimize_ports_strict', False):
            required = getattr(operation, '_required_ports', None)
            if required is not None:
                # Replace ports dict with an enforcing tracker limited to required keys
                from sigexec.core.port_optimizer import EnforcingPortAccessTracker
                # Build base dict with metadata and data (if present)
                base = optimized_input.metadata.copy()
                data_val = optimized_input.ports.get('data') if hasattr(optimized_input, 'ports') else None
                if data_val is not None:
                    base['data'] = data_val
                tracker = EnforcingPortAccessTracker(base, allowed=required | ({'data'} if data_val is not None else set()))
                tracked_input = GraphData(data=data_val)
                tracked_input.ports = tracker
                optimized_input = tracked_input
                if verbose:
                    print(f"    Strict optimization: enforcing access to {required}")
        
        # Execute operation with optimized input
        result = operation(optimized_input)
        
        # Restore full metadata
        return self._restore_full_metadata(result, original_metadata)
    
    def get_intermediate_results(self) -> List[GraphData]:
        """
        Get all intermediate results from the last run.
        
        Returns:
            List of GraphData from each stage
        """
        return self._intermediate_results.copy()
    
    def clear_cache(cls):
        """Clear the global cache for all pipelines. This is a class method."""
        cls._global_cache.clear()
    
    clear_cache = classmethod(clear_cache)
    
    def _run_with_branches(
        self,
        initial_data: Optional[GraphData],
        verbose: bool,
        cache_enabled: bool
    ) -> Dict[str, GraphData]:
        """
        Execute graph with branch support, returning results for each branch.
        
        Returns:
            Dict mapping branch names to their GraphData results
        """
        # Track active branches - starts with "main" branch
        active_branches = {"main": initial_data}
        
        for i, op in enumerate(self.operations):
            op_type = op.get('type')
            
            if op_type == 'branch':
                # Create new branches - each gets a copy of current data or is created
                # by a provided per-branch function.
                branch_names = op['names']
                functions = op.get('functions')
                if verbose:
                    print(f"Creating branches: {branch_names}")
                
                # Get the current data from main branch (or first active branch)
                source_data = next(iter(active_branches.values()))
                
                # Create isolated copies for each new branch
                new_branches = {}
                for idx, name in enumerate(branch_names):
                    if functions and functions[idx] is not None:
                        # Apply function to a copy of the source data
                        src = source_data.copy() if source_data else None
                        new_branches[name] = functions[idx](src)
                    else:
                        new_branches[name] = source_data.copy() if source_data else None
                
                # Replace active branches with new ones
                active_branches = new_branches
                
            elif op_type == 'merge':
                # Merge branches using merge function
                merge_func = op.get('func')
                target_branches = op.get('branches')
                
                # Determine which branches to merge
                if target_branches:
                    branches_to_merge = {
                        k: v for k, v in active_branches.items()
                        if k in target_branches
                    }
                else:
                    branches_to_merge = active_branches
                
                if verbose:
                    print(f"Merging branches {list(branches_to_merge.keys())} with {op.get('name')}")
                
                # Call merge function with a BranchesView (ordered) so the
                # user can index by name (branches['x']) or by position (branches[0]).
                bv = BranchesView(OrderedDict(branches_to_merge))
                merged_data = merge_func(bv)

                # Replace all branches with single "main" branch containing merged result
                active_branches = {"main": merged_data}
                
            else:
                # Regular operation
                func = op.get('func')
                target_branch = op.get('branch')
                
                if target_branch:
                    # Execute only on specific branch
                    if target_branch in active_branches:
                        if verbose:
                            print(f"Executing {op['name']} on branch '{target_branch}'")
                        
                        cache_key = f"{self._get_cache_key(i)}_{target_branch}"
                        
                        if cache_enabled and op.get('cacheable', True) and cache_key in Graph._global_cache:
                            if verbose:
                                print(f"  [CACHED]")
                            active_branches[target_branch] = Graph._global_cache[cache_key]
                        else:
                            active_branches[target_branch] = self._execute_with_metadata_optimization(
                                func, active_branches[target_branch], verbose
                            )
                            if cache_enabled and op.get('cacheable', True):
                                Graph._global_cache[cache_key] = active_branches[target_branch]
                else:
                    # Execute on all active branches
                    if verbose:
                        print(f"Executing {op['name']} on all branches: {list(active_branches.keys())}")
                    
                    for branch_name in list(active_branches.keys()):
                        cache_key = f"{self._get_cache_key(i)}_{branch_name}"
                        
                        if cache_enabled and op.get('cacheable', True) and cache_key in Graph._global_cache:
                            if verbose:
                                print(f"  [{branch_name}: CACHED]")
                            active_branches[branch_name] = Graph._global_cache[cache_key]
                        else:
                            active_branches[branch_name] = self._execute_with_metadata_optimization(
                                func, active_branches[branch_name], verbose
                            )
                            if cache_enabled and op.get('cacheable', True):
                                Graph._global_cache[cache_key] = active_branches[branch_name]
        
        return active_branches



    def run(
        self,
        initial_data: Optional[GraphData] = None,
        verbose: bool = False,
        save_intermediate: bool = False,
        use_cache: Optional[bool] = None,
        on_variant_complete: Optional[Callable[[Dict[str, List[str]], GraphData], None]] = None,
        return_results: bool = True
    ) -> Union[GraphData, List[Tuple[Dict[str, List[str]], GraphData]], Dict[str, GraphData]]:
        """
        Execute the graph with memoization and branch support.
        
        Returns:
        - Single GraphData if no branches or variants
        - Dict[str, GraphData] if branches exist (maps branch names to results)
        - List of (params, result) tuples if variants exist
        
        Args:
            initial_data: Initial signal data (can be None for generators)
            verbose: Print execution progress
            save_intermediate: Save intermediate results for inspection
            use_cache: Override cache setting for this run
            on_variant_complete: Callback for variant completion
            return_results: Whether to accumulate results
            
        Returns:
            GraphData, Dict[branch -> GraphData], or List of variant results
        """
        data = initial_data if initial_data is not None else self._input_data
        cache_enabled = use_cache if use_cache is not None else self._enable_cache
        
        # Check if graph has branches or variants
        has_branches = any(op.get('type') in ('branch', 'merge') for op in self.operations)
        has_variants = any(op.get('variant_spec') for op in self.operations)
        
        if has_branches and has_variants:
            raise NotImplementedError("Branches and variants together not yet supported. Use one or the other.")
        
        if has_branches:
            # Execute with branch support
            branch_results = self._run_with_branches(data, verbose, cache_enabled)
            
            # If only one branch remains, return its data directly
            if len(branch_results) == 1:
                return next(iter(branch_results.values()))
            
            return branch_results
        
        # No branches - check for variants
        variant_ops = [op for op in self.operations if op.get('variant_spec')]
        
        if not variant_ops:
            # Simple sequential execution using flow graph
            if save_intermediate:
                self._intermediate_results = []
            
            # Use flow graph execution for efficiency
            result = self._execute_from_flow_graph(data, verbose)
            
            if save_intermediate and result is not None:
                self._intermediate_results.append(result.copy())
            
            return result

        # Handle variants: produce the cartesian product of all variant configs
        from itertools import product as _product

        variant_specs = [op.get('variant_spec') for op in self.operations if op.get('variant_spec')]
        configs_list = [vs['configs'] for vs in variant_specs]
        names_list = [vs['names'] for vs in variant_specs]

        results = []

        for combo in _product(*configs_list):
            # Build params dict for this combo (variant names are available in names_list)
            params = {'variant': [str(name) for name in combo]}

            # Build expanded operation list for this combo
            expanded_ops = []
            variant_index = 0
            for op in self.operations:
                if op.get('variant_spec'):
                    vs = op['variant_spec']
                    config = combo[variant_index]
                    func = vs['factory'](config)
                    expanded_ops.append({
                        'name': f"variant_{variant_index}",
                        'func': func,
                        'cacheable': op.get('cacheable', True)
                    })
                    variant_index += 1
                else:
                    expanded_ops.append(op)

            # Execute expanded ops sequentially
            current_data = data
            if save_intermediate:
                self._intermediate_results = []

            for i, op in enumerate(expanded_ops):
                if verbose:
                    print(f"[variant] Executing: {op.get('name', i)}...")
                current_data = self._execute_with_metadata_optimization(
                    op['func'], current_data, verbose
                )
                if save_intermediate and current_data is not None:
                    self._intermediate_results.append(current_data.copy())

            # Call callback if present
            if on_variant_complete:
                on_variant_complete(params, current_data)

            if return_results:
                results.append(({'variant': [str(c) for c in combo]}, current_data))

        return results

    
    def run_and_compare(
        self,
        initial_data: Optional[GraphData] = None,
        comparison_func: Optional[Callable[[List[GraphData]], Any]] = None
    ) -> Tuple[GraphData, List[GraphData]]:
        """
        Run graph with intermediate results and optionally compare them.
        
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
    
    def __call__(self, initial_data: Optional[GraphData] = None) -> GraphData:
        """
        Make the graph callable.
        
        Args:
            initial_data: Initial signal data
            
        Returns:
            Final processed GraphData
        """
        return self.run(initial_data)
    
    def __len__(self) -> int:
        """Return the number of operations in the graph."""
        return len(self.operations)
    
    def branch(self, names: Union[str, List[str]], functions: Optional[List[Callable]] = None) -> 'Graph':
        """
        Create one or more branches with isolated port namespaces.
        
        Branches allow parallel execution paths where each path has its own
        isolated port namespace. This prevents port name collisions and allows
        blocks to be written without knowledge of the graph context.
        
        Args:
            names: Branch name(s) - single string or list of strings
            functions: Optional list of functions (one per branch) to generate branch data
                from the source data. If provided, length must match `names`.
            
        Returns:
            Self for chaining
            
        Example:
            >>> graph = (Graph("Process")
            ...     .add(generate_data)
            ...     .branch(["path1", "path2"], functions=[f1, f2])
            ...     .add(process_a, branch="path1")
            ...     .add(process_b, branch="path2")
            ...     .merge())
        """
        if isinstance(names, str):
            names = [names]
        
        if functions is not None and len(functions) != len(names):
            raise ValueError("'functions' length must match 'names' length")
        
        self.operations.append({
            'type': 'branch',
            'names': names,
            'functions': functions
        })
        return self
    
    def merge(
        self,
        merge_func: Optional[Callable[[Dict[str, GraphData]], GraphData]] = None,
        branches: Optional[List[str]] = None,
        *,
        combiner: Optional[Callable[[Dict[str, GraphData]], GraphData]] = None,
        name: Optional[str] = None
    ) -> 'Graph':
        """
        Merge multiple branches by combining their data.

        The merge function receives a BranchesView object which supports both
        mapping (branches['name']) and sequence (branches[0]) access. The
        merge function MUST return a single GraphData object that combines
        the provided branch data.

        Note: **You must explicitly specify which branches to merge** via the
        `branches` parameter (e.g., `branches=['a','b']`). This avoids ambiguity
        in complex pipelines and keeps merges explicit and auditable.

        Args:
            merge_func: Function that takes a BranchesView and returns a single
                        merged GraphData
            branches: List of branch names to merge (required)
            name: Optional name for this merge operation

        Returns:
            Self for chaining

        Example:
            >>> def avg_merge(branches):
            ...     # branches can be indexed by name or position
            ...     arr0 = branches['a'].data
            ...     arr1 = branches[1].data
            ...     result = GraphData()
            ...     result.data = np.mean([arr0, arr1], axis=0)
            ...     return result
            >>>
            >>> graph = (Graph("Process")
            ...     .branch(["a", "b"])
            ...     .add(process_a, branch="a")
            ...     .add(process_b, branch="b")
            ...     .merge(avg_merge, branches=['a','b']))
        """
        # Backwards compatibility: allow calling .merge([branches], combiner=fn)
        # If the caller passed the branches list as the first argument, support it.
        if isinstance(merge_func, (list, tuple)):
            if combiner is None or not callable(combiner):
                raise ValueError("If calling .merge with a branches list as first arg, provide a callable via 'combiner='.")
            branches = list(merge_func)
            merge_func = combiner

        if branches is None:
            raise ValueError("Merge must specify branch names via `branches=[...]` to avoid ambiguity")

        if not callable(merge_func):
            raise ValueError("merge function must be callable")

        self.operations.append({
            'type': 'merge',
            'func': merge_func,
            'branches': list(branches),
            'name': name or f"Merge{len(self.operations)}"
        })
        return self
    
    def __repr__(self) -> str:
        """Return string representation of the graph."""
        num_ops = len(self.operations)
        variant_ops = sum(1 for op in self.operations if op.get('variant_spec'))
        cache_status = " [cached]" if self._enable_cache else ""
        if variant_ops > 0:
            return f"Graph('{self.name}'{cache_status}, ops={num_ops}, variants={variant_ops})"
        return f"Graph('{self.name}'{cache_status}, ops={num_ops})"
    
    def _build_port_flow_graph(self, initial_ports: set) -> Dict[str, Any]:
        """
        Analyze the graph to determine actual port flows between operations.
        
        This performs static analysis to determine which ports each operation
        actually accesses, then builds an execution graph showing the real
        data flow paths including bypass connections.
        
        Args:
            initial_ports: Set of ports available initially (from initial_data)
        
        Returns a dictionary representing the execution graph:
        {
            'nodes': [
                {
                    'id': 'node1',
                    'name': 'Operation Name',
                    'func': callable,
                    'consumes': {'data', 'port1'},  # Ports this operation reads
                    'produces': {'data', 'port2'},  # Ports this operation writes
                },
                ...
            ],
            'edges': [
                {
                    'from': 'node1',
                    'to': 'node2', 
                    'ports': {'data'},  # Ports flowing on this edge
                },
                {
                    'from': 'node1',
                    'to': 'node3',
                    'ports': {'reference_pulse'},  # Bypass edge
                },
                ...
            ]
        }
        """
        nodes = []
        edges = []
        available_ports = initial_ports.copy() if initial_ports else {'data'}  # Start with initial ports
        port_sources = {'data': None}  # Track which node produced each port (None = initial)
        
        for idx, op in enumerate(self.operations):
            op_type = op.get('type', 'operation')
            
            # Handle tap operations - they pass through all data but should be visualized
            if op.get('is_tap', False):
                node_id = f"node{idx}"
                name = op.get('name', f'Op{idx}')
                consumes = available_ports.copy()
                produces = available_ports.copy()
                nodes.append({
                    'id': node_id,
                    'name': name,
                    'func': op.get('func'),
                    'consumes': consumes,
                    'produces': produces,
                    'is_tap': True,
                })
                # Create edges for all consumed ports from their actual sources
                for port in consumes:
                    src = port_sources.get(port)
                    edges.append({
                        'from': src,
                        'to': node_id,
                        'ports': {port},
                    })
                    # Update source for this port
                    port_sources[port] = node_id
                continue
            
            node_id = f"node{idx}"
            name = op.get('name', f'Op{idx}')
            func = op.get('func')
            has_variants = bool(op.get('variant_spec'))
            
            # Handle variant operations specially
            if has_variants:
                # Variant operation - pass through all available ports
                consumes = available_ports.copy()
                produces = available_ports.copy()
                nodes.append({
                    'id': node_id,
                    'name': name,
                    'func': None,
                    'consumes': consumes,
                    'produces': produces,
                    'has_variants': True,
                })
                # Create edges for all consumed ports from their actual sources
                for port in consumes:
                    src = port_sources.get(port)
                    edges.append({
                        'from': src,
                        'to': node_id,
                        'ports': {port},
                    })
                    # Update source for this port
                    port_sources[port] = node_id
                continue
            
            # Skip non-operation types (branch/merge)
            if op_type not in ('operation', None):
                continue
            
            if not func:
                continue
            
            # Determine what ports this operation actually needs via static analysis
            needed_ports = self._analyze_operation_metadata_usage(func, None)
            if needed_ports is None:
                # If analysis fails, assume it needs all available ports
                needed_ports = available_ports.copy()
            
            consumes = needed_ports & available_ports  # Only consume what's available
            
            # Execute with sample data to determine what it produces
            try:
                sample_data = GraphData()
                for port_name in available_ports:
                    if port_name == 'data':
                        sample_data.set(port_name, np.zeros((2, 10), dtype=complex))
                    else:
                        sample_data.set(port_name, np.array([1.0]))
                
                ports_before = set(sample_data.ports.keys())
                result = func(sample_data)
                ports_after = set(result.ports.keys())
                produces = ports_after - ports_before
                
                # If operation accesses 'data', it becomes new source of 'data'
                if 'data' in consumes:
                    produces.add('data')
                    
            except Exception:
                # Assume it produces 'data' at minimum
                produces = {'data'}
            
            # Create node
            has_variants = bool(op.get('variant_spec'))
            nodes.append({
                'id': node_id,
                'name': name,
                'func': func,
                'consumes': consumes,
                'produces': produces,
                'has_variants': has_variants,
            })

            # Create edges based on port sources
            # Group consumed ports by their source
            ports_by_source = {}
            for port in consumes:
                src = port_sources.get(port)
                if src not in ports_by_source:
                    ports_by_source[src] = set()
                ports_by_source[src].add(port)

            for src_node, ports in ports_by_source.items():
                edges.append({
                    'from': src_node,
                    'to': node_id,
                    'ports': ports,
                })

            # Update available ports and sources
            bypass_ports = available_ports - consumes
            available_ports = produces | bypass_ports

            # Always update port_sources for all produced ports
            for port in produces:
                port_sources[port] = node_id
            # For the very first operation, ensure all available ports have a source
            if idx == 0:
                for port in available_ports:
                    if port_sources.get(port) is None:
                        port_sources[port] = node_id
        
        return {'nodes': nodes, 'edges': edges}
    
    def to_mermaid(self, show_ports: bool = True) -> str:
        """
        Generate a Mermaid flowchart diagram based on actual port flow analysis.
        
        Args:
            show_ports: If True, show which ports flow between operations
        
        Returns:
            String containing Mermaid diagram syntax
        """
        # Build the execution graph showing actual port flows
        # Use 'data' as default initial port for visualization
        flow_graph = self._build_port_flow_graph({'data'})
        
        lines = ["```mermaid", "flowchart TD"]
        
        # Add nodes
        for node in flow_graph['nodes']:
            node_id = node['id']
            name = node['name']
            has_variants = node.get('has_variants', False)
            is_tap = node.get('is_tap', False)
            # Use hexagon shape for operations with variants
            if has_variants:
                lines.append(f"    {node_id}{{{{{name}}}}}")
            elif is_tap:
                lines.append(f"    {node_id}>" + name + "<]")  # Parallelogram for tap nodes
            else:
                lines.append(f"    {node_id}[{name}]")

        # Add edges
        for edge in flow_graph['edges']:
            from_node = edge['from']
            to_node = edge['to']
            ports = edge['ports']

            if from_node is None:
                # Skip initial data flow
                continue

            if show_ports and ports:
                port_str = ', '.join(sorted(ports))
                # Determine if this is a direct edge or bypass edge
                # Direct edge: from_node is immediately before to_node
                from_idx = int(from_node.replace('node', ''))
                to_idx = int(to_node.replace('node', ''))
                if to_idx == from_idx + 1:
                    # Direct connection (solid line)
                    lines.append(f"    {from_node} --|{port_str}|--> {to_node}")
                else:
                    # Bypass connection (dotted line)
                    lines.append(f"    {from_node} -.{port_str}.-> {to_node}")
            else:
                lines.append(f"    {from_node} --> {to_node}")

        lines.append("```")
        return '\n'.join(lines)

    
    def visualize(self, filename: str) -> None:
        """
        Save a Mermaid visualization of the graph to a file.
        
        Args:
            filename: Path to save the Mermaid diagram
        """
        mermaid = self.to_mermaid()
        with open(filename, 'w') as f:
            f.write(mermaid + '\n')


def create_graph(
    name: str = "Graph", 
    enable_cache: bool = True, 
    input_data: Optional[GraphData] = None,
    optimize_ports: bool = True
) -> 'Graph':
    """
    Factory function to create a new graph.
    
    Args:
        name: Optional name for the graph
        enable_cache: Whether to enable memoization
        input_data: Optional initial data to process
        optimize_ports: If True, analyzes operations to optimize metadata usage
                          for implicit branching and memory efficiency
        
    Returns:
        A new Graph instance
    """
    return Graph(
        name, 
        enable_cache=enable_cache, 
        input_data=input_data,
        optimize_ports=optimize_ports
    )
