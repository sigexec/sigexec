"""
Port optimization utilities for implicit branching and memory efficiency.

This module provides tools to analyze which port fields are actually accessed
by operations, allowing the graph executor to create optimized port subsets
and enable implicit branching when operations access different fields.

Note: This is currently a stub implementation. Port optimization will be
implemented to work with GraphData's attribute-based access patterns.
"""

import ast
import inspect
from typing import Callable, Set, Optional, Dict, Any
from .data import GraphData


class PortAnalyzer:
    """Analyzes operations to determine which port fields they access."""
    
    @staticmethod
    def analyze_operation_static(operation: Callable) -> Optional[Set[str]]:
        """
        Statically analyze an operation to determine port access patterns.
        
        Currently returns None (cannot be determined) - to be implemented.
        
        Args:
            operation: The operation to analyze
            
        Returns:
            Set of port keys accessed, or None if cannot be determined statically
        """
        return None
    
    @staticmethod
    def analyze_operation_runtime(operation: Callable, gdata: GraphData) -> Set[str]:
        """
        Run operation and track which ports are accessed.
        
        Currently returns empty set - to be implemented.
        
        Args:
            operation: The operation to analyze
            gdata: GraphData to run operation on
            
        Returns:
            Set of port keys that were accessed
        """
        return set()
    
    @staticmethod
    def get_operation_metadata_keys(operation: Callable, gdata: GraphData = None) -> Optional[Set[str]]:
        """
        Get the port keys accessed by an operation.
        
        Currently returns None - to be implemented.
        
        Args:
            operation: The operation to analyze
            gdata: Optional GraphData for runtime analysis
            
        Returns:
            Set of port keys, or None if cannot be determined
        """
        return None


def create_port_subset(gdata: GraphData, keys: Set[str]) -> GraphData:
    """
    Create a new GraphData with only specified ports.
    
    Currently returns the original GraphData unchanged - to be implemented.
    
    Args:
        gdata: Source GraphData
        keys: Port keys to include
        
    Returns:
        New GraphData with subset of ports
    """
    return gdata


def merge_port_subsets(target: GraphData, source: GraphData) -> GraphData:
    """
    Merge ports from source into target.
    
    Currently returns target unchanged - to be implemented.
    
    Args:
        target: Target GraphData to merge into
        source: Source GraphData to merge from
        
    Returns:
        Target GraphData with merged ports
    """
    return target

    
    def keys(self):
        """Return keys from underlying ports."""
        return self._ports.keys()
    
    def values(self):
        """Return values from underlying ports."""
        return self._ports.values()
    
    def items(self):
        """Return items from underlying ports."""
        return self._ports.items()
    
    @property
    def accessed_keys(self) -> Set[str]:
        """Get the set of keys that were accessed."""
        return self._accessed_keys.copy()


class PortAnalyzer:
    """Analyzes operations to determine which ports fields they access."""
    
    @staticmethod
    def analyze_operation_static(operation: Callable) -> Optional[Set[str]]:
        """
        Statically analyze an operation to determine ports access patterns.
        
        This uses AST parsing to look for ports dictionary accesses.
        Returns None if analysis cannot determine the fields (e.g., dynamic access).
        
        Args:
            operation: The operation to analyze
            
        Returns:
            Set of ports keys accessed, or None if cannot be determined statically
        """
        try:
            # Get source code
            source = inspect.getsource(operation)
            tree = ast.parse(source)
            
            accessed_keys = set()
            
            # Look for patterns like:
            # - signal_data.ports['key']
            # - signal_data.ports.get('key')
            # - 'key' in signal_data.ports
            
            class PortVisitor(ast.NodeVisitor):
                def visit_Subscript(self, node):
                    # Check for ports['key'] pattern
                    if isinstance(node.value, ast.Attribute):
                        if node.value.attr == 'ports':
                            if isinstance(node.slice, ast.Constant):
                                accessed_keys.add(node.slice.value)
                            elif isinstance(node.slice, ast.Str):  # Python 3.7 compatibility
                                accessed_keys.add(node.slice.s)
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    # Check for ports.get('key') pattern
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr == 'get':
                            if isinstance(node.func.value, ast.Attribute):
                                if node.func.value.attr == 'ports':
                                    if node.args and isinstance(node.args[0], ast.Constant):
                                        accessed_keys.add(node.args[0].value)
                                    elif node.args and isinstance(node.args[0], ast.Str):  # Python 3.7
                                        accessed_keys.add(node.args[0].s)
                    self.generic_visit(node)
                
                def visit_Compare(self, node):
                    # Check for 'key' in ports pattern
                    for op, comparator in zip(node.ops, node.comparators):
                        if isinstance(op, ast.In):
                            if isinstance(comparator, ast.Attribute):
                                if comparator.attr == 'ports':
                                    if isinstance(node.left, ast.Constant):
                                        accessed_keys.add(node.left.value)
                                    elif isinstance(node.left, ast.Str):  # Python 3.7
                                        accessed_keys.add(node.left.s)
                    self.generic_visit(node)
            
            visitor = PortVisitor()
            visitor.visit(tree)
            
            # Return None if we found nothing (could be dynamic access or no access)
            # Let runtime analysis handle it
            return accessed_keys if len(accessed_keys) > 0 else None
            
        except (OSError, TypeError, SyntaxError):
            # Cannot get source (built-in, lambda, etc.) or parse error
            return None
    
    @staticmethod
    def analyze_operation_runtime(
        operation: Callable,
        sample_input: GraphData
    ) -> Set[str]:
        """
        Dynamically analyze an operation by running it with a tracking ports dict.
        
        This executes the operation with instrumented ports to see what it accesses.
        
        Args:
            operation: The operation to analyze
            sample_input: Sample GraphData to run the operation on
            
        Returns:
            Set of ports keys that were accessed during execution
        """
        # Create a copy with tracked ports
        tracker = PortAccessTracker(sample_input.ports)
        tracked_signal = GraphData(
            data=sample_input.data.copy(),
            ports=tracker
        )
        
        try:
            # Run the operation
            operation(tracked_signal)
            return tracker.accessed_keys
        except Exception:
            # If operation fails, return empty set (better than crashing)
            return set()
    
    @staticmethod
    def get_operation_ports_keys(
        operation: Callable,
        sample_input: Optional[GraphData] = None
    ) -> Optional[Set[str]]:
        """
        Determine which ports keys an operation accesses.
        
        Tries static analysis first, falls back to runtime analysis if available.
        
        Args:
            operation: The operation to analyze
            sample_input: Optional sample input for runtime analysis
            
        Returns:
            Set of ports keys, or None if cannot be determined
        """
        # Try static analysis first (fast, no execution needed)
        static_keys = PortAnalyzer.analyze_operation_static(operation)
        if static_keys is not None and len(static_keys) > 0:
            return static_keys
        
        # Fall back to runtime analysis if we have sample data
        # This is important for dynamically created functions
        if sample_input is not None:
            runtime_keys = PortAnalyzer.analyze_operation_runtime(operation, sample_input)
            if len(runtime_keys) > 0:
                return runtime_keys
        
        # Cannot determine - assume needs all ports
        return None


def create_port_subset(
    full_ports: Dict[str, Any],
    keys: Optional[Set[str]]
) -> Dict[str, Any]:
    """
    Create a subset of ports containing only the specified keys.
    
    Args:
        full_ports: The complete ports dictionary
        keys: Set of keys to include, or None to include all
        
    Returns:
        Subset dictionary containing only the specified keys that exist
    """
    if keys is None:
        return full_ports.copy()
    
    return {
        k: v for k, v in full_ports.items()
        if k in keys
    }


def merge_port_subsets(
    subsets: list[Dict[str, Any]],
    strategy: str = 'union'
) -> Dict[str, Any]:
    """
    Merge multiple ports subsets.
    
    Args:
        subsets: List of ports dictionaries to merge
        strategy: Merge strategy - 'union' (default) or 'intersection'
        
    Returns:
        Merged ports dictionary
    """
    if not subsets:
        return {}
    
    if strategy == 'union':
        # Union: combine all keys (later values overwrite earlier ones)
        result = {}
        for subset in subsets:
            result.update(subset)
        return result
    
    elif strategy == 'intersection':
        # Intersection: only keep keys that appear in all subsets
        if len(subsets) == 1:
            return subsets[0].copy()
        
        common_keys = set(subsets[0].keys())
        for subset in subsets[1:]:
            common_keys &= set(subset.keys())
        
        # Take values from first subset
        return {k: subsets[0][k] for k in common_keys}
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")
