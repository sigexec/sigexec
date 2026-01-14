"""
Port optimization utilities for implicit branching and memory efficiency.

This module provides tools to analyze which port fields are actually accessed
by operations, allowing the graph executor to create optimized port subsets
and enable implicit branching when operations access different fields.

Implemented features:
- Static AST-based detection (including attribute access like `g.data`)
- Runtime instrumentation via a `PortAccessTracker` to record accessed keys
- Utilities to create and merge port subsets
"""

import ast
import inspect
import logging
from typing import Callable, Set, Optional, Dict, Any
from collections.abc import MutableMapping
from .data import GraphData

logger = logging.getLogger(__name__)


class PortAccessTracker(MutableMapping):
    """Dict-like wrapper that records which keys are accessed."""
    def __init__(self, base: Optional[Dict[str, Any]] = None):
        self._base: Dict[str, Any] = dict(base) if base is not None else {}
        self._accessed_keys: Set[str] = set()

    # Mapping protocol
    def __getitem__(self, key):
        self._accessed_keys.add(key)
        return self._base[key]

    def get(self, key, default=None):
        # Access via .get should also be considered a read
        self._accessed_keys.add(key)
        return self._base.get(key, default)

    def __setitem__(self, key, value):
        self._base[key] = value

    def __delitem__(self, key):
        del self._base[key]

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    @property
    def accessed_keys(self) -> Set[str]:
        return set(self._accessed_keys)

    def keys(self):
        return self._base.keys()

    def items(self):
        return self._base.items()

    def values(self):
        return self._base.values()


class PortAnalyzer:
    """Analyzes operations to determine which ports fields they access."""

    @staticmethod
    def analyze_operation_static(operation: Callable) -> Optional[Set[str]]:
        """
        Statically analyze an operation to determine ports access patterns.

        This uses AST parsing to look for both attribute-style accesses (e.g.
        `g.data`) and explicit ports dict access (e.g. `g.ports['meta']` or
        `g.ports.get('meta')`). Returns None if analysis cannot determine the
        fields (e.g., dynamic access).
        """
        try:
            source = inspect.getsource(operation)
            tree = ast.parse(source)
            accessed_keys: Set[str] = set()

            # Try to determine the name of the first argument (e.g. 'g' in `def op(g):`)
            try:
                sig = inspect.signature(operation)
                params = list(sig.parameters.keys())
                first_param = params[0] if params else None
            except Exception:
                first_param = None

            class PortVisitor(ast.NodeVisitor):
                def visit_Attribute(self, node: ast.Attribute):
                    # Matches patterns like `g.data` (attribute access)
                    if isinstance(node.value, ast.Name) and node.attr:
                        # If we were able to determine the function's first parameter
                        # prefer attributes accessed on that object; otherwise fall
                        # back to any attribute access (conservative but practical)
                        if first_param is None or node.value.id == first_param:
                            # Skip infrastructure attributes like 'ports' itself
                            if node.attr != 'ports':
                                accessed_keys.add(node.attr)
                    self.generic_visit(node)

                def visit_Subscript(self, node: ast.Subscript):
                    # Matches g.ports['key']
                    if isinstance(node.value, ast.Attribute):
                        if node.value.attr == 'ports':
                            # Python 3.9+: node.slice may be ast.Constant
                            index = getattr(node.slice, 'value', None)
                            if index is None and hasattr(node.slice, 's'):
                                index = node.slice.s
                            if isinstance(index, str):
                                accessed_keys.add(index)
                    self.generic_visit(node)

                def visit_Call(self, node: ast.Call):
                    # Matches g.ports.get('key')
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr == 'get' and isinstance(node.func.value, ast.Attribute):
                            if node.func.value.attr == 'ports' and node.args:
                                first = node.args[0]
                                key = getattr(first, 'value', None) or getattr(first, 's', None)
                                if isinstance(key, str):
                                    accessed_keys.add(key)
                    self.generic_visit(node)

                def visit_Compare(self, node: ast.Compare):
                    # Matches 'key' in g.ports
                    for op, comparator in zip(node.ops, node.comparators):
                        if isinstance(op, ast.In) and isinstance(comparator, ast.Attribute):
                            if comparator.attr == 'ports':
                                left = getattr(node.left, 'value', None) or getattr(node.left, 's', None)
                                if isinstance(left, str):
                                    accessed_keys.add(left)
                    self.generic_visit(node)

            visitor = PortVisitor()
            visitor.visit(tree)

            return accessed_keys if len(accessed_keys) > 0 else None

        except (OSError, TypeError, SyntaxError):
            return None

    @staticmethod
    def analyze_operation_runtime(operation: Callable, sample_input: GraphData) -> Set[str]:
        """
        Dynamically analyze an operation by running it with an instrumented
        ports mapping that records which keys are read during execution.

        The sample_input should be a small, safe example (Graph._analyze passes
        a tiny sample to avoid expensive computations).
        """
        # Use a shallow copy of ports to avoid mutating the original sample
        base_ports = sample_input.metadata.copy()
        # Also include 'data' port if present
        if 'data' in sample_input.ports:
            base_ports['data'] = sample_input.ports['data']

        tracker = PortAccessTracker(base_ports)

        # Create a copy of the sample GraphData and replace its ports with tracker
        tracked = sample_input.copy()
        # Assign tracker as the ports dict (special-case in GraphData.__setattr__)
        tracked.ports = tracker

        try:
            operation(tracked)
            return tracker.accessed_keys
        except Exception:
            # If operation fails during instrumentation, return empty set
            return set()

    @staticmethod
    def get_operation_metadata_keys(operation: Callable, sample_input: GraphData = None) -> Optional[Set[str]]:
        """
        Determine which metadata (port) keys an operation accesses.

        Tries static analysis first, falls back to runtime analysis if available.
        Returns None if unknown (consumer should assume all keys are needed).
        """
        static_keys = PortAnalyzer.analyze_operation_static(operation)
        op_name = getattr(operation, '__name__', str(operation))
        if static_keys is not None and len(static_keys) > 0:
            logger.debug(f"PortAnalyzer: static analysis determined keys {static_keys} for {op_name}")
            return static_keys

        if sample_input is not None:
            runtime_keys = PortAnalyzer.analyze_operation_runtime(operation, sample_input)
            if len(runtime_keys) > 0:
                logger.debug(f"PortAnalyzer: runtime analysis determined keys {runtime_keys} for {op_name}")
                return runtime_keys
            else:
                logger.debug(f"PortAnalyzer: runtime analysis found no keys for {op_name}")

        logger.debug(f"PortAnalyzer: could not determine accessed keys for {op_name}; falling back to full metadata")
        return None


def create_port_subset(full_ports: Dict[str, Any], keys: Optional[Set[str]]) -> Dict[str, Any]:
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

    return {k: v for k, v in full_ports.items() if k in keys}


def merge_port_subsets(subsets: list[Dict[str, Any]], strategy: str = 'union') -> Dict[str, Any]:
    """
    Merge multiple ports subsets.

    Args:
        subsets: List of ports dictionaries to merge
        strategy: Merge strategy - 'union' (default) or 'intersection')

    Returns:
        Merged ports dictionary
    """
    if not subsets:
        return {}

    if strategy == 'union':
        result: Dict[str, Any] = {}
        for subset in subsets:
            result.update(subset)
        return result

    elif strategy == 'intersection':
        if len(subsets) == 1:
            return subsets[0].copy()
        common_keys = set(subsets[0].keys())
        for subset in subsets[1:]:
            common_keys &= set(subset.keys())
        return {k: subsets[0][k] for k in common_keys}

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")
