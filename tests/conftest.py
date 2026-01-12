"""
Pytest configuration and shared fixtures.
"""

import numpy as np
import pytest
from sigexec import SignalData, Graph


@pytest.fixture
def simple_signal():
    """Simple 1D signal for testing."""
    return SignalData(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


@pytest.fixture
def signal_with_metadata():
    """Signal with metadata for testing."""
    return SignalData(
        np.array([1.0, 2.0, 3.0]),
        metadata={'sample_rate': 1000.0, 'units': 'volts'}
    )


@pytest.fixture
def complex_signal():
    """Complex-valued signal for testing."""
    return SignalData(np.array([1+2j, 3+4j, 5+6j]))


@pytest.fixture
def empty_pipeline():
    """Empty graph for testing."""
    return Graph()


@pytest.fixture
def pipeline_with_cache():
    """Graph with caching enabled."""
    return Graph(enable_cache=True)


@pytest.fixture
def pipeline_without_cache():
    """Graph with caching disabled."""
    return Graph(enable_cache=False)


@pytest.fixture(autouse=True)
def clear_pipeline_cache():
    """Clear graph cache before each test."""
    Graph.clear_cache()
    yield
    Graph.clear_cache()
