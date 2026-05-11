"""Tests for cumcif.cumulative_incidence_function."""
import numpy as np
import pytest
from morie.fn.cumcif import cumulative_incidence_function


def test_cumcif_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = cumulative_incidence_function(time, cause)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cumcif_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = cumulative_incidence_function(time, cause)
    assert isinstance(result, dict)
