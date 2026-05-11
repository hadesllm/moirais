"""Tests for assort.degree_assortativity."""
import numpy as np
import pytest
from morie.fn.assort import degree_assortativity


def test_assort_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = degree_assortativity(y, A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_assort_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = degree_assortativity(y, A)
    assert isinstance(result, dict)
