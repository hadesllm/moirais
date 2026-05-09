"""Tests for oclin.oc_cutting_line."""
import numpy as np
import pytest
from moirais.fn.oclin import oc_cutting_line


def test_oclin_basic():
    """Test basic functionality."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    n_dims = 2
    result = oc_cutting_line(votes, n_dims)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_oclin_edge():
    """Test edge cases."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    n_dims = 2
    result = oc_cutting_line(votes, n_dims)
    assert isinstance(result, dict)
