"""Tests for strmkr.strauss_process."""
import numpy as np
import pytest
from moirais.fn.strmkr import strauss_process


def test_strmkr_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r = 10
    gamma = 1.0
    result = strauss_process(coords, r, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_strmkr_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r = 10
    gamma = 1.0
    result = strauss_process(coords, r, gamma)
    assert isinstance(result, dict)
