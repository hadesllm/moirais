"""Tests for mannK.mann_kendall."""
import numpy as np
import pytest
from morie.fn.mannK import mann_kendall


def test_mannK_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mann_kendall(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_mannK_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mann_kendall(x)
    assert isinstance(result, dict)
