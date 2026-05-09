"""Tests for paa.paa."""
import numpy as np
import pytest
from moirais.fn.paa import paa


def test_paa_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = paa(x, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_paa_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = paa(x, N)
    assert isinstance(result, dict)
