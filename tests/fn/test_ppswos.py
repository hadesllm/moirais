"""Tests for ppswos.pps_without_replacement."""
import numpy as np
import pytest
from moirais.fn.ppswos import pps_without_replacement


def test_ppswos_basic():
    """Test basic functionality."""
    sizes = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = pps_without_replacement(sizes, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ppswos_edge():
    """Test edge cases."""
    sizes = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = pps_without_replacement(sizes, n)
    assert isinstance(result, dict)
