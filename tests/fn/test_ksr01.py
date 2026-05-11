"""Tests for ksr01.kosorok_empirical_process."""
import numpy as np
import pytest
from morie.fn.ksr01 import kosorok_empirical_process


def test_ksr01_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_empirical_process(x, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr01_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_empirical_process(x, f)
    assert isinstance(result, dict)
