"""Tests for sensSlp.sen_slope."""
import numpy as np
import pytest
from moirais.fn.sensSlp import sen_slope


def test_sensSlp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sen_slope(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sensSlp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sen_slope(x)
    assert isinstance(result, dict)
