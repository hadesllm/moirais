"""Tests for egarch.egarch_model."""
import numpy as np
import pytest
from morie.fn.egarch import egarch_model


def test_egarch_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = egarch_model(y, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_egarch_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = egarch_model(y, p, q)
    assert isinstance(result, dict)
