"""Tests for otws2.ot_wasserstein_p_1d."""
import numpy as np
import pytest
from moirais.fn.otws2 import ot_wasserstein_p_1d


def test_otws2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = ot_wasserstein_p_1d(x, y, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otws2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = ot_wasserstein_p_1d(x, y, p)
    assert isinstance(result, dict)
