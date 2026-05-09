"""Tests for huberw.huber_weight."""
import numpy as np
import pytest
from moirais.fn.huberw import huber_weight


def test_huberw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = huber_weight(y, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_huberw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = huber_weight(y, k)
    assert isinstance(result, dict)
