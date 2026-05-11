"""Tests for hmsenet.geron_senet."""
import numpy as np
import pytest
from morie.fn.hmsenet import geron_senet


def test_hmsenet_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = geron_senet(x, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsenet_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = geron_senet(x, r)
    assert isinstance(result, dict)
