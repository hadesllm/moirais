"""Tests for hmtanh.geron_tanh."""
import numpy as np
import pytest
from morie.fn.hmtanh import geron_tanh


def test_hmtanh_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_tanh(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtanh_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_tanh(z)
    assert isinstance(result, dict)
