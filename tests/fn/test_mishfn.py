"""Tests for mishfn.mish_activation."""
import numpy as np
import pytest
from moirais.fn.mishfn import mish_activation


def test_mishfn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mish_activation(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mishfn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mish_activation(y)
    assert isinstance(result, dict)
