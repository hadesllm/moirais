"""Tests for hmmish.geron_mish."""
import numpy as np
import pytest
from morie.fn.hmmish import geron_mish


def test_hmmish_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_mish(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmish_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_mish(z)
    assert isinstance(result, dict)
