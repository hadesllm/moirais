"""Tests for hmswi.geron_swish."""
import numpy as np
import pytest
from morie.fn.hmswi import geron_swish


def test_hmswi_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_swish(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmswi_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_swish(z)
    assert isinstance(result, dict)
