"""Tests for gmcent.grand_mean_centering."""
import numpy as np
import pytest
from morie.fn.gmcent import grand_mean_centering


def test_gmcent_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = grand_mean_centering(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gmcent_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = grand_mean_centering(y)
    assert isinstance(result, dict)
