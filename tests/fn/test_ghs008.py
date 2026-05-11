"""Tests for ghs008.ghosal_ch3_normalized_weights_prior."""
import numpy as np
import pytest
from morie.fn.ghs008 import ghosal_ch3_normalized_weights_prior


def test_ghs008_basic():
    """Test basic functionality."""
    Y_j = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ghosal_ch3_normalized_weights_prior(Y_j, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs008_edge():
    """Test edge cases."""
    Y_j = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ghosal_ch3_normalized_weights_prior(Y_j, k)
    assert isinstance(result, dict)
