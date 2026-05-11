"""Tests for hmsftm.geron_softmax_function."""
import numpy as np
import pytest
from morie.fn.hmsftm import geron_softmax_function


def test_hmsftm_basic():
    """Test basic functionality."""
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = geron_softmax_function(scores)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsftm_edge():
    """Test edge cases."""
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = geron_softmax_function(scores)
    assert isinstance(result, dict)
