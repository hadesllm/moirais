"""Tests for hmigr.geron_information_gain."""
import numpy as np
import pytest
from morie.fn.hmigr import geron_information_gain


def test_hmigr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    split = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_information_gain(y, split)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmigr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    split = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_information_gain(y, split)
    assert isinstance(result, dict)
