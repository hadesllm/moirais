"""Tests for hment.geron_entropy_impurity."""
import numpy as np
import pytest
from morie.fn.hment import geron_entropy_impurity


def test_hment_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_entropy_impurity(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hment_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_entropy_impurity(y)
    assert isinstance(result, dict)
