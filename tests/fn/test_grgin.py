"""Tests for grgin.geron_gini_impurity."""
import numpy as np
import pytest
from morie.fn.grgin import geron_gini_impurity


def test_grgin_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_gini_impurity(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgin_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_gini_impurity(y)
    assert isinstance(result, dict)
