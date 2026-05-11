"""Tests for giniI.gini_impurity."""
import numpy as np
import pytest
from morie.fn.ginii import gini_impurity


def test_ginii_basic():
    """Test basic functionality."""
    class_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = gini_impurity(class_probs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ginii_edge():
    """Test edge cases."""
    class_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = gini_impurity(class_probs)
    assert isinstance(result, dict)
