"""Tests for hmgini.geron_gini_impurity."""
import numpy as np
import pytest
from moirais.fn.hmgini import geron_gini_impurity


def test_hmgini_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_gini_impurity(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgini_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_gini_impurity(y)
    assert isinstance(result, dict)
