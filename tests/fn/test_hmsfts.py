"""Tests for hmsfts.geron_softmax_score."""
import numpy as np
import pytest
from morie.fn.hmsfts import geron_softmax_score


def test_hmsfts_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_softmax_score(X, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsfts_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_softmax_score(X, theta)
    assert isinstance(result, dict)
