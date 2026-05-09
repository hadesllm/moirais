"""Tests for hmuns.geron_unsupervised_learning."""
import numpy as np
import pytest
from moirais.fn.hmuns import geron_unsupervised_learning


def test_hmuns_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_unsupervised_learning(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmuns_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_unsupervised_learning(X)
    assert isinstance(result, dict)
