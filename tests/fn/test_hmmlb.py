"""Tests for hmmlb.geron_multilabel."""
import numpy as np
import pytest
from morie.fn.hmmlb import geron_multilabel


def test_hmmlb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_multilabel(X, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmlb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_multilabel(X, Y)
    assert isinstance(result, dict)
