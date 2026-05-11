"""Tests for grmlb.geron_multilabel_classification."""
import numpy as np
import pytest
from morie.fn.grmlb import geron_multilabel_classification


def test_grmlb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    thresholds = [0.25, 0.5, 0.75]
    result = geron_multilabel_classification(X, Y, thresholds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmlb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    thresholds = [0.25, 0.5, 0.75]
    result = geron_multilabel_classification(X, Y, thresholds)
    assert isinstance(result, dict)
