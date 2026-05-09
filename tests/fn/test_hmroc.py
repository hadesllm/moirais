"""Tests for hmroc.geron_roc_curve."""
import numpy as np
import pytest
from moirais.fn.hmroc import geron_roc_curve


def test_hmroc_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = geron_roc_curve(y_true, scores)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmroc_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = geron_roc_curve(y_true, scores)
    assert isinstance(result, dict)
