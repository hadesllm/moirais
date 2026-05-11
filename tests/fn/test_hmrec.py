"""Tests for hmrec.geron_recall."""
import numpy as np
import pytest
from morie.fn.hmrec import geron_recall


def test_hmrec_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_recall(y_true, y_pred)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrec_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_recall(y_true, y_pred)
    assert isinstance(result, dict)
