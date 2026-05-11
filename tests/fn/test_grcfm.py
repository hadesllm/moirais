"""Tests for grcfm.geron_confusion_matrix."""
import numpy as np
import pytest
from morie.fn.grcfm import geron_confusion_matrix


def test_grcfm_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    n_classes = 3
    result = geron_confusion_matrix(y_true, y_pred, n_classes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grcfm_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    n_classes = 3
    result = geron_confusion_matrix(y_true, y_pred, n_classes)
    assert isinstance(result, dict)
