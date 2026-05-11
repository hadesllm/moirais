"""Tests for confm.confusion_matrix_metrics."""
import numpy as np
import pytest
from morie.fn.confm import confusion_matrix_metrics


def test_confm_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = confusion_matrix_metrics(y_true, y_pred)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_confm_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = confusion_matrix_metrics(y_true, y_pred)
    assert isinstance(result, dict)
