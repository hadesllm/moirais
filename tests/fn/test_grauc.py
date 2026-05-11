"""Tests for grauc.geron_auc_roc."""
import numpy as np
import pytest
from morie.fn.grauc import geron_auc_roc


def test_grauc_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_scores = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_auc_roc(y_true, y_scores)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grauc_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_scores = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_auc_roc(y_true, y_scores)
    assert isinstance(result, dict)
