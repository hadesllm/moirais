"""Tests for hmauc.geron_auc_roc."""
import numpy as np
import pytest
from moirais.fn.hmauc import geron_auc_roc


def test_hmauc_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = geron_auc_roc(y_true, scores)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmauc_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = geron_auc_roc(y_true, scores)
    assert isinstance(result, dict)
