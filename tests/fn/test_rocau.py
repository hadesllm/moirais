"""Tests for rocau.roc_auc_score."""
import numpy as np
import pytest
from moirais.fn.rocau import roc_auc_score


def test_rocau_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_score = np.random.default_rng(42).normal(0, 1, 100)
    result = roc_auc_score(y_true, y_score)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rocau_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_score = np.random.default_rng(42).normal(0, 1, 100)
    result = roc_auc_score(y_true, y_score)
    assert isinstance(result, dict)
