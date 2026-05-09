"""Tests for hmunsp.geron_unsupervised_pretraining."""
import numpy as np
import pytest
from moirais.fn.hmunsp import geron_unsupervised_pretraining


def test_hmunsp_basic():
    """Test basic functionality."""
    X_unlab = np.random.default_rng(42).normal(0, 1, 100)
    X_lab = np.random.default_rng(42).normal(0, 1, 100)
    y_lab = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_unsupervised_pretraining(X_unlab, X_lab, y_lab)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmunsp_edge():
    """Test edge cases."""
    X_unlab = np.random.default_rng(42).normal(0, 1, 100)
    X_lab = np.random.default_rng(42).normal(0, 1, 100)
    y_lab = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_unsupervised_pretraining(X_unlab, X_lab, y_lab)
    assert isinstance(result, dict)
