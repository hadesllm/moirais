"""Tests for brcls.brier_score."""
import numpy as np
import pytest
from moirais.fn.brcls import brier_score


def test_brcls_basic():
    """Test basic functionality."""
    y_prob = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    result = brier_score(y_prob, y_true)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_brcls_edge():
    """Test edge cases."""
    y_prob = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    result = brier_score(y_prob, y_true)
    assert isinstance(result, dict)
