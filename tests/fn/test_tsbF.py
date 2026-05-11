"""Tests for tsbF.tsb."""
import numpy as np
import pytest
from morie.fn.tsbF import tsb


def test_tsbF_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = tsb(y, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tsbF_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = tsb(y, alpha, beta)
    assert isinstance(result, dict)
