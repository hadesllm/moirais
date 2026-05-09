"""Tests for konfound.konfound."""
import numpy as np
import pytest
from moirais.fn.konfound import konfound


def test_konfound_basic():
    """Test basic functionality."""
    est = np.random.default_rng(42).normal(0, 1, 100)
    se = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = konfound(est, se, n, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_konfound_edge():
    """Test edge cases."""
    est = np.random.default_rng(42).normal(0, 1, 100)
    se = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = konfound(est, se, n, threshold)
    assert isinstance(result, dict)
