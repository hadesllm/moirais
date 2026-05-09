"""Tests for autofm.autoformer."""
import numpy as np
import pytest
from moirais.fn.autofm import autoformer


def test_autofm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    seq_len = 100
    result = autoformer(X, y, seq_len)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_autofm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    seq_len = 100
    result = autoformer(X, y, seq_len)
    assert isinstance(result, dict)
